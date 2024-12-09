/*
The CUDA Implementation of Affine-invariant Template Mutual Matching (ATMM) and Pixel-level Template Selection (PTS).

:Author:
  `Zixuan Chen <http://narcissusex.github.io/>`_
  
:Email:
  chenzx3@mail2.sysu.edu.cn

:Organization:
  Sun Yat-Sen University (SYSU)

:Date: 2021-09-08
*/

#include <cmath>
#include <iostream>
#include <torch/types.h>
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <c10/util/Half.h>

#define CUDA_NUM_THREADS 1024

template<typename scalar_t>
__global__ void forward_ATM_cuda_kernel(
    const scalar_t *__restrict__ query, 
    const scalar_t *__restrict__ temp, 
    scalar_t *__restrict__ match_map, 
    int b, int t, int c, int w, int h, int d, int wh, int cwh) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < b * t * w * h) {
        auto bt = int(idx / wh);
        auto bi = bt / t, ti = bt % t;
        auto wi = int(idx / h) % w, hi = int(idx % h);
        auto qidx = bi * cwh + wi * h + hi;
        scalar_t maxv = -1;
        for (auto dwi = wi - d; dwi <= wi + d; ++ dwi) {
            for (auto dhi = hi - d; dhi <= hi + d; ++ dhi) {
                if (dwi >= 0 && dwi < w && dhi >= 0 && dhi < h) {
                    auto tidx = ti * cwh + dwi * h + dhi;
                    scalar_t sumv = 0;
                    for (auto i = 0; i < c; ++ i) {
                        auto iwh = i * wh;
                        sumv += (query[qidx + iwh] * temp[tidx + iwh]);
                    }
                    maxv = (sumv > maxv) ? sumv : maxv;
                }
            }
        }
        match_map[idx] = maxv;
    }
}

template<typename scalar_t>
__global__ void backward_ATM_cuda_kernel(
    const scalar_t *__restrict__ query, 
    const scalar_t *__restrict__ temp, 
    scalar_t *__restrict__ match_map, 
    int b, int t, int c, int w, int h, int d, int wh, int cwh) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < b * t * w * h) {
        auto bt = int(idx / wh);
        auto bi = bt / t, ti = bt % t;
        auto wi = int(idx / h) % w, hi = int(idx % h);
        auto tidx = ti * cwh + wi * h + hi;
        scalar_t maxv = -1;
        for (auto dwi = wi - d; dwi <= wi + d; ++ dwi) {
            for (auto dhi = hi - d; dhi <= hi + d; ++ dhi) {
                if (dwi >= 0 && dwi < w && dhi >= 0 && dhi < h) {
                    auto qidx = bi * cwh + dwi * h + dhi;
                    scalar_t sumv = 0;
                    for (auto i = 0; i < c; ++ i) {
                        auto iwh = i * wh;
                        sumv += (query[qidx + iwh] * temp[tidx + iwh]);
                    }
                    maxv = (sumv > maxv) ? sumv : maxv;
                }
            }
        }
        match_map[idx] = maxv;
    }
}

template<typename scalar_t>
__global__ void element_wise_unique(
    const long *__restrict__ clus,
    long *__restrict__ unique_set,
    long N, long WH
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < WH) {
        auto count = 0;
        for (auto i = 0; i < N; i ++) {
            const auto cidx = i * WH + idx;
            if (clus[cidx] != -1) {
                bool unique_flag = true;
                for (auto j = 0; j < count; j ++) {
                    if (unique_set[j * WH + idx] == clus[cidx]) {
                        unique_flag = false;
                        break;
                    }
                }
                if (unique_flag) {
                    unique_set[count ++ * WH + idx] = clus[cidx];
                }
            }
        }
    }
}

template<typename scalar_t>
__global__ void easy_normal_prototype_init(
    const scalar_t *__restrict__ temp, 
    const long *__restrict__ clus, 
    bool *__restrict__ flags,
    long *__restrict__ unique_set,
    long *__restrict__ counts,
    scalar_t *__restrict__ rets,
    long tsize, long N, long C, long WH, long CWH
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < WH) {
        for (auto i = 0; i < N; i ++) {
            long uid = unique_set[i * WH + idx], choose_idx = -1;
            scalar_t max_value = -1e5;
            if (uid == -1) break;
            for (auto j = 0; j < N; j ++) {
                if (clus[j * WH + idx] == uid) {
                    auto jCWH = j * CWH + idx;
                    scalar_t sum = 0;
                    for (auto k = 0; k < N; k ++) {
                        if (clus[k * WH + idx] == uid) {
                            auto kCWH = k * CWH + idx;
                            for (auto l = 0; l < C; l ++) {
                                sum += temp[jCWH + l * WH] * temp[kCWH + l * WH];
                            }
                        }
                    }
                    if (sum > max_value) {
                        max_value = sum;
                        choose_idx = j;
                    }
                    // flags[j * WH + idx] = false;
                }
            }
            if (counts[idx] >= tsize || choose_idx == -1) break;
            auto countIdx = counts[idx] * CWH + idx, chooseIdx = choose_idx * CWH + idx;
            for (auto l = 0; l < C; l ++) {
                rets[countIdx + l * WH] = temp[chooseIdx + l * WH];
            }
            flags[choose_idx * WH + idx] = false;
            counts[idx] ++;
        }
    }
}

template<typename scalar_t>
__global__ void global_center_init(
    const scalar_t *__restrict__ temp, 
    bool *__restrict__ flags,
    long *__restrict__ counts,
    scalar_t *__restrict__ rets,
    long N, long C, long WH, long CWH
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < WH) {
        if (counts[idx] == 0) {
            scalar_t max_value = -1e5;
            long choose_idx = -1;
            for (auto i = 0; i < N; i ++) {
                auto iCWH = i * CWH + idx;
                scalar_t sum = 0;
                for (auto j = 0; j < N; j ++) {
                    auto jCWH = j * CWH + idx;
                    for (auto k = 0; k < C; k ++) {
                        sum += temp[iCWH + k * WH] * temp[jCWH + k * WH];
                    }
                }
                if (sum > max_value) {
                    max_value = sum;
                    choose_idx = i;
                }
            }
            auto countIdx = counts[idx] * CWH + idx, chooseIdx = choose_idx * CWH + idx;
            for (auto l = 0; l < C; l ++) {
                rets[countIdx + l * WH] = temp[chooseIdx + l * WH];
            }
            counts[idx] ++;
            flags[choose_idx * WH + idx] = false;
        }
    }
}

template<typename scalar_t>
__global__ void hard_normal_prototype_selection(
    const scalar_t *__restrict__ temp,
    bool *__restrict__ flags,
    long *__restrict__ counts,
    scalar_t *__restrict__ dp,
    scalar_t *__restrict__ rets,
    long tsize, long N, long C, long WH, long CWH
) {
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < WH) {
        for (auto count = counts[idx]; count < tsize; count ++) {
            scalar_t min_value = 1e5;
            long choose_idx = -1;
            for (auto i = 0; i < N; i ++) {
                auto iWH = i * WH + idx;
                if (flags[iWH]) {
                    auto iCWH = i * CWH + idx;
                    scalar_t sum = 0;
                    if (count == counts[idx]) {
                        for (auto j = 0; j < count; j ++) {
                            auto jCWH = j * CWH + idx;
                            for (auto k = 0; k < C; k ++) {
                                sum += temp[iCWH + k * WH] * rets[jCWH + k * WH];
                            }
                        }
                    } else {
                        auto jCWH = (count - 1) * CWH + idx;
                        for (auto k = 0; k < C; k ++) {
                            sum += temp[iCWH + k * WH] * rets[jCWH + k * WH];
                        }
                    }
                    dp[iWH] += sum;
                    if (dp[iWH] < min_value) {
                        min_value = dp[iWH];
                        choose_idx = i;
                    }
                }
            }
            if (choose_idx == -1) break;
            auto countIdx = count * CWH + idx, chooseIdx = choose_idx * CWH + idx;
            for (auto l = 0; l < C; l ++) {
                rets[countIdx + l * WH] = temp[chooseIdx + l * WH];
            }
            flags[choose_idx * WH + idx] = false;
        }
    }
}

torch::Tensor forward_ATM_cuda(torch::Tensor query, torch::Tensor temp, int patch) {
    const auto b = query.size(0);
    const auto t = temp.size(0);
    const auto c = temp.size(1);
    const auto w = temp.size(2);
    const auto h = temp.size(3);

    const auto d = int(patch / 2);
    const auto wh = w * h;
    const auto cwh = c * wh;
    const auto blocks = (b * t * w * h + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    auto match_map = torch::full({b, t, w, h}, -1, torch::CUDA(query.scalar_type()));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "forward_ATM_cuda", ([&] {
        forward_ATM_cuda_kernel<scalar_t><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
            query.data_ptr<scalar_t>(), 
            temp.data_ptr<scalar_t>(), 
            match_map.data_ptr<scalar_t>(),
            b, t, c, w, h, d, wh, cwh
        );
    }));

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cout << "forward_ATM_cuda: "
                << cudaGetErrorString(err);
    }
    return match_map;
}

torch::Tensor backward_ATM_cuda(torch::Tensor query, torch::Tensor temp, int patch) {
    const auto b = query.size(0);
    const auto t = temp.size(0);
    const auto c = temp.size(1);
    const auto w = temp.size(2);
    const auto h = temp.size(3);

    const auto d = int(patch / 2);
    const auto wh = w * h;
    const auto cwh = c * wh;
    const auto blocks = (b * t * w * h + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    auto match_map = torch::full({b, t, w, h}, -1, torch::CUDA(query.scalar_type()));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(query.scalar_type(), "backward_ATM_cuda", ([&] {
        backward_ATM_cuda_kernel<scalar_t><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
            query.data_ptr<scalar_t>(), 
            temp.data_ptr<scalar_t>(), 
            match_map.data_ptr<scalar_t>(),
            b, t, c, w, h, d, wh, cwh
        );
    }));

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cout << "Error in backward_ATM_cuda: "
                << cudaGetErrorString(err);
    }
    return match_map;
}

torch::Tensor PTS_cuda(torch::Tensor temp, torch::Tensor clus, long tsize) {
    const auto N = temp.size(0);
    const auto C = temp.size(1);
    const auto W = temp.size(2);
    const auto H = temp.size(3);
    const auto WH = W * H;
    const auto CWH = C * WH;
    const auto blocks = (WH + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
    auto rets = torch::zeros({tsize, C, W, H}, torch::CUDA(temp.scalar_type()));
    auto flags = torch::full({N, W, H}, true, torch::CUDA(torch::kBool));
    auto unique_set = torch::full({N, W, H}, -1, torch::CUDA(clus.scalar_type()));
    auto counts = torch::zeros({W, H}, torch::CUDA(clus.scalar_type()));
    auto dp = torch::zeros({N, W, H}, torch::CUDA(temp.scalar_type()));
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(temp.scalar_type(), "element_wise_unique", ([&] {
        element_wise_unique<scalar_t><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
            clus.data_ptr<long>(), 
            unique_set.data_ptr<long>(),
            N, WH
        );
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(temp.scalar_type(), "easy_normal_prototype_init", ([&] {
        easy_normal_prototype_init<scalar_t><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
            temp.data_ptr<scalar_t>(), 
            clus.data_ptr<long>(), 
            flags.data_ptr<bool>(),
            unique_set.data_ptr<long>(),
            counts.data_ptr<long>(),
            rets.data_ptr<scalar_t>(),
            tsize, N, C, WH, CWH
        );
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(temp.scalar_type(), "global_center_init", ([&] {
        global_center_init<scalar_t><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
            temp.data_ptr<scalar_t>(), 
            flags.data_ptr<bool>(),
            counts.data_ptr<long>(),
            rets.data_ptr<scalar_t>(),
            N, C, WH, CWH
        );
    }));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(temp.scalar_type(), "hard_normal_prototype_selection", ([&] {
        hard_normal_prototype_selection<scalar_t><<<blocks, CUDA_NUM_THREADS, 0, stream>>>(
            temp.data_ptr<scalar_t>(), 
            flags.data_ptr<bool>(),
            counts.data_ptr<long>(),
            dp.data_ptr<scalar_t>(),
            rets.data_ptr<scalar_t>(),
            tsize, N, C, WH, CWH
        );
    }));

    cudaError_t err = cudaGetLastError();

    if (err != cudaSuccess) {
        std::cout << "PTS_cuda: " << cudaGetErrorString(err);
    }
    return rets;
}