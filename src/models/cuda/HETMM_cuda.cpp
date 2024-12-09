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

#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")

#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor forward_ATM_cuda(torch::Tensor query, torch::Tensor temp, int patch);
torch::Tensor backward_ATM_cuda(torch::Tensor query, torch::Tensor temp, int patch);
torch::Tensor PTS_cuda(torch::Tensor temp, torch::Tensor clus, long tsize);

torch::Tensor forward_ATM(torch::Tensor query, torch::Tensor temp, int patch) {
  CHECK_INPUT(query);
  CHECK_INPUT(temp);
  return forward_ATM_cuda(query, temp, patch);
}
torch::Tensor backward_ATM(torch::Tensor query, torch::Tensor temp, int patch) {
  CHECK_INPUT(query);
  CHECK_INPUT(temp);
  return backward_ATM_cuda(query, temp, patch);
}
torch::Tensor PTS(torch::Tensor temp, torch::Tensor clus, long tsize) {
  CHECK_INPUT(temp);
  CHECK_INPUT(clus);
  return PTS_cuda(temp, clus, tsize);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward_ATM", &forward_ATM, "Forward Affine-invariant Template Matching (CUDA)");
  m.def("backward_ATM", &backward_ATM, "Backward Affine-invariant Template Matching (CUDA)");
  m.def("PTS", &PTS, "Pixel-level Template Selection (CUDA)");
}