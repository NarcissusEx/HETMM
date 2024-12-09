import os
import math
from joblib import Parallel, delayed
import threading
from functools import reduce
import queue

import numpy as np
import torch

from torch.utils.cpp_extension import load
parent_dir = os.path.dirname(os.path.abspath(__file__))
HETMM_cuda = load(
    name='HETMM_cuda',
    sources=[os.path.join(parent_dir, 'models', 'cuda', file) for file in ['HETMM_cuda.cpp', 'HETMM_cuda.cu']], 
    verbose=False
)

def CHECK_INPUT_CUDA(data): 
    CHECK_TYPE = lambda x : type(x) == torch.Tensor
    CHECK_CUDA = lambda x : x.is_cuda
    if not CHECK_TYPE(data):
        if not isinstance(data, (np.ndarray, list, tuple)):
            print (f'The type is wrong! Expect torch.Tensor / np.ndarray / list / tuple, but got {type(data)}')
            raise TypeError
        
        else:
            data = torch.tensor(data)

    if not CHECK_CUDA(data):
        data = data.cuda()

    return data

def CHECK_INPUT_NUMPY(data): 
    CHECK_TYPE = lambda x : type(x) == np.ndarray
    CHECK_CUDA = lambda x : x.is_cuda

    if CHECK_TYPE(data):
        return data
    
    elif isinstance(data, (list, tuple)):
        return np.array(data)

    elif isinstance(data, (torch.Tensor)):
        return data.cpu().numpy() if CHECK_CUDA(data) else data.numpy()
    
    else:
        print (f'The type is wrong! Expect torch.Tensor / np.ndarray / list / tuple, but got {type(data)}')
        raise TypeError

def dfs_update_configs(cfg):
    q = queue.Queue()
    q.put(cfg)
    while not q.empty():
        tmp = q.get()
        if isinstance(tmp, dict):
            for key in tmp.keys():
                if isinstance(tmp[key], str):
                    if '$' in tmp[key]:
                        klist = [k if i % 2 == 0 else cfg[k] for i, k in enumerate(tmp[key].split('$'))]
                        try:
                            tmp[key] = cfg[reduce(lambda x1, x2 : x1 + x2, filter(lambda x : x != '', klist))]
                        except:
                            tmp[key] = reduce(lambda x1, x2 : x1 + x2, filter(lambda x : x != '', klist))
                q.put(tmp[key])

        elif isinstance(tmp, list):
            for idx, item in enumerate(tmp):
                if isinstance(item, str):
                    if '$' in item:
                        klist = [k if i % 2 == 0 else cfg[k] for i, k in enumerate(item.split('$'))]
                        try:
                            tmp[idx] = cfg[reduce(lambda x1, x2 : x1 + x2, filter(lambda x : x != '', klist))]
                        except:
                            tmp[idx] = reduce(lambda x1, x2 : x1 + x2, filter(lambda x : x != '', klist))
                q.put(tmp[idx])

def merge_configs(args, *cfgs):
    cfgs = reduce(lambda x1, x2 : {**x1, **x2}, cfgs)
    nargs = {k : v for k, v in filter(lambda x : x[1] is not None or x[0] not in cfgs.keys(), args.items())}
    return {**cfgs, **nargs}

def mlt_process(func, params, const_params, num_workers, isTorch=False):
    with Parallel(n_jobs=num_workers) as parallel:
        return parallel(delayed(func)(*param, *const_params) for param in params)

# def mlt_process(func, params, num_workers=8):
#     with multiprocessing.Pool(num_workers) as pool:
#         rets = pool.starmap(func, params)
#     return rets

# def mlt_process(rets, const_params, params, function, num_workers=8, is_tqdm=False):
#     q = queue.Queue(num_workers)
#     for idx, param in enumerate(tqdm(params) if is_tqdm else params):
#         if num_workers > 0:
#             t = Process(target=function, args=(rets, idx, *const_params, *param))
#             t.start()
#             q.put(t)
#             if q.full():
#                 while not q.empty():
#                     t = q.get()
#                     t.join()
#         else:
#             function(rets, idx, *const_params, *param)

#     while not q.empty():
#         t = q.get()
#         t.join()
    
#     return rets

# def mlt_process(rets, const_params, params, function, num_workers=8, is_tqdm=False):
#     q = queue.Queue(num_workers)
#     for idx, param in enumerate(tqdm(params) if is_tqdm else params):
#         t = threading.Thread(target=function, args=(rets, idx, *const_params, *param))
#         t.start()
#         q.put(t)
#         if q.full():
#             while not q.empty():
#                 t = q.get()
#                 t.join()
#     while not q.empty():
#         t = q.get()
#         t.join()

#     return rets