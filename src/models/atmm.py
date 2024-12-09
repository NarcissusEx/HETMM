import os

import torch
from torch import nn
import torch.nn.functional as F
from functools import reduce, partial

from .backbone import *
from src.tools import blur, amax, mp, func_iters

from torch.utils.cpp_extension import load
from HETMM_cuda import forward_ATM, backward_ATM

class BASE(nn.Module):

    def __init__(self, **params):
        super().__init__()
        for key, value in params.items():
            setattr(self, key, value)

        encoder = globals()[self.backbone.lower()](pretrained=True).eval()
        self.encoder = encoder.half() if self.ishalf else encoder
        self.upsample = lambda x : F.interpolate(x, size=self.out_size, mode='bilinear', align_corners=True)
        self.add = lambda x1, x2 : x1 + x2

    def forward(self, x):
        return {k : F.normalize(v, p=2) for k, v in self.encoder(x.half() if self.ishalf else x).items()}

    def impl(self, querys, temps, patches):
        querys, temps = [{k : v.contiguous() for k, v in x.items()} for x in [querys, temps]] 
        return [[querys[f'layer{idx + 1}'], temps[f'layer{idx + 1}']] for idx, patch in enumerate(patches) if patch != 0]

    def make_template(self, x):
        return self(x)

    def load_template(self, tpath, is_cuda=True):
        return torch.load(tpath, weights_only=True, map_location='cuda' if is_cuda else 'cpu')
    
    def update(self, cat=None):
        try:
            for key, value in self.decode_str(self.mparams[cat]).items():
                self.__setattr__(key, value)
        except:
            print (f'{cat} is not in the categories!')

        for key, value in self.decode_str(self.mparams[cat]).items():
                self.__setattr__(key, value)

    def decode_mstr(self, mstr):
        mfuncs = {}
        for sm_str in mstr.split('|'):
            met, mparams = sm_str.split(':')
            funcs = []
            if mparams == 'clean':
                funcs.append(nn.Identity())

            else:
                func_items  = mparams.split('_')
                funcs.append(globals()[func_items[0]](half=self.ishalf, **{item.split('=')[0] : float(item.split('=')[1]) for item in func_items[1:]}))

            if met == 'img_AUC':
                funcs.append(amax())

            mfuncs[met] = partial(func_iters, funcs=funcs)

        return {'mfuncs' : mfuncs}
    
    def decode_str(self, cstr):
        pstr, mstr = cstr.split(';')
        return {
            **self.decode_mstr(mstr),
            **self.decode_pstr(pstr)
        }
    
    def post_process(self, x):
        return {m : func(x) for m, func in self.mfuncs.items()} 
    
    def match(self, feas, patches, func):
        return [self.upsample(1 - func(q, t, r).amax(dim=1).reshape(q.shape[0], 1, *q.shape[-2:])) for (q, t), r in zip(feas, patches)]

class ATMM(BASE):

    def __init__(self, **params):
        super().__init__(**params)

    def decode_pstr(self, pstr):
        fore_patches, back_patches, alphas, ldas = [], [], [], []
        for ls_str in pstr.split('|'):
            ls_str = ls_str.split(':')[-1]
            fbp, alpha, lda = ls_str.split('_')
            fore_patch, back_patch = [int(x) for x in fbp.split('x')]
            fore_patches.append(fore_patch)
            back_patches.append(back_patch)
            alphas.append(float(alpha.split('=')[-1]))
            ldas.append(float(lda.split('=')[-1]))
        
        return {
            'fore_patches' : fore_patches,
            'back_patches' : back_patches,
            'alphas'       : alphas,
            'ldas'         : ldas
        }

    def impl(self, querys, temps):
        fores = self.match(super().impl(querys, temps, self.fore_patches), self.fore_patches, forward_ATM)
        backs = self.match(super().impl(querys, temps, self.back_patches), self.back_patches, backward_ATM)
        return reduce(self.add, [lda * (fp * alpha + (1. - alpha) * bp) for fp, bp, alpha, lda in zip(fores, backs, self.alphas, self.ldas)])