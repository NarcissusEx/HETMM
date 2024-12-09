import numpy as np
from skimage import measure
import torch
from sklearn.metrics import roc_auc_score, auc
from torchmetrics import AUROC

from functools import reduce

from src import *
from src.tools import decimal_round
from src.utils import mlt_process, CHECK_INPUT_CUDA, CHECK_INPUT_NUMPY
import os
import json

roc_auc_score = AUROC(task='binary')

class Metrics:

    def __init__(self, num_workers, functions):
        self.num_workers = num_workers
        self.metrics = {func : getattr(self, func) for func in functions}
        self.results = {'mean' : {func : [] for func in functions}}
        self.cats = []

    def evaluate(self, cat, *res): 
        self.record(cat, {name : func(res) for name, func in self.metrics.items()})

    def PRO(self, res, steps=300, beta=0.3, eps=1e-5):
        rescale = lambda x : (x - x.min()) / (x.max() - x.min() + eps)
        def PRO_unit(thred, ams, gts, neg_gts, neg_gts_sum):
            maps = ams > thred
            fp = np.logical_and(neg_gts, maps)
            pro = []
            for i in range(len(maps)):
                label_map = measure.label(gts[i], connectivity=2)
                props = measure.regionprops(label_map, maps[i])
                for prop in props:
                    pro.append(prop.intensity_image.sum() / (prop.area + eps))
            return [fp.sum() / neg_gts_sum, np.array(pro).mean()]

        ams, gts = [CHECK_INPUT_NUMPY(x) for x in [res[2]['PRO'], res[1]]]

        const_params = [ams, gts, ~gts, (~gts).sum()]
        params = [[thred] for thred in np.linspace(ams.min(), ams.max(), steps)]
        rets = mlt_process(PRO_unit, params, const_params, self.num_workers)
        fprs = np.array([ret[0] for ret in rets])
        pros = np.array([ret[1] for ret in rets])

        idx = fprs <= beta    
        fprs_selected = fprs[idx]
        fprs_selected = rescale(fprs_selected)
        pros_selected = rescale(pros[idx])
        pro = auc(fprs_selected, pros_selected) * 100.
        return pro

    def img_AUC(self, res):
        ams, gts = [CHECK_INPUT_CUDA(x) for x in [res[2]['img_AUC'], res[0]]]
        with torch.no_grad():
            img_auc = roc_auc_score(ams, gts.cuda()).item() * 100.
            roc_auc_score.reset()
            return img_auc

    def pix_AUC(self, res):
        ams, gts = [CHECK_INPUT_CUDA(x) for x in [res[2]['pix_AUC'], res[1]]]
        with torch.no_grad():
            pix_auc = roc_auc_score(ams.ravel(), gts.ravel().cuda()).item() * 100.
            roc_auc_score.reset()
            return pix_auc
        
    def record(self, cat, results):
        self.results[cat] = results
        self.cats.append(cat)

        for name, result in results.items():
            self.results['mean'][name].append(result)

    def show(self, silence=False):
        mean = lambda xs : sum(xs) / len(xs)

        for cat in self.cats:
            rlst = [f'{name} : {result:<4.1f}' for name, result in self.results[cat].items()]
            ptxt = reduce(lambda x1, x2 : f'{x1} | {x2}', rlst)
            print (f'{cat:<15}{ptxt}')

        tdict, odict, adict = {}, {}, {}
        for name, plist in self.results['mean'].items():
            adict[name] = mean(plist)
            tdict[name] = mean(plist[:5])
            odict[name] = mean(plist[5:])
        print ('----------------------------------------------------')
        for abbr, pdict in zip(['Texture', 'Object', 'Average'], [tdict, odict, adict]):
            plist = [f'{name} : {result:<4.1f}' for name, result in pdict.items()]
            ptext = reduce(lambda x1, x2 : f'{x1} | {x2}', plist)
            print (f'{abbr:<15}{ptext}')