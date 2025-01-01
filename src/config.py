import os
import yaml

import torch
from torch.utils.data import DataLoader
from functools import reduce
import copy

from . import dataset, metrics, models
from .utils import dfs_update_configs, merge_configs, mlt_process, CHECK_INPUT_CUDA

class Cfg:

    def __init__(self, args):

        with open(os.path.join('configs', 'base.yaml'), 'r') as bfile:
            bcfg = yaml.safe_load(bfile)

        with open(os.path.join('configs', f'{args.dataset}.yaml'), 'r') as dfile:
            dcfg = yaml.safe_load(dfile)            

        cfg = merge_configs(args.__dict__, bcfg, dcfg)
        dfs_update_configs(cfg)
        self.cfg = cfg
        self.load_cfg()

    def load_cfg(self):
        for key, value in filter(lambda x : x[0] not in ['data', 'template'], self.cfg.items()):
            if hasattr(self, f'load_{key}'):
                getattr(self, f'load_{key}')(value)
                continue

            setattr(self, key, value)
        
    def update(self, category):
        self.category = category
        for key in  filter(lambda x : x.endswith('path'), dir(self)):
            os.makedirs(os.path.join(getattr(self, key), self.dataset, self.category), exist_ok=True)

        for func in ['data', 'template']:
            try:
                getattr(self, f'load_{func}')()
            except:
                pass

        self.model.update(category)

    def load_data(self):
        # for mode in ['test', 'temp']:
        cfg = copy.deepcopy(self.cfg['data'])
        setattr(self, f'{self.mode}set', getattr(dataset, cfg.pop('name'))(mode=self.mode, category=self.category, **cfg))
        setattr(self, f'{self.mode}loader', DataLoader(getattr(self, f'{self.mode}set'), **getattr(self, f'{self.mode}loaderparams')))
        setattr(self, f'{self.mode}_nums', getattr(self, f'{self.mode}set').__len__())

    def load_model(self, cfg):
        self.model = getattr(models, cfg.pop('name'))(**cfg).cuda()

    def load_template(self):
        if hasattr(self, 'tpath'):
            tname = f'{self.model.backbone.lower()}_ALL.pkl' if self.ttype == 'ALL' else f'{self.model.backbone.lower()}_{self.ttype}x{self.tsize}.pkl'
            tpath = os.path.join(self.tpath, self.dataset, self.category)
            temp = self.model.load_template(os.path.join(tpath, tname))
            self.temp = {k : CHECK_INPUT_CUDA(v).half() if self.half else CHECK_INPUT_CUDA(v) for k, v in temp.items()}

    def load_metrics(self, cfg):
        self.metrics = metrics.Metrics(**cfg)