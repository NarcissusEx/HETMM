import argparse
from tqdm import tqdm
import os
import copy

import torch
import torch.nn.functional as F
import numpy as np

from src import tools, Cfg, template as tl

def argParse():

    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--mode', choices=['test', 'temp'], default='test')
    parser.add_argument('--method', default='ATMM')
    parser.add_argument('--ttype', choices=['ALL', 'PTS'], default='ALL')
    parser.add_argument('--tsize', type=int, default=0)
    parser.add_argument('--datapath', help='your own data path')
    parser.add_argument('--dataset', type=str, default='MVTec_AD')
    parser.add_argument('--categories', type=str, nargs='+', default=None)
    parser.add_argument('--half', action='store_true')
    parser.add_argument('--save_map', action='store_true')
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--silence', action='store_true')

    args = parser.parse_args()
    return args

def test(cfg):
    rpath = os.path.join(cfg.rpath, cfg.dataset, cfg.category)
    with torch.no_grad():
        scores, gts, imgs = [], [], []
        for batch in cfg.testloader:
            img, gt = batch
            query = cfg.model(img.cuda())
            score = cfg.model.impl(query, cfg.temp)
            scores.append(score), gts.append(gt), imgs.append(cfg.testset.inv_trans(img))

        scores = torch.cat(scores, 0)
        ans = cfg.model.post_process(scores)
        ans = {k : v if 'img_AUC' in k else torch.squeeze(v, 1) for k, v in ans.items()}               
        gts = tools.binarize(torch.squeeze(torch.cat(gts, 0), 1))
        gls = torch.tensor(cfg.testset.labels)
        imgs = torch.cat(imgs, 0)

        cfg.metrics.evaluate(cfg.category, gls, gts, ans)
        if cfg.save_map:
            tools.save_anomaly_map(ans['pix_AUC'], imgs, gts, rpath, cfg.testset.filenames, 'HETMM', cfg.testset.types)

def temp(cfg):
    tpath = os.path.join(cfg.tpath, cfg.dataset, cfg.category)
    tname = f'{cfg.model.backbone.lower()}_ALL.pkl' if cfg.ttype == 'ALL' else f'{cfg.model.backbone.lower()}_{cfg.ttype}x{cfg.tsize}.pkl'
    os.makedirs(tpath, exist_ok=True)

    def get_ALL(cfg, tpath):
        try:
            tdict = cfg.model.load_template(os.path.join(tpath, f'{cfg.model.backbone.lower()}_ALL.pkl'))

        except:
            tdict = tl.gen_by_ALL(cfg.model, cfg.temploader, tpath, cfg.model.backbone.lower(), cfg.half, save=True)

        return tdict

    if cfg.ttype == 'ALL':
        return get_ALL(cfg, tpath)

    else:
        try:
            tdict = cfg.model.load_template(os.path.join(tpath, tname))

        except:
            tdict = getattr(tl, f'gen_by_{cfg.ttype}')(get_ALL(cfg, tpath), cfg.tsize, tpath, cfg.model.backbone.lower(), num_workers=cfg.num_workers, save=True)

    return tdict

if __name__ == '__main__':
    args = argParse()
    cfg = Cfg(args)
    categories = tqdm(cfg.categories)
    for category in categories:
        categories.set_description(category)
        cfg.update(category)
        globals()[args.mode](cfg)

    if args.mode == 'test':
        cfg.metrics.show()
