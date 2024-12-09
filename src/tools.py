import os
from functools import reduce, partial
import numbers
import math

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F

from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from .utils import CHECK_INPUT_NUMPY

cvt2heatmap = lambda x : cv2.applyColorMap(np.uint8(x), cv2.COLORMAP_JET)
min_max_norm = lambda x : (x - x.min()) / (x.max() - x.min())

def make_mask(img, mask, th=0.5):
    
    almask = np.zeros((*img.shape[:2], 4), dtype=np.uint8)
    colors = np.unique(mask)
    almask[mask >= th] = np.array([0, 215, 255, 60])
    almask = Image.fromarray(almask).convert('RGBA')
    _, _, _, a = almask.split()
    Kimg = Image.fromarray(img)
    Kimg.paste(almask, (0, 0), mask=a)
    
    return Kimg

def show_cam_on_image(img, anomaly_map):
    heatmap = cv2.applyColorMap(np.uint8(anomaly_map), cv2.COLORMAP_JET)
    cam = np.float32(heatmap) + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def heatmap_on_image(heatmap, image):
    out = np.float32(heatmap) + np.float32(image)
    return np.uint8(out / np.max(out) * 255)

def save_anomaly_map(ams, imgs, gts, result_path, filenames, method, types):
    ams, imgs, gts = [CHECK_INPUT_NUMPY(x) for x in [ams, imgs, gts]]
    ams = min_max_norm(ams)
    for am, img, gt, filename, x_type in zip(ams, imgs, gts, filenames, types):
        input_img = np.flip(img.transpose((1, 2, 0)) * 255, 2)
        anomaly_map_norm = min_max_norm(am)
        hm = cvt2heatmap(anomaly_map_norm * 255)
        hmimg = heatmap_on_image(hm, input_img)
        if not os.path.exists(os.path.join(result_path, f'{x_type}_{filename}.png')):
            cv2.imwrite(os.path.join(result_path, f'{x_type}_{filename}.png'), input_img)
            cv2.imwrite(os.path.join(result_path, f'{x_type}_{filename}_gt.png'), gt * 255)

        cv2.imwrite(os.path.join(result_path, f'{x_type}_{filename}_{method}.png'), hm)
        cv2.imwrite(os.path.join(result_path, f'{x_type}_{filename}_{method}_cam.png'), hmimg)

def save_results(ams, als, save_path, method, cat, defects, names):
    s_path = os.path.join(save_path, method, cat)
    os.makedirs(s_path, exist_ok=True)

    _als = {}
    np.save(f'{s_path}.npy', als)
    for am, al, defect, name in zip(ams, als, defects, names):
        np.save(os.path.join(s_path, f'{defect}_{name}.npy'), am)
        _als[f'{defect}_{name}'] = al
    np.save(f'{s_path}.npy', _als)

class GaussianFilter(torch.nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        k_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, sigma, k_size, dim=2):
        self._k_size = k_size
        super(GaussianFilter, self).__init__()
        if isinstance(k_size, numbers.Number):
            k_size = [k_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in k_size
            ]
        )
        for size, std, mgrid in zip(k_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self._k_size//2)

def func_iters(x, funcs):
    c = torch.clone(x)
    for func in funcs:
        c = func(c)
    return c

def blur(s, k, half=False):
    bfunc = GaussianFilter(1, s, int(k)).cuda()
    return bfunc.half() if half else bfunc

def amax(half=False):
    return lambda x : torch.max(torch.reshape(x, (x.shape[0], -1)), 1)[0].half() if half else torch.max(torch.reshape(x, (x.shape[0], -1)), 1)[0]

def mp(k, half=False):
    return lambda x : torch.mean(F.max_pool2d(x, int(k), int(k), padding=0).reshape(x.shape[0], -1), -1).half() if half else torch.mean(F.max_pool2d(x, int(k), int(k), padding=0).reshape(x.shape[0], -1), -1)

def decimal_round(n, d):
    return float(Decimal(str(n)).quantize(Decimal('0.' + '0' * d), ROUND_HALF_UP))

def binarize(data):
    data[data >= 0.5] = 1
    data[data < 0.5] = 0
    return data.bool()