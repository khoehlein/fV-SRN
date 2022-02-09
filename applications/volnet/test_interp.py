import argparse
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import utils
import pyrenderer


def pyrenderer_interp1D(fp, x):
    # fp: Tensor of shape (B, C, N)
    # x: Tensor of shape (B, M)
    x = x[..., None] * 2. / (fp.shape[-1] - 1.) - 1.
    grid = torch.stack([torch.zeros_like(x), x], dim=-1)
    out = F.grid_sample(fp[..., None], grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out[..., 0]


def main():
    device = torch.device('cuda:0')
    fp = torch.randn(1, 4, 16)
    fp = fp.to(device)
    x = torch.linspace(0, 15, 7)[None, ...]
    x = x.to(device)
    i1 = pyrenderer_interp1D(fp, x)
    i2 = pyrenderer.interp1D(fp, x)
    print('Finished')


if __name__ == '__main__':
    main()
