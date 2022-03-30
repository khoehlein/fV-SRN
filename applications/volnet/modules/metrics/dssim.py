from numbers import Number
from typing import Union, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn, Tensor

from volnet.modules.metrics.gaussian_filter import GaussianFilter2d


class DataSSIM2d(nn.Module):

    def __init__(self, filter: Optional[GaussianFilter2d] = None, c1=1.e-8, c2=1.e-8, crop=True):
        """
        Pytorch implementation of the Data SSIM, as shown in http://arxiv.org/abs/2202.02616

        Note: The handling of NaNs is different! NaNs are neglected in the computation, instead of being interpolated.
        """
        super(DataSSIM2d, self).__init__()
        if filter is None:
            filter = GaussianFilter2d(11, 1.5)
        self.filter = filter
        self.c1 = c1
        self.c2 = c2
        self.crop = crop

    def forward(self, x: Tensor, y: Tensor):
        x_shape = x.shape
        y_shape = y.shape
        assert x_shape == y_shape
        in_shape = x_shape
        if len(in_shape) > 4:
            # case of vertical levels
            x = x.view(in_shape[0], -1, *in_shape[-2:])
            y = y.view(in_shape[0], -1, *in_shape[-2:])
        x, y = self._apply_normalization(x, y)
        mu_x, var_x = self._apply_filter(x)
        mu_y, var_y = self._apply_filter(y)
        xy = x * y
        c_xy, nans_xy = self._compute_correlation(xy, mu_x * mu_y)
        ssim_t1 = 2. * mu_x * mu_y + self.c1
        ssim_t2 = 2.* c_xy + self.c2
        ssim_b1 = mu_x ** 2 + mu_y ** 2 + self.c1
        ssim_b2 = var_x + var_y + self.c2
        ssim_1 = ssim_t1 / ssim_b1
        ssim_2 = ssim_t2 / ssim_b2
        ssim_mat = ssim_1 * ssim_2
        if self.crop:
            ssim_mat = self.filter.crop(ssim_mat)
            nans_xy = self.filter.crop(nans_xy)
        ssim_mat[nans_xy] = float('nan')
        new_shape = ssim_mat.shape
        mean_ssim = torch.nansum(ssim_mat, dim=(-1, -2)) / torch.sum(~nans_xy, dim=(-1, -2))
        if len(in_shape) > 4:
            mean_ssim = mean_ssim.view(in_shape[:-2])
            mean_ssim = torch.amin(mean_ssim, dim=-1)
        return mean_ssim

    @staticmethod
    def _apply_normalization(x: Tensor, y: Tensor):
        min_x = torch.nanquantile(x.view(*x.shape[:-2], -1), 0., dim=-1, keepdim=True)
        max_x = torch.nanquantile(x.view(*x.shape[:-2], -1), 1., dim=-1, keepdim=True)
        min_y = torch.nanquantile(y.view(*y.shape[:-2], -1), 0., dim=-1, keepdim=True)
        max_y = torch.nanquantile(y.view(*y.shape[:-2], -1), 1., dim=-1, keepdim=True)
        min_total = torch.minimum(min_x, min_y)[..., None]
        max_total = torch.maximum(max_x, max_y)[..., None]
        span = torch.abs(max_total - min_total)
        x, y = (x - min_total) / span, (y - min_total) / span
        return torch.round(x * 255) / 255, torch.round(y * 255) / 255

    def _apply_filter(self, x: Tensor):
        nans = torch.isnan(x)
        if torch.any(nans):
            x = torch.nan_to_num(x, nan=0.)
        valid_mask = torch.where(nans, 0., 1.)
        x = self.filter.pad(x)
        valid_mask = self.filter.pad(valid_mask)
        weights = self.filter(valid_mask)
        mu = self.filter(x) / weights
        var = self.filter(x ** 2) / weights - mu ** 2
        return mu, var

    def _compute_correlation(self, xy, mu_xy):
        nans = torch.isnan(xy)
        valid_mask = torch.where(nans, 0., 1.)
        xy = self.filter.pad(xy)
        valid_mask = self.filter.pad(valid_mask)
        weights = self.filter(valid_mask)
        c = self.filter(xy) / weights - mu_xy
        return c, nans


class DataDSSIM2d(DataSSIM2d):

    def forward(self, x: Tensor, y: Tensor):
        return (1. - super(DataDSSIM2d, self).forward(x, y)) / 2.


def test():
    ssim = DataSSIM2d()

    a = torch.randn(10, 4, 32, 64)
    b = a #torch.randn(10, 4, 32, 64)

    out = ssim(a, b)

    print(out)

    print('Finished')


if __name__ == '__main__':
    test()
