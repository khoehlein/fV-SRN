from numbers import Number
from typing import Union, Tuple

import torch
import torch.nn.functional as F
from torch import nn, Tensor


class _GaussianFilterNd(nn.Module):

    def __init__(self, dim: int, k: Union[Tuple[int, ...], int], sigma: Union[float, Tuple[float, ...]]):
        super().__init__()
        self._parse_inputs(dim, k, sigma)
        self._build_kernel()

    def _parse_inputs(self, dim, k, sigma):
        self.dim = dim
        if isinstance(k, Number):
            k = [k, ] * dim
        k = tuple(k)
        assert len(k) == dim
        self.k = k
        if isinstance(sigma, Number):
            sigma = [sigma, ] * dim
        sigma = tuple(sigma)
        assert len(sigma) == dim
        self.sigma = sigma

    def _build_kernel(self):

        def gaussian_kernel_1d(k: int, sigma: float):
            x = torch.arange(k) - (k - 1) / 2.
            weight = torch.exp(- x ** 2 / (2 * sigma ** 2))
            return weight

        dim_kernels = torch.meshgrid(*[gaussian_kernel_1d(k, sigma) for k, sigma in zip(self.k, self.sigma)])
        kernel = torch.prod(torch.stack(dim_kernels, dim=0), dim=0)
        self.register_buffer('kernel', kernel)

    def forward(self, x: Tensor):
        self._verify_input_shape(x)  # expected shape: (batch_size, num_channels, *(spatial size))
        num_channels = x.shape[1]
        kernel = self._prepare_kernel(num_channels)
        return self.convolve(x, kernel, num_channels)

    def _verify_input_shape(self, x: Tensor):
        assert len(x.shape) == (2 + self.dim), \
            f'[ERROR] Gausian kernel module expects {self.dim + 2}d input, but got {len(x.shape)}d instead.'

    def _prepare_kernel(self, num_channels: int):
        kernel = self.kernel[None, None, ...].tile(num_channels, 1, 1, 1)
        return kernel

    def convolve(self, x: Tensor, kernel: Tensor, num_channels):
        raise NotImplementedError()

    def kernel_size(self):
        return self.k

    def pad(self, x: Tensor, method='constant', value=0.):
        padding = (self.k[1] // 2, self.k[1] // 2, self.k[0] // 2, self.k[0] // 2)
        return F.pad(x, padding, mode=method, value=value)

    def crop(self, x: Tensor):
        l0 = self.k[0] // 2
        l1 = self.k[1] // 2
        out = x[..., l0:-l0, l1:-l1]
        return out


class GaussianFilter2d(_GaussianFilterNd):

    def __init__(self, k, sigma):
        super(GaussianFilter2d, self).__init__(2, k, sigma)

    def convolve(self, x: Tensor, kernel: Tensor, num_channels: int):
        return F.conv2d(x, kernel, bias=None, groups=num_channels)


class GaussianFilter3d(_GaussianFilterNd):

    def __init__(self, k, sigma):
        super(GaussianFilter3d, self).__init__(3, k, sigma)

    def convolve(self, x: Tensor, kernel: Tensor, num_channels: int):
        return F.conv3d(x, kernel, bias=None, groups=num_channels)