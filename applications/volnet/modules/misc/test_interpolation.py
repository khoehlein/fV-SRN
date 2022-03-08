import pyrenderer
import torch
from torch import Tensor
from torch.nn import functional as F

def my_interpolate(fp_: Tensor, x_: Tensor):
    key_times = torch.linspace(0, 1, fp_.shape[-1],device=x.device)
    indices = torch.searchsorted(key_times, x_)
    upper = torch.clamp(indices, 0, len(key_times) - 1)
    lower = torch.where(
        torch.lt(x_, key_times[-1]),
        torch.clamp(indices - 1, 0, len(key_times) - 1),
        torch.full_like(indices, len(key_times) - 1)
    )
    fraction = torch.zeros_like(x_)
    bounds_differ = torch.not_equal(lower, upper)
    time_lower = key_times[lower[bounds_differ]]
    time_upper = key_times[upper[bounds_differ]]
    fraction[bounds_differ] = (x_[bounds_differ] - time_lower) / (time_upper - time_lower)
    features_lower = fp_[..., lower]
    features_upper = fp_[..., upper]
    features = (1. - fraction[None, :]) * features_lower + fraction[None, :] * features_upper
    return features


device = torch.device('cuda:0')
fp = torch.randn(1, 1, 3).to(device)
x = torch.sort(torch.rand(1, 10))[0].to(device)
y = torch.zeros_like(x)
yp = pyrenderer.interp1D(fp, 2 * x)
yt = F.grid_sample(fp[:, :, None, :], torch.stack([x[None, ...] * 2 - 1, y[None, ...]], dim=-1), align_corners=True)
yi = my_interpolate(fp[0], x.view(-1))
print('Finished')