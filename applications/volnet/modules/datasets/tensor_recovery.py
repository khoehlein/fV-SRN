import numpy as np
import pyrenderer
import torch
from matplotlib import pyplot as plt

from volnet.modules.datasets.sampling.original_grid import get_normalized_positions


def _test_weird_offsetting():
    device = torch.device('cuda:0')

    # load scene from tensor
    s = 6
    data1 = torch.arange(s ** 3, dtype=torch.float32).reshape(1, s, s, s)
    data2 = torch.randn(1, s, s, s)
    for i, data in enumerate([data1, data2]):
        print(f'Data {i + 1}:')

        volume = pyrenderer.Volume()
        volume.add_feature_from_tensor('feature:0', data)
        feature = volume.get_feature(0)
        values_original = data.view(-1)
        values_recovered = feature.get_level(0).to_tensor().view(-1)

        # get positions of tensor values
        positions = torch.meshgrid(torch.linspace(0., 1., 4097), torch.arange(1.), torch.arange(1.))
        positions = torch.stack(positions, dim=-1).to(device).view(-1, 3)

        # interpolate
        interpolator = pyrenderer.VolumeInterpolationGrid()
        interpolator.setInterpolation(interpolator.Trilinear)
        interpolator.grid_resolution_new_behavior = True
        interpolator.setSource(volume, 0)
        values_pyrenderer = interpolator.evaluate(positions).view(-1)

        plt.figure(figsize=(10, 5), dpi=600)
        plt.plot(
            positions[:, 0].data.cpu().numpy(),
            values_pyrenderer.view(-1).data.cpu().numpy())
        plt.show()
        plt.close()

    print('Finished')

def _test_normalized_positions():
    device = torch.device('cuda:0')
    s = 4
    data1 = torch.arange(s ** 3, dtype=torch.float32).reshape(1, s, s, s)
    data2 = torch.randn(1, s, s, s)
    for i, data in enumerate([data1, data2]):
        print(f'Data {i + 1}:')
        volume = pyrenderer.Volume()
        volume.add_feature_from_tensor('feature:0', data)
        feature = volume.get_feature(0)
        values_original = data.view(-1)
        values_recovered = feature.get_level(0).to_tensor().view(-1)

        # get positions of tensor values
        positions = torch.meshgrid(*[get_normalized_positions(r) for r in (s, s, s)])
        positions = torch.stack(positions, dim=-1).to(device).view(-1, 3)

        # interpolate
        interpolator = pyrenderer.VolumeInterpolationGrid()
        interpolator.setInterpolation(interpolator.Trilinear)
        interpolator.setSource(volume, 0)
        values_interpolated = interpolator.evaluate(positions).view(-1)

        print('Original:', values_original)
        print('Recovered:', values_recovered)
        print('Interpolated:', values_interpolated)

    print('Finished')


if __name__ == '__main__':
    _test_weird_offsetting()