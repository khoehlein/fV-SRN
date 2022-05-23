import os
import socket

import numpy as np
import torch
import pyrenderer
from torch import Tensor

DATA_BASE_PATH = {
    'tuini15-cg05-cg-in-tum-de': '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/cvol/single_variable',
    'gpusrv01': '/home/hoehlein/data/1000_member_ensemble/cvol/single_variable'
}


class SingleVariableData(object):

    def __init__(self, variable_name: str, normalization: str, device=None, dtype=None):
        self.variable_name = variable_name
        self.normalization = normalization
        self.device = torch.device('cpu') if device is not None else device
        self.dtype = dtype

    @staticmethod
    def get_data_base_path():
        host_name = socket.gethostname()
        return DATA_BASE_PATH[host_name]

    def get_variable_base_path(self):
        return os.path.join(
            SingleVariableData.get_data_base_path(),
            f'{self.normalization}_scaling',
            self.variable_name
        )

    def load_volume(self, member: int, timestep: int):
        path = os.path.join(
            self.get_variable_base_path(),
            'member{:04d}'.format(member),
            't{:02d}.cvol'.format(timestep),
        )
        return pyrenderer.Volume(path)

    def _postprocess(self, data: Tensor):
        if data.device != self.device:
            data = data.to(self.device)
        if self.dtype is not None and data.dtype != self.dtype:
            data = data.to(self.dtype)
        return data

    def load_tensor(self, member: int, timestep: int):
        volume = self.load_volume(member, timestep)
        data = volume.get_feature(0).get_level(0).to_tensor()
        return self._postprocess(data)

    def load_scales(self, return_volume=False):
        scales_file = os.path.join(self.get_variable_base_path(), 'scales.npz')
        scales_data = np.load(scales_file)
        offset = torch.from_numpy(scales_data['offset'][None, ...])
        scale = torch.from_numpy(scales_data['scale'][None, ...])
        offset = self._postprocess(offset)
        scale = self._postprocess(scale)
        if return_volume:
            vol = pyrenderer.Volume()
            vol.worldX = 10.
            vol.worldY = 10.
            vol.worldZ = 1.
            vol.add_feature_from_tensor('offset', offset.to(torch.float32).cpu())
            vol.add_feature_from_tensor('scale', scale.to(torch.float32).cpu())
            return vol
        return offset, scale


def _test():
    data = SingleVariableData('tk', 'level-min-max')
    volume = data.load_tensor(1, 4)
    scales = data.load_scales(return_volume=True)
    print('Finished')


if __name__ == '__main__':
    _test()
