import os

import numpy as np
import torch
import tqdm

from volnet.experiments.ensemble_training.directories import get_data_base_path
from volnet.modules.datasets.cvol_reader import VolumetricScene
from volnet.modules.metrics.dssim import DataSSIM2d

data_base_path = get_data_base_path()
file_pattern = 'tk/member{member:04d}/t04.cvol'

file_path_pattern = os.path.join(data_base_path, file_pattern)

MIN_MEMBER = 1
MAX_MEMBER = 128


def load_data(file_name):
    vol = VolumetricScene.from_cvol(file_name)
    feature = vol.get_active_feature()
    data = feature.data[0].T
    return data[None, ...]


members = list(range(MIN_MEMBER, MAX_MEMBER + 1))
num_members = len(members)
num_levels = 12


values = np.zeros((num_members, num_members, num_levels))

dssim = DataSSIM2d()

with tqdm.tqdm(total=num_members * (num_members - 1) // 2) as pbar:
    for i, member_i in enumerate(range(MIN_MEMBER, MAX_MEMBER)):
        data_i = load_data(file_path_pattern.format(member=member_i))
        for j, member_j in enumerate(range(member_i + 1, MAX_MEMBER + 1)):
            data_j = load_data(file_path_pattern.format(member=member_j))
            dssim_ij = dssim(data_i, data_j)[0].data.cpu().numpy()
            # dssim_ij = torch.sqrt(torch.sum((data_i - data_j) ** 2, dim=(0, 2, 3)))
            values[i, i + j + 1] = dssim_ij
            values[i + j + 1, i] = dssim_ij
            pbar.update(1)

np.savez('similarities.npz', similarities=values, members=members)
print('Finished')
