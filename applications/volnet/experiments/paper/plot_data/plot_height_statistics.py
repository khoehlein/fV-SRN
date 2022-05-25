import os

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

path = '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/raw/member0001.nc'

data = xr.open_dataset(path).isel(time=-2)

variable_names = ['tk', 'rh', 'qv']

fig, ax = plt.subplots(12, len(variable_names), gridspec_kw={'hspace': 0.}, figsize=(8, 6))
nbins = 100
bounds = [[210, 280], [-5, 105], [-0.0005, 0.0065]]
for j, vn in enumerate(variable_names):
    variable = data[vn]
    min_val, max_val = bounds[j]
    bins = (max_val - min_val) * np.arange(nbins) / nbins + min_val
    for i, l in enumerate(range(1, 13)):
        selection = data[vn].isel(lev=-l)
        ax[i, j].hist(selection.values.ravel(), bins=bins)
        ax[i, j].set(yticklabels=[], yticks=[])
        if i < 11:
            ax[i, j].set(xticklabels=[], xticks=[])


for i in range(len(ax)):
    ax[i, 0].set(xlim=bounds[0])
    ax[i, 1].set(xlim=bounds[1])
    ax[i, 2].set(xlim=bounds[2])

ax[-1, 0].set(xlabel='Temperature [K]')
ax[-1, 1].set(xlabel='Relative humidity [%]')
ax[-1, 2].set(xlabel='Water vapor mixing ratio')
plt.tight_layout()
plt.show()

print('Finished')