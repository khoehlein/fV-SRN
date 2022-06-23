import os

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt

from data.necker_ensemble.single_variable import load_ensemble


variable_names = ['tk', 'rh', 'qv']
dim_names = ['lat', 'lon', 'lev']

fig, ax = plt.subplots(12, len(variable_names), gridspec_kw={'hspace': 0.}, figsize=(8, 6))
nbins = 100
bounds = [[210, 280], [-5, 105], [-0.0005, 0.0065]]
for j, vn in enumerate(variable_names):
    variable = load_ensemble('global-min-max', vn, time=4, min_member=1, max_member=2)
    variable = xr.DataArray(
        variable[0][0],
        dims=dim_names,
        coords={
            dim: (dim, np.arange(l))
            for dim, l in zip(dim_names, variable[0][0].shape)
        }
    )
    min_val, max_val = bounds[j]
    bins = (max_val - min_val) * np.arange(nbins) / nbins + min_val
    for i, l in enumerate(range(1, 13)):
        selection = variable.isel(lev=-l)
        ax[i, j].hist(selection.values.ravel(), bins=bins)
        ax[i, j].set(yticklabels=[], yticks=[])
        if i < 11:
            ax[i, j].set(xticklabels=[], xticks=[])
        if j == 0:
            ax[i, 0].set(ylabel=f'lev {i}')


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