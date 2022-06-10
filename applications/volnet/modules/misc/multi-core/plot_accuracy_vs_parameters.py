import os

import numpy as np
from matplotlib import pyplot as plt

from volnet.modules.misc.experiment_loading import load_experiment_data


experiment_directory = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_core/num_channels'
sub_experiments = [
    '6-44-31_32_1-65_fast',
    '6-44-31_64_1-65_fast',
    '12-176-125_32_1-65_fast',
    '12-176-125_64_1-65_fast',
]

loss_keys = ['l1', 'l2']

data = [load_experiment_data(os.path.join(experiment_directory, se), loss_keys) for se in sub_experiments]

fig, ax = plt.subplots(1, 1)

for d in data:
    label = '64 mem., {} res., {} channels'.format(d.loc[0, 'network:latent_features:volume:grid:resolution'], d.loc[0, 'network:core:layer_sizes'].split(':')[0])
    ax.plot(d.loc[:, 'num_params'] / (352 * 12 * 250 * 64 * 2), np.sqrt(d.loc[:, 'l2:min_val']), label=label)

sub_experiments = [
    '6-44-31_32_1-33_fast',
    '6-44-31_64_1-33_fast',
]

data = [load_experiment_data(os.path.join(experiment_directory, se), loss_keys) for se in sub_experiments]

for d in data:
    label = '32 mem., {} res., {} channels'.format(d.loc[0, 'network:latent_features:volume:grid:resolution'], d.loc[0, 'network:core:layer_sizes'].split(':')[0])
    ax.plot(d.loc[:, 'num_params'] / (352 * 12 * 250 * 32 * 2), np.sqrt(d.loc[:, 'l2:min_val']), label=label)

ax.set(xscale='log', yscale='log', xlabel='#parameters (normalized)', ylabel='RMSE (rescaled)')
plt.legend()
plt.tight_layout()
plt.show()


print('Finished')