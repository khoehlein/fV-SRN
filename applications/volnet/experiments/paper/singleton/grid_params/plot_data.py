import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIRECTORY = '/home/hoehlein/PycharmProjects/results/fvsrn'
EXPERIMENT_NAME = 'paper/single_member/grid_params'
# PARAMETER_NAME = 'tk/hres'
CHECKPOINT_NAME = 'model_epoch_250.pth'

directory = os.path.join(ROOT_DIRECTORY, EXPERIMENT_NAME)
variable_names = ['tk/hres', 'rh/hres']
pn = 12 * 352 * 250
fig, ax = plt.subplots(len(variable_names), 2, figsize=(8, 6), sharex='all', gridspec_kw={'hspace': 0.})

for i, variable_name in enumerate(variable_names):
    data = pd.read_csv(
        os.path.join(directory, variable_name, 'accuracies.csv'))
    for normalization in ['level-min-max_scaling', 'local-min-max_scaling', 'global-min-max_scaling']:
        norm_data = data.loc[data['normalization'] == normalization, :]
        ax[0, i].scatter(norm_data['num_parameters'] / pn, norm_data['l1'], label=f'{normalization} scaling')
        ax[1, i].scatter(norm_data['num_parameters'] / pn, norm_data['l1r'], label=f'{normalization} scaling')
    ax[0, i].legend()
    ax[1, i].legend()
    ax[0, i].set(ylabel='MAE (rescaled)', xscale='log', yscale='log')
    ax[1, i].set(ylabel='MAE (original)', xscale='log', yscale='log')

for i, vn in enumerate(variable_names):
    ax[0, i].set(title=f'Parameter {vn}')
    ax[-1, i].set(xlabel='#parameters')


plt.tight_layout()
plt.show()