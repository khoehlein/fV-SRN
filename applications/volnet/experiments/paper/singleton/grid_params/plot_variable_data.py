import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIRECTORY = '/home/hoehlein/PycharmProjects/results/fvsrn'
EXPERIMENT_NAME = 'paper/single_member/grid_params'
CHECKPOINT_NAME = 'model_epoch_250.pth'

directory = os.path.join(ROOT_DIRECTORY, EXPERIMENT_NAME)
variable_name = 'qv'
parameters = ['hres', 'vres', 'num_channels']
loss = 'mse'

pn = 12 * 352 * 250
fig, ax = plt.subplots(2, len(parameters), figsize=(8, 4), gridspec_kw={'hspace': 0.}, sharey='row', dpi=300)
fig.suptitle(f'Parameter {variable_name}')

labels = [[0.01, 0.1, 1], [0.02, 0.05, 0.1, 0.2], [0.02, 0.05, 0.1]]
subplot_titles ={
    'hres': 'Horizontal resolution',
    'vres': 'Vertical resolution',
    'num_channels': 'Channel number'
}
loss_titles = {
    'mae': 'MAE',
    'mse': 'MSE',
    'rmse': 'RMSE',
}
loss_keys = {
    'mae': 'l1',
    'mse': 'l2',
    'rmse': 'l2',
}

for i, p in enumerate(parameters):
    ax[0, i].set_yscale('log')
    ax[1, i].set_yscale('log')
    ax[0, i].set(title=subplot_titles[p], xticks=[], xticklabels=[])
    ax[1, i].set(xlabel='#parameters (normalized)', xticks=np.log(labels[i]), xticklabels=labels[i])
    data = pd.read_csv(
        os.path.join(directory, variable_name, p, 'accuracies.csv'))
    for normalization in ['level-min-max_scaling', 'local-min-max_scaling', 'global-min-max_scaling']:
        norm_data = data.loc[data['normalization'] == normalization, :]
        if loss == 'rmse':
            norm_data['l2'] = np.sqrt(norm_data['l2'])
            norm_data['l2r'] = np.sqrt(norm_data['l2r'])
        norm_data_mean = norm_data.groupby('num_parameters').mean()
        norm_data_std = norm_data.groupby('num_parameters').std()
        # ax[0, i].scatter(norm_data_mean.index.values / pn, norm_data_mean['l1'].values, label=normalization.split('-')[0])
        # ax[1, i].scatter(norm_data_mean.index.values / pn, norm_data_mean['l1r'].values, label=normalization.split('-')[0])
        ax[0, i].errorbar(np.log(norm_data_mean.index / pn), norm_data_mean[loss_keys[loss]], yerr=norm_data_std[loss_keys[loss]], label=normalization.split('-')[0])
        ax[1, i].errorbar(np.log(norm_data_mean.index / pn), norm_data_mean[loss_keys[loss] + 'r'], yerr=norm_data_std[loss_keys[loss] + 'r'], label=normalization.split('-')[0])

ax[0, 0].set(ylabel=loss_titles[loss] + ' (rescaled)')
ax[1, 0].set(ylabel=loss_titles[loss] + ' (original)')
ax[0, -1].legend()
plt.tight_layout()
        # ax[-1, i].set(xticks=labels[i], xticklabels=labels[i])

plt.show()

