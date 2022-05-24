import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

results_root_path = '/home/hoehlein/PycharmProjects/results/fvsrn'
variable_names = ['tk', 'rh']
fig, ax = plt.subplots(2, 2, figsize=(8, 8), sharex='col', sharey='row')

for i, variable_name in enumerate(variable_names):
    data = pd.read_csv(
        os.path.join(results_root_path, 'normalization', 'single_member', variable_name, 'accuracies.csv'))
    for normalization in ['level', 'local', 'global']:
        norm_data = data.loc[data['normalization'] == normalization, :]
        ax[i, 0].scatter(norm_data['l1'], norm_data['l1r'], label=f'{normalization} scaling')
        ax[i, 1].scatter(np.sqrt(norm_data['l2']), np.sqrt(norm_data['l2r']), label=f'{normalization} scaling')
    ax[i, 0].legend()
    ax[i, 1].legend()
    ax[i, 0].set(ylabel='MAE (rescaled)', xscale='log', yscale='log', title=f'Variable {variable_name}')
    ax[i, 1].set(ylabel='RMSE (rescaled)', xscale='log', yscale='log', title=f'Variable {variable_name}')

ax[-1, 0].set(xlabel='MAE')
ax[-1, 1].set(xlabel='RMSE')


plt.tight_layout()
plt.show()
