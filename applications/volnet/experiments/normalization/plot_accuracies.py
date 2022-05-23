import os

import matplotlib.pyplot as plt
import pandas as pd

results_root_path = '/home/hoehlein/PycharmProjects/results/fvsrn'
data = pd.read_csv(os.path.join(results_root_path, 'normalization', 'single_member', 'accuracies.csv'))

fig, ax = plt.subplots(1, 2, figsize=(10,5))
for normalization in ['level', 'local', 'global']:
    norm_data = data.loc[data['normalization'] == normalization, :]
    ax[0].scatter(norm_data['l1'], norm_data['l1r'], label=f'{normalization} scaling')
    ax[1].scatter(norm_data['l2'], norm_data['l2r'], label=f'{normalization} scaling')
ax[0].legend()
ax[0].set(xlabel='L1', ylabel='L1 rescaled', xscale='log', yscale='log')
ax[1].legend()
ax[1].set(xlabel='L2', ylabel='L2 rescaled', xscale='log', yscale='log')
plt.tight_layout()
plt.show()
