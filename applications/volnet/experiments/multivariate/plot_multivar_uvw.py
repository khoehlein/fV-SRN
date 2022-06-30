import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

RESOLUTION_KEY = 'network:latent_features:volume:grid:resolution'
CORE_KEY = 'network:core:layer_sizes'


pth = 'model_epoch_250.pth'


def plot_multivariate_data(ax):
    path = '/home/hoehlein/PycharmProjects/results/fvsrn/multi-variate/single-member/parameter_interplay/stats/run_statistics.csv'
    data = pd.read_csv(path)
    data['checkpoint'] = [os.path.split(c)[-1] for c in data['checkpoint']]
    data = data.loc[data['checkpoint'] == pth, :]
    variables = np.sort(np.unique(data['variable']))
    for i, variable_name in enumerate(variables):
        sel1 = data.loc[data['variable'] == variable_name]
        resolutions = np.unique(sel1[RESOLUTION_KEY])
        for r in resolutions:
            sel2 = sel1.loc[sel1[RESOLUTION_KEY] == r, :]
            layer_sizes = np.unique(sel1[CORE_KEY])
            for ls in layer_sizes:
                sel = sel2.loc[sel2[CORE_KEY] == ls, :]
                compression_ratio = (352 * 250 * 12 * 3) / sel['num_parameters']
                order = np.argsort(compression_ratio)
                ax[0, i].plot(compression_ratio.values[order], sel['rmse_reverted'].values[order], label=f'multivariate {r} {ls}', marker='.')
                ax[1, i].plot(compression_ratio.values[order], sel['dssim_reverted'].values[order], label=f'multivariate {r} {ls}', marker='.')

def plot_univariate_data(ax):
    path = '/home/hoehlein/PycharmProjects/results/fvsrn/multi-variate/single-member/parameter_interplay_univariate/stats/run_statistics.csv'
    data = pd.read_csv(path)
    data['checkpoint'] = [os.path.split(c)[-1] for c in data['checkpoint']]
    data = data.loc[data['checkpoint'] == 'model_epoch_250.pth', :]
    variables = np.sort(np.unique(data['variable']))
    for i, variable_name in enumerate(variables):
        sel1 = data.loc[data['variable'] == variable_name]
        resolutions = np.unique(sel1[RESOLUTION_KEY])
        for r in resolutions:
            sel2 = sel1.loc[sel1[RESOLUTION_KEY] == r, :]
            layer_sizes = np.unique(sel1[CORE_KEY])
            for ls in layer_sizes:
                sel = sel2.loc[sel2[CORE_KEY] == ls, :]
                compression_ratio = (352 * 250 * 12) / sel['num_parameters']
                order = np.argsort(compression_ratio)
                ax[0, i].plot(compression_ratio.values[order], sel['rmse_reverted'].values[order], label=f'univariate {r} {ls}', linestyle='--', marker='.')
                ax[1, i].plot(compression_ratio.values[order], sel['dssim_reverted'].values[order], label=f'univariate {r} {ls}', linestyle='--', marker='.')


def main():
    fix, axs = plt.subplots(2, 3, sharex='all', figsize=(20,10))
    plot_multivariate_data(axs)
    plot_univariate_data(axs)
    for ax in axs[0]:
        ax.set(xscale='log',yscale='log')
    for ax in axs[1]:
        ax.set(xscale='log')
    axs[0, 0].legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
