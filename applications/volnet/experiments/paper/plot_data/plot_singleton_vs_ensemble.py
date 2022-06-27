import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from compression.experiments.plot_compression_stats import draw_compressor_stats
from volnet.analysis.plot_retraining_data import get_member_count, get_channel_count
from volnet.experiments.paper.singleton.grid_params.plot_parameter_interplay import plot_single_member_data


def load_multi_core_data():
    base_folder = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_core/num_channels'
    configurations = sorted(os.listdir(base_folder))
    return {
        c: pd.read_csv(os.path.join(base_folder, c, 'stats', 'run_statistics.csv'))
        for c in configurations if get_member_count(c) == 64 and get_channel_count(c) == 64
    }


def plot_multi_core_data(ax):
    data = load_multi_core_data()
    for configuration in data:
        data_reduced = data[configuration]
        data_reduced['checkpoint'] = [os.path.split(c)[-1] for c in data_reduced['checkpoint']]
        checkpoints = np.sort(np.unique(data_reduced['checkpoint']))
        valid = data_reduced['checkpoint'] == checkpoints[-1]
        print(np.sum(valid))
        data_reduced = data_reduced.loc[valid,:]
        compression_rate = (352 * 250 * 12) * data_reduced['num_members'] / data_reduced['num_parameters']
        loss = data_reduced['rmse_reverted']
        ax[0].plot(compression_rate.loc[compression_rate > 1.], loss.loc[compression_rate > 1.])
        loss = data_reduced['dssim_reverted']
        ax[1].plot(compression_rate[compression_rate > 1.], loss[compression_rate > 1.])
    ax[0].set(xscale='log', yscale='log', title='multi-decoder')
    ax[1].set(xscale='log')
    ax[0].grid()
    ax[1].grid()


def test():
    plot_multi_core_data(None)


def main():
    fig, axs = plt.subplots(2, 4, sharex='all', figsize=(8,4))
    plot_multi_core_data(axs[:, 3])
    plot_single_member_data(axs[:, 3])
    draw_compressor_stats(axs[:, :3], ['level'], 'reverted')
    for i, ax in enumerate(axs[0]):
        ax.set(ylim=(2.e-8, 2.e0))
        if i > 0:
            ax.set(yticklabels=[])
    for i, ax in enumerate(axs[1]):
        ax.set(ylim=(0.49, 1.1), xlabel='compression ratio')
        if i > 0:
            ax.set(yticklabels=[])
    axs[0, 0].set(ylabel='RMSE (original)')
    axs[1, 0].set(ylabel='DSSIM (original)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()