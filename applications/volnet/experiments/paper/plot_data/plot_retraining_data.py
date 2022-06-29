import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from volnet.analysis.plot_retraining_data import get_member_count, get_channel_count
from volnet.experiments.paper.plot_data.plot_singleton_vs_ensemble import (
    load_multi_core_data as load_multi_core_baseline,
    load_multi_grid_data as load_multi_grid_baseline, get_checkpoint_data
)


def load_multi_core_retraining(num_channels=None):
    if num_channels is None:
        num_channels = 64
    base_folder = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_core/retraining'
    configurations = sorted(os.listdir(base_folder))
    return {
        c: pd.read_csv(os.path.join(base_folder, c, 'stats', 'run_statistics.csv'))
        for c in configurations if get_member_count(c) == 64 and get_channel_count(c) == num_channels
    }


def load_multi_grid_retraining(num_channels=None):
    if num_channels is None:
        num_channels = 64
    base_folder = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_grid/retraining'
    configurations = sorted(os.listdir(base_folder))
    return {
        c: pd.read_csv(os.path.join(base_folder, c, 'stats', 'run_statistics.csv'))
        for c in configurations if get_member_count(c) == 64 and get_channel_count(c) == num_channels
    }


def _filter_data(data_orig, data_ret):
    common_keys = list(set(data_ret.keys()).intersection(set(data_orig.keys())))

    def filter_data(data):
        return {key: data[key] for key in common_keys}

    data_orig = filter_data(data_orig)
    data_ret = filter_data(data_ret)

    return data_orig, data_ret


def load_multi_core_data(num_channels=None):
    return _filter_data(load_multi_core_baseline(num_channels=num_channels), load_multi_core_retraining(num_channels=num_channels))


def load_multi_grid_data(num_channels=None):
    return _filter_data(load_multi_grid_baseline(num_channels=num_channels), load_multi_grid_retraining(num_channels=num_channels))


def _plot_loss_data(data, ax, linestyle=None):
    data_orig, data_ret = data
    for configuration in sorted(data_orig.keys()):
        sel1_orig = get_checkpoint_data(data_orig[configuration])
        sel1_ret = get_checkpoint_data(data_ret[configuration])
        s = sel1_orig['num_parameters'] / (sel1_orig['num_members'] * 352 * 250 * 12)
        print(np.max(s))
        ax.scatter(sel1_orig['rmse_reverted'], sel1_ret['rmse_reverted'], s=100*s)
        ax.plot(sel1_orig['rmse_reverted'], sel1_ret['rmse_reverted'], label=get_label(configuration), linestyle=linestyle)
    ax.set(xscale='log', yscale='log')
    ax.legend(loc='lower right')


def plot_multi_core_data(ax):
    ax.set(title='multi-decoder')
    _plot_loss_data(load_multi_core_data(num_channels=64), ax)
    ax.set_prop_cycle(None)
    _plot_loss_data(load_multi_core_data(num_channels=32), ax, linestyle='--')


def plot_multi_grid_data(ax):
    ax.set(title='multi-grid')
    _plot_loss_data(load_multi_grid_data(num_channels=64), ax)
    ax.set_prop_cycle(None)
    _plot_loss_data(load_multi_grid_data(num_channels=32), ax, linestyle='--')
    # ax.set_prop_cycle(None)
    # _plot_loss_data(load_multi_grid_data(num_channels=128), ax, linestyle=':')


def get_label(run_name):
    r_code, c_code, *_ = run_name.split('_')
    r = r_code.replace('-', ':')
    c = int(c_code)
    # m = get_member_count(run_name)
    return f'R: {r}, C: {c}'


def plot_diagonal(ax):
    xlim = np.array(ax.get_xlim())
    ylim = np.array(ax.get_ylim())
    max_lims = np.fmax(xlim, ylim)
    min_lims = np.fmin(xlim, ylim)
    ax.plot([min_lims[0], min_lims[1]], [min_lims[0], min_lims[1]], ls="dotted", c="k")


def add_axis_labels(ax):
    ax.set(xlim=(9.e-3, .2), ylim=(8.e-3, .25))
    ax.set(xlabel='RMSE (original)', ylabel='RMSE (retrained)')
    plot_diagonal(ax)


def main():
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), sharex='all', sharey='all')
    plot_multi_core_data(axs[0])
    add_axis_labels(axs[0])
    plot_multi_grid_data(axs[1])
    add_axis_labels(axs[1])
    axs[1].set(ylabel=None)
    plt.tight_layout()
    plt.savefig('retraining.pdf')
    plt.show()


def test():
    fig, ax = plt.subplots(1, 1)
    plot_multi_grid_data(ax)
    plt.show()


if __name__ == '__main__':
    main()
