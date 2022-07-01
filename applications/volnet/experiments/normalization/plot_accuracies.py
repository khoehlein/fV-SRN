import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from volnet.analysis.plot_retraining_data import plot_diagonal


def plot_old_figure():
    results_root_path = '/home/hoehlein/PycharmProjects/results/fvsrn'
    variable_names = ['tk', 'rh']
    pn = 12 * 352 * 250
    fig, ax = plt.subplots(len(variable_names), 2, figsize=(8, 6), sharex='all', gridspec_kw={'hspace': 0.})

    for i, variable_name in enumerate(variable_names):
        data = pd.read_csv(
            os.path.join(results_root_path, 'normalization', 'single_member', variable_name, 'accuracies.csv'))
        for normalization in ['level', 'local', 'global']:
            norm_data = data.loc[data['normalization'] == normalization, :]
            ax[0, i].plot(norm_data['num_parameters'] / pn, norm_data['l1'], label=f'{normalization} scaling', marker='.')
            ax[1, i].plot(norm_data['num_parameters'] / pn, norm_data['l1r'], label=f'{normalization} scaling', marker='.')
        ax[0, i].legend()
        ax[1, i].legend()
        ax[0, i].set(ylabel='MAE (rescaled)', xscale='log', yscale='log')
        ax[1, i].set(ylabel='MAE (original)', xscale='log', yscale='log')

    for i, vn in enumerate(variable_names):
        ax[0, i].set(title=f'Parameter {vn}')
        ax[-1, i].set(xlabel='#parameters')


    plt.tight_layout()
    plt.show()


def plot_new_figure():
    results_root_path = '/home/hoehlein/PycharmProjects/results/fvsrn'
    variable_names = ['tk', 'rh']
    pn = 12 * 352 * 250
    fig, ax = plt.subplots(2, len(variable_names), figsize=(8, 4), sharex='all')

    for i, variable_name in enumerate(variable_names):
        data = pd.read_csv(
            os.path.join(results_root_path, 'normalization', 'single_member', variable_name, 'accuracies.csv'))
        for normalization in ['level', 'local', 'global']:
            norm_data = data.loc[data['normalization'] == normalization, :]
            norm_data = norm_data.loc[norm_data['network:latent_features:volume:num_channels'] == 8, :]
            for a in ['4', '8', '12']:
                lev_data = norm_data.loc[[x.startswith(a) for x in norm_data['network:latent_features:volume:grid:resolution']], :]
                lev_data = lev_data.groupby('num_parameters')
                lev_data = lev_data.mean()
                compression_ratio = pn / lev_data.index.values
                order = np.argsort(compression_ratio)
                ax[0, i].plot(compression_ratio, lev_data['rmse_rescaled'], label=f'{normalization} scaling', marker='.')
                ax[1, i].plot(compression_ratio, lev_data['rmse_reverted'], label=f'{normalization} scaling', marker='.')
        ax[0, i].set(ylabel='RMSE (rescaled)', xscale='log', yscale='log')
        ax[1, i].set(ylabel='RMSE (original)', xscale='log', yscale='log')

    ax[1, 1].legend()

    for i, vn in enumerate(variable_names):
        ax[0, i].set(title=f'Parameter {vn}')
        ax[-1, i].set(xlabel='#parameters')

    plt.tight_layout()
    plt.show()


def plot_new_figure_2():
    results_root_path = '/home/hoehlein/PycharmProjects/results/fvsrn'
    variable_names = ['tk', 'qv','rh']
    fig, ax = plt.subplots(1, len(variable_names), figsize=(8, 3), sharex='col')

    for i, variable_name in enumerate(variable_names):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        data = pd.read_csv(
            os.path.join(results_root_path, 'normalization', 'single_member', variable_name, 'accuracies.csv'))
        data = data.groupby(by=['normalization', 'network:latent_features:volume:num_channels', 'network:latent_features:volume:grid:resolution'])
        data = data.mean()
        loss = data['rmse_reverted'].unstack(level='normalization')
        compression = data['compression_ratio'].unstack(level='normalization')
        ax[i].plot([loss['global'].values, loss['global'].values], [loss['global'].values, loss['level'].values], c=colors[0], linestyle='dotted')
        ax[i].scatter(loss['global'], loss['level'], label='level norm', s=(200. / compression['global']))
        ax[i].plot([loss['global'].values, loss['global'].values], [loss['global'].values, loss['local'].values], c=colors[1], linestyle='dotted')
        ax[i].scatter(loss['global'], loss['local'], label='local norm', s=(200. / compression['global']))
        plot_diagonal(ax[i])
        ax[i].set(xscale='log', yscale='log', xlabel='RMSE (original, global norm)', title=variable_name)
        ax[i].grid()
        print('Done')
    ax[0].set(ylabel='RMSE (original)')
    ax[-1].legend()
    plt.tight_layout()
    plt.savefig('normalization-comparison.pdf')
    plt.show()


if __name__ == '__main__':
    plot_new_figure_2()
