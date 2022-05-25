import os

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

from volnet.modules.misc import experiment_loading

loss_keys = ['l1', 'l2', 'total']
base_directory = f'/home/hoehlein/PycharmProjects/results/fvsrn'


def load_experiment_data(experiment_name):
    experiment_directory = os.path.join(base_directory, experiment_name)
    return experiment_loading.load_experiment_data(experiment_directory, loss_keys)


def plot_data(data: pd.DataFrame, resolution_key, num_channels_key, experiment_name, normalization=1.):
    fig, ax = plt.subplots(3, len(loss_keys), figsize=(5*len(loss_keys), 12),sharex='all', sharey='col')
    fig.suptitle(f'Experiment: {experiment_name}')

    num_channels = data[num_channels_key].values[:, None]
    channel_values = np.unique(num_channels)
    channel_values = channel_values[None, :]
    channel_class = (num_channels == channel_values)

    num_nodes = np.array([int(s.split(':')[1]) for s in data[resolution_key]])[:, None]
    node_values = np.sort(np.unique(num_nodes))
    node_values = node_values[None, :]
    node_class = (num_nodes == node_values)

    num_nodes_2 = np.array([int(s.split(':')[-1]) for s in data[resolution_key]])[:, None]
    node_values_2 = np.sort(np.unique(num_nodes_2))[None, :]

    num_levels = np.array([int(s.split(':')[0]) for s in data[resolution_key]])

    for i, loss_key in enumerate(loss_keys):
        abscissa = data['num_params'].values / normalization
        ordinate = data[f'{loss_key}:min_val'].values
        for j in range(channel_class.shape[-1]):
            for k in range(node_class.shape[-1]):
                mask = np.logical_and(channel_class[:, j], node_class[:, k])
                order = np.argsort(abscissa[mask])
                ax[j, i].plot(
                    abscissa[mask][order], ordinate[mask][order],
                    color='k', alpha=0.5
                )
                ax[j, i].scatter(
                    abscissa[mask][order], ordinate[mask][order],
                    s=num_levels[mask][order] ** 2, alpha=0.5,
                    label=f'channels={channel_values[0, j]}, nodes=({node_values[0, k]},{node_values_2[0, k]})'
                )
                ax[-1, i].plot(
                    abscissa[mask][order], ordinate[mask][order],
                    color='k', alpha=0.5
                )
                ax[-1, i].scatter(
                    abscissa[mask][order], ordinate[mask][order],
                    s=num_levels[mask][order] ** 2, alpha=0.5,
                    label=f'channels={channel_values[0, j]}, nodes=({node_values[0, k]},{node_values_2[0, k]})'
                )
            ax[j, i].set(xscale='log', yscale='log', title=loss_key, xlabel='#parameters / data size', ylabel=f'{loss_key} loss')
            ax[j, i].legend()
        ax[-1, i].set(xscale='log', yscale='log', title=loss_key, xlabel='#parameters / data size', ylabel=f'{loss_key} loss')
        # ax[-1, i].legend()


    plt.tight_layout()
    plt.show()


def plot_ensemble_experiment(experiment_name):
    data = load_experiment_data(experiment_name=experiment_name)
    plot_data(
        data,
        'network:latent_features:ensemble:grid:resolution', 'network:latent_features:ensemble:num_channels',
        experiment_name,
        normalization=12*250*352
    )


def plot_singleton_experiment(experiment_name, num_members=1):
    data = load_experiment_data(experiment_name=experiment_name)
    plot_data(
        data,
        'network:latent_features:volume:grid:resolution', 'network:latent_features:volume:num_channels',
        experiment_name,
        normalization=12*250*352*num_members
    )


def draw_data_to_axes(data, axs, resolution_key, num_channels_key, normalization,alpha=0.1):
    num_channels = data[num_channels_key].values[:, None]
    channel_values = np.unique(num_channels)
    channel_values = channel_values[None, :]
    channel_class = (num_channels == channel_values)

    num_nodes = np.array([int(s.split(':')[1]) for s in data[resolution_key]])[:, None]
    node_values = np.sort(np.unique(num_nodes))
    node_values = node_values[None, :]
    node_class = (num_nodes == node_values)

    num_nodes_2 = np.array([int(s.split(':')[-1]) for s in data[resolution_key]])[:, None]
    node_values_2 = np.sort(np.unique(num_nodes_2))[None, :]

    num_levels = np.array([int(s.split(':')[0]) for s in data[resolution_key]])

    for i, loss_key in enumerate(loss_keys):
        abscissa = data['num_params'].values / normalization
        ordinate = data[f'{loss_key}:min_val'].values
        for j in range(channel_class.shape[-1]):
            for k in range(node_class.shape[-1]):
                mask = np.logical_and(channel_class[:, j], node_class[:, k])
                order = np.argsort(abscissa[mask])
                axs[i].plot(
                    abscissa[mask][order], ordinate[mask][order],
                    color='k', alpha=alpha
                )
                axs[i].scatter(
                    abscissa[mask][order], ordinate[mask][order],
                    s=num_levels[mask][order] ** 2, alpha=alpha,
                    label=f'channels={channel_values[0, j]}, nodes=({node_values[0, k]},{node_values_2[0, k]})'
                )
        axs[i].set(xscale='log', yscale='log', title=loss_key, xlabel='#parameters / data size',
                       ylabel=f'{loss_key} loss')


def plot_ensemble_progression(normalization=1.):
    fig, axs = plt.subplots(1, 3, figsize=(5 * len(loss_keys), 4), sharex='all', dpi=300)

    experiment_name = 'rescaled_ensemble/single_member_grid_size_comparison'
    data = load_experiment_data(experiment_name)
    draw_data_to_axes(
        data, axs,
        'network:latent_features:volume:grid:resolution', 'network:latent_features:volume:num_channels',
        normalization=normalization
    )

    for ensemble_size in [2, 4, 8]:
        for ax in axs:
            ax.set_prop_cycle(None)
        experiment_name = f'rescaled_ensemble/multi_member_grid_size_comparison/{ensemble_size}m'
        data = load_experiment_data(experiment_name)
        draw_data_to_axes(
            data, axs,
            'network:latent_features:ensemble:grid:resolution', 'network:latent_features:ensemble:num_channels',
            normalization=normalization * ensemble_size
        )

    axs[0].set(yticks=[1.e-3, 2.e-3, 5.e-3, 1.e-2, 2.e-2, 5.e-2])

    plt.tight_layout()
    plt.show()


def main():
    plot_singleton_experiment('/home/hoehlein/PycharmProjects/results/fvsrn/paper/single_member/grid_params/tk/vres', num_members=1)
    # plot_ensemble_experiment('rescaled_ensemble/multi_member_grid_size_comparison/2m')
    # plot_singleton_experiment('rescaled_ensemble/single_member_grid_size_comparison')
    # plot_ensemble_progression(normalization=12*352*250)


if __name__ == '__main__':
    main()
