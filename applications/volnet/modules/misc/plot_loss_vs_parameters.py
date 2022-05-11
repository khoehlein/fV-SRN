import os

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

experiment_name = 'rescaled_ensemble/single_member_grid_size_comparison'
loss_keys = ['l1', 'l2', 'total']
base_directory = f'/home/hoehlein/PycharmProjects/results/fvsrn/{experiment_name}'

results_directory = os.path.join(base_directory, 'results')
hdf5_directory = os.path.join(results_directory, 'hdf5')
model_directory = os.path.join(results_directory, 'model')


def load_sample_model_for_run(run_name: str):
    run_directory = os.path.join(model_directory, run_name)
    checkpoint_name = next(iter(f for f in os.listdir(run_directory) if f.endswith('.pth')))
    model = torch.load(os.path.join(run_directory, checkpoint_name), map_location='cpu')
    return model['model']


def compute_parameter_count(model: nn.Module):
    return sum(p.numel() for p in model.parameters())


def get_model_size(run_name: str):
    model = load_sample_model_for_run(run_name)
    return compute_parameter_count(model)


def get_best_loss(h5_file: h5py.File, loss_key: str):
    losses = h5_file[loss_key][...]
    return np.min(losses)


def summarize_loss_data(losses: np.ndarray):
    min_loss_idx = np.argmin(losses)
    min_loss = losses[min_loss_idx]
    final_loss = losses[-1]
    return min_loss, min_loss_idx, final_loss


def read_data():
    hdf5_files = sorted(os.listdir(hdf5_directory))
    data = []
    for current_file_name in hdf5_files:
        try:
            current_file = h5py.File(os.path.join(hdf5_directory, current_file_name), mode='r')
        except Exception as err:
            continue
        else:
            run_name = os.path.splitext(current_file_name)[0]
            loss_summary = {}
            for loss_key in loss_keys:
                min_loss, min_loss_idx, final_loss = summarize_loss_data(current_file[loss_key][...])
                loss_summary.update({
                    f'{loss_key}:min_val': min_loss,
                    f'{loss_key}:min_idx': min_loss_idx,
                    f'{loss_key}:final_val': final_loss,
                })
            file_data = {
                'run_name': run_name,
                'num_params': get_model_size(run_name),
                **current_file.attrs,
                **loss_summary
            }
            data.append(file_data)
            current_file.close()
    data = pd.DataFrame(data)
    return data


def plot_data(data: pd.DataFrame, normalization=1.):
    fig, ax = plt.subplots(2, len(loss_keys), figsize=(5*len(loss_keys), 8),sharex='all', sharey='col')
    fig.suptitle(f'Experiment: {experiment_name}')

    num_channels = data['network:latent_features:volume:num_channels'].values[:, None]
    channel_values = np.unique(num_channels)
    channel_values = channel_values[None, :]
    channel_class = (num_channels == channel_values)

    num_nodes = np.array([int(s.split(':')[1]) for s in data['network:latent_features:volume:grid:resolution']])[:, None]
    node_values = np.sort(np.unique(num_nodes))
    node_values = node_values[None, :]
    node_class = (num_nodes == node_values)

    num_nodes_2 = np.array([int(s.split(':')[-1]) for s in data['network:latent_features:volume:grid:resolution']])[:, None]
    node_values_2 = np.sort(np.unique(num_nodes_2))[None, :]

    num_levels = np.array([int(s.split(':')[0]) for s in data['network:latent_features:volume:grid:resolution']])

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
            ax[j, i].set(xscale='log', yscale='log', title=loss_key, xlabel='#parameters / data size', ylabel=f'{loss_key} loss')
            ax[j, i].legend()

    plt.tight_layout()
    plt.show()


def plot_data_pretty(data: pd.DataFrame, normalization=1.):

    fig, ax = plt.subplots(1, 1, figsize=(5, 6))
    fig.suptitle(f'Model complexity vs. reconstruction accuracy')

    abscissa = data['num_parameters'].values / normalization
    x = np.logspace(np.log(np.min(abscissa)), np.log(np.max(abscissa)), 100, base=np.exp(1.))
    loss_key = 'l1'
    ax.plot(x[:75], .003 / x[:75] ** 0.4, color='k')
    ax.plot(x[60:], .0045 / x[60:], color='k')
    ax.scatter(abscissa, data[f'{loss_key}:min_val'].values, c=np.log(abscissa))
    ax.set(xscale='log', yscale='log', xlabel='Compression ratio', ylabel=f'Reconstruction accuracy')
    plt.text(6.e-2, 6e-3, '$\propto r^{\,0.4}$')
    plt.text(7.e-2, 7e-2, '$\propto r}$')
    plt.tight_layout()
    plt.show()


def main():
    data = read_data()
    plot_data(data, normalization=12*250*352)


if __name__ == '__main__':
    main()
