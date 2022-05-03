import os

import h5py
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from torch import nn

experiment_name = 'rescaled_ensemble/overfitting/multi_member_multi_grid_h'
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


def read_data():
    hdf5_files = sorted(os.listdir(hdf5_directory))
    parameter_counts = []
    best_losses = {loss_key: [] for loss_key in loss_keys}
    for current_file_name in hdf5_files:
        try:
            current_file = h5py.File(os.path.join(hdf5_directory, current_file_name), mode='r')
        except Exception as err:
            continue
        else:
            run_name = os.path.splitext(current_file_name)[0]
            parameter_counts.append(get_model_size(run_name))
            for loss_key in loss_keys:
                best_losses[loss_key].append(get_best_loss(current_file, loss_key))
            current_file.close()
    return pd.DataFrame({'num_params': parameter_counts, **{f'best_loss_{loss_key}': best_losses[loss_key] for loss_key in loss_keys}})


def plot_data(data: pd.DataFrame, normalization=1.):
    fig, ax = plt.subplots(1, len(loss_keys), figsize=(5*len(loss_keys), 6))
    fig.suptitle(f'Experiment: {experiment_name}')

    for i, loss_key in enumerate(loss_keys):
        ax[i].scatter(data['num_params'].values / normalization, data[f'best_loss_{loss_key}'].values)
        ax[i].set(xscale='log', yscale='log', title=loss_key, xlabel='#parameters / data size', ylabel=f'{loss_key} loss')

    plt.tight_layout()
    plt.show()


def main():
    data = read_data()
    plot_data(data, normalization=64*12*250*352)


if __name__ == '__main__':
    main()
