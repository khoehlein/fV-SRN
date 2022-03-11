from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def test_plotting_function(num_trials: Optional[int] = 10):
    parameter_names = ['tcc', 't2m', 'u10', 'v10', 'tp', 'tisr']
    num_parameters = len(parameter_names)
    # get some randomized ranks
    # has shape (num_trials, num_parameters)
    # this can be updated to match the true data
    observed_rankings = np.array([
        np.random.permutation(num_parameters)
        for _ in range(num_trials)
    ])
    get_plot(parameter_names, observed_rankings)


def read_loss_directory(csv_directory, count_values, parameter_names):
    import os
    import pandas as pd
    observed_losses = np.zeros((len(count_values), len(parameter_names)))
    for i, cv in enumerate(count_values):
        for j, pn in enumerate(parameter_names):
            file_name = os.path.join(csv_directory, 'losses_{}_{:04d}.csv'.format(pn, cv))
            observed_losses[i, j] = pd.read_csv(file_name, header=0)['Value'].values[-1]
    return observed_losses


def plot_loss_rankings(csv_directory: str):

    count_values = [1, 2, 3, 4]
    parameter_names = ['tcc', 't2m', 'u10', 'v10', 'tp', 'tisr']

    observed_losses = read_loss_directory(csv_directory,count_values,parameter_names)

    observed_rankings = np.argsort(observed_losses, axis=-1)

    get_plot(parameter_names, observed_rankings)


def get_plot(labels: List[str], rankings: np.ndarray, bar_width=0.35):
    num_labels = len(labels)
    num_trials = rankings.shape[0]
    assert rankings.shape[-1] == num_labels, \
        '[ERROR] Number of ranks does not match the number of labels'

    rank_axis = np.arange(num_labels)

    counts_per_rank = np.sum(
        rank_axis[:, None, None] == rankings[None, ...],
        axis=1
    )  # has shape (num_ranks, num_parameters), whereby num_ranks = num_parameters

    # check that the count contributions per parameter sum up to the number of trials
    assert np.all(np.sum(counts_per_rank, axis=-1) == num_trials), \
        '[ERROR] Sum of rank counts is not consistent with the number of trials. Probably something is wrong with the rankings.'

    labeled_counts_per_rank = {
        l: counts_per_rank[:, i]
        for i, l in enumerate(labels)
    }

    fig, ax = plt.subplots()
    bottom = np.zeros_like(rank_axis)
    for l in labels:
        frequency = labeled_counts_per_rank[l]
        ax.bar(rank_axis, frequency, bar_width, label=l, bottom=bottom)
        bottom = bottom + frequency

    ax.set_xlabel('Ranks')
    ax.set_ylabel('Number of trials')
    ax.set_yticks(np.arange(num_trials + 1).astype(int).tolist())
    # ax.set_title('Ranks per Configuration')
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    csv_directory = 'C:\\Users\\hoehlein\\Downloads\\losses'
    plot_loss_rankings(csv_directory)
