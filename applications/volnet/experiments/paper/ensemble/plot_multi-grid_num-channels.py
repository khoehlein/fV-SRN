import os

import matplotlib.pyplot as plt
import numpy as np

from volnet.analysis.plot_retraining_data import get_member_count, get_label
from volnet.modules.misc.experiment_loading import load_experiment_data


def main():
    loss_keys = ['l2']
    base_folder = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_grid/num_channels'
    run_names = sorted(os.listdir(base_folder))
    fig, ax = plt.subplots(1, 1)
    for run_name in run_names:
        path = os.path.join(base_folder, run_name)
        data = load_experiment_data(path, loss_keys)
        compression = (352 * 250 * 12 * get_member_count(run_name)) / data['num_params']
        ax.plot(compression, np.sqrt(data['l2:min_val']), label=get_label(run_name), marker='.')
    ax.set(xscale='log', yscale='log', xlabel='compression ratio', ylabel='RMSE (rescaled)')
    ax.legend()
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    main()
