import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

base_directory = '/home/hoehlein/PycharmProjects/results/fvsrn/multi-variate/single-member/uvw'
experiments = sorted(os.listdir(base_directory))
variables = ['u', 'v', 'w']


def load_variable_data(experiment_name):
    path = os.path.join(base_directory, experiment_name, 'stats', 'run_statistics.csv')
    data = pd.read_csv(path)
    data['checkpoint'] = [os.path.split(c)[-1] for c in data['checkpoint']]
    data = data.loc[data['checkpoint'] == 'model_epoch_250.pth']
    return data


def main():
    fig, ax = plt.subplots(1, len(variables))
    for experiment_name in experiments:
        data = load_variable_data(experiment_name)
        for i, variable in enumerate(variables):
            sel1 = data.loc[data['variable'] == variable, :]
            if len(sel1) > 0:
                resolutions = np.unique(data['network:latent_features:volume:grid:resolution'])
                for r in resolutions:
                    core_channels = np.unique(data['network:core:layer_sizes'])
                    for c in core_channels:
                        raise NotImplementedError()


if __name__ =='__main__':
    main()
