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

def draw_plots(ax, layers):
    linestyles = ['solid', 'dashed', 'dotted']
    for experiment_name in ['u', 'v', 'w', 'u-v', 'u-v-w']:
        data = load_variable_data(experiment_name)
        num_variables = len(experiment_name.split('-'))
        for i, variable in enumerate(variables):
            ax[0, i].set_prop_cycle(None)
            ax[1, i].set_prop_cycle(None)
            sel1 = data.loc[data['variable'] == variable, :]
            if len(sel1) > 0:
                resolutions = np.unique(data['network:latent_features:volume:grid:resolution'])
                for r in resolutions:
                    sel2 = sel1.loc[sel1['network:latent_features:volume:grid:resolution'] == r, :]
                    core_channels = np.unique(sel2['network:core:layer_sizes'])
                    for c in core_channels:
                        if c == layers: #'32:32:32':
                            sel3 = sel2.loc[sel2['network:core:layer_sizes'] == c, :]
                            sel3 = sel3.groupby(by='num_parameters').mean()
                            compression_ratio = (352 * 250 * 12 * num_variables) / sel3.index.values
                            order = np.argsort(compression_ratio)
                            loss = sel3['rmse_reverted'].values
                            loss = loss[order]
                            ls = linestyles[int(num_variables - 1)]
                            channels = c.split(':')[0]
                            ax[0, i].plot(compression_ratio[order], loss, label=f'R: {r}', linestyle=ls)
                            loss = sel3['dssim_reverted'].values
                            loss = loss[order]
                            ax[1, i].plot(compression_ratio[order], loss, label=f'R: {r}', linestyle=ls)
            ax[0, i].set(xscale='log', yscale='log', title=variable, ylim=(3.e-2, 8.e-1))
            ax[1, i].set(xscale='log', xlabel='compression ratio', ylim=(-0.01, 1.01))

def main():
    fig, ax = plt.subplots(2, len(variables), sharex='all', sharey='row', figsize=(6, 4))
    draw_plots(ax[:, :3], '32:32:32')
    ax[0, 0].set(ylabel='RMSE (original)')
    ax[1, 0].set(ylabel='DSSIM (original)')
    plt.tight_layout()
    # ax[1, 0].legend()
    plt.savefig('multi-parameter_models.pdf')
    plt.show()




if __name__ =='__main__':
    main()
