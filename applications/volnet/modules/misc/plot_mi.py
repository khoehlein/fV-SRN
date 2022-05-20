import os

import pandas as pd
from matplotlib import pyplot as plt

from volnet.modules.misc.compute_mi import MeasureKey


DIMENSION_LABELS = ['lat', 'lon', 'z']
NORMALIZATION_KEYS = ['global', 'level', 'local']
MEASURE_KEYS = [MeasureKey.MI, MeasureKey.PEARSON, MeasureKey.SPEARMAN, MeasureKey.KENDALL]

def load_data(variable_name: str, normalization: str):
    return pd.read_csv(f'/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/measures_{variable_name}_{normalization}.csv')


def plot_measures_for_variable(variable_name: str):
    fig, axs = plt.subplots(4, 3, figsize=(20, 15), sharex='all', sharey='row', dpi=300)
    fig.suptitle(f'Variable: {variable_name}')
    for n, normalization_key in enumerate(NORMALIZATION_KEYS):
        all_data = load_data(variable_name, normalization_key)
        members = all_data.loc[:, 'member']
        for i, member in enumerate(members.unique()[:64]):
            data = all_data.loc[members == member, :]
            for d, dim_label in enumerate(DIMENSION_LABELS):
                selection = data.loc[:, 'dimension'] == d
                delay = data.loc[selection, 'delay']
                for m, measure_key in enumerate(MEASURE_KEYS):
                    if d == 0:
                        axs[m, n].set_prop_cycle(None)
                    values = data.loc[selection, measure_key].values
                    axs[m, n].plot(delay.values / (len(delay.values) + 1), values, label=f'dim = {dim_label}', alpha=0.2)
                    axs[m, n].set(title=f'{normalization_key} normalization: {measure_key}')
        for m, _ in enumerate(MEASURE_KEYS):
            if m == 0:
                axs[m, n].set(yscale='log', ylim=[0.01, 10])
            if m > 0:
                axs[m, n].axhline(y=0, c='k')
                axs[m, n].set(ylim=[-1.05, 1.05])
        axs[-1, n].set(xlabel='Stride along dimension (rescaled)')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    for variable_name in ['tk', 'rh', 'qv', 'z', 'dbz', 'qhydro']:
        plot_measures_for_variable(variable_name)
