import os

import numpy as np
from matplotlib import pyplot as plt

from volnet.modules.misc.experiment_loading import load_experiment_data


def get_specs(run_name):
    r_code, c_code, m_code, *_ = run_name.split('_')
    r = r_code.replace('-', ':')
    c = int(c_code)
    min_m, max_m = [int(m) for m in m_code.split('-')]
    m = max_m - min_m
    return r, c, m


def get_label(run_name):
    return 'R: {}, C: {}, M: {}'.format(*get_specs(run_name))


def get_linestyle(run_name):
    c_code = run_name.split('_')[1]
    if c_code == '32':
        return 'solid'
    elif c_code == '64':
        return 'dashed'
    raise Exception()


experiment_directory = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/ensemble/multi_core/num_channels'
sub_experiments = sorted([
    '6-44-31_32_1-65_fast',
    '6-44-31_64_1-65_fast',
    '6-44-31_32_1-33_fast',
    '6-44-31_64_1-33_fast',
    '6-88-63_32_1-65_fast',
    '6-88-63_64_1-65_fast',
    '12-176-125_32_1-65_fast',
    '12-176-125_64_1-65_fast',
])

loss_keys = ['l1', 'l2']

data = {
    run_name: load_experiment_data(os.path.join(experiment_directory, run_name), loss_keys)
    for run_name in sub_experiments
}

fig, ax = plt.subplots(1, 1, dpi=300)

for run_name in data:
    r, c, m = get_specs(run_name)
    d = data[run_name]
    label = get_label(run_name)
    ax.plot(d.loc[:, 'num_params'] / (352 * 12 * 250 * m * 2), np.sqrt(d.loc[:, 'l2:min_val']), label=label, marker='.', linestyle=get_linestyle(run_name))

ax.set(xscale='log', yscale='log', xlabel='#parameters (normalized)', ylabel='RMSE (rescaled)')
plt.legend()
plt.tight_layout()
plt.show()


print('Finished')