from math import prod

import numpy as np
import pandas as pd
import torch
import pyrenderer
from matplotlib import pyplot as plt
from scipy import stats
from volnet.modules.misc.mutual_information import mi


class MeasureKey(object):
    MI = 'mi'
    PEARSON = 'pearson'
    SPEARMAN = 'spearman'
    KENDALL = 'kendall'


def value_if_significant(x, alpha):
    r, p = x
    if p < alpha:
        return r
    return 0.

ALPHA = 0.05


def load_data(variable_name: str, norm: str, member: int):
    path = '/home/hoehlein/data/1000_member_ensemble/cvol/single_variable/{}-min-max_scaling/{}/member{:04d}/t04.cvol'.format(norm, variable_name, member)
    volume = pyrenderer.Volume(path)
    feature = volume.get_feature(0)
    level = feature.get_level(0)
    data = level.to_tensor().data.cpu().numpy()[0]
    return data


gen = np.random.Generator(np.random.PCG64(1234))


def compute_measures(data: np.ndarray):
    shape = data.shape
    all_measures = []
    NUM_SAMPLES = 1
    for i in range(len(shape)):
        for j in range(1, shape[i]):
            select_upper = [np.s_[int(j):] if k == i else np.s_[:] for k in range(len(shape))]
            select_lower = [np.s_[0:-int(j)] if k == i else np.s_[:] for k in range(len(shape))]

            upper_data = data[tuple(select_upper)].ravel()[:, None]
            lower_data = data[tuple(select_lower)].ravel()[:, None]

            num_samples = 250 * 12  # prod([s if i != k else 1 for k, s in enumerate(shape)])
            measures = {}
            for k in range(NUM_SAMPLES):
                indices = gen.choice(len(upper_data), size=(num_samples,), replace=False)
                measures[MeasureKey.MI] = mi(upper_data[indices], lower_data[indices])
                measures[MeasureKey.PEARSON] = value_if_significant(
                    stats.pearsonr(upper_data[indices, 0], lower_data[indices, 0]), ALPHA)
                measures[MeasureKey.SPEARMAN] = value_if_significant(
                    stats.spearmanr(upper_data[indices, 0], lower_data[indices, 0]), ALPHA)
                measures[MeasureKey.KENDALL] = value_if_significant(
                    stats.kendalltau(upper_data[indices], lower_data[indices]), ALPHA)
            all_measures.append({'dimension': i, 'delay': j, **measures})
            # print(f'Dim: {i}, Dist: {j} -> MI: {measures[MeasureKey.MI]}')
    all_measures = pd.DataFrame(all_measures)
    return all_measures


def main():
    for variable_name in ['tk', 'rh', 'qv', 'z', 'dbz', 'qhydro', 'u', 'v', 'w']:
        for norm in ['global', 'level', 'local']:
            all_data = []
            for member in range(1, 129):
                print(f'[INFO] {variable_name}, {norm} norm, member {member}')
                data = load_data(variable_name, norm, member)
                measures = compute_measures(data)
                measures['member'] = np.full((len(measures),), member)
                measures['norm'] = np.full((len(measures),), norm)
                measures['variable'] = np.full((len(measures),), variable_name)
                all_data.append(measures)
            all_data = pd.concat(all_data, axis=0)
            all_data.to_csv(f'measures_{variable_name}_{norm}.csv')


if __name__ == '__main__':
    main()
