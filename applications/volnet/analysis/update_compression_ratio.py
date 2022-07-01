import os
from functools import lru_cache

import pandas as pd
import torch

from volnet.analysis.deviation_statistics import CompressionStatistics


@lru_cache(maxsize=10)
def load_checkpoint(checkpoint_file):
    return torch.load(checkpoint_file, map_location='cpu')['model']


def update_run_statistics(path):
    data = pd.read_csv(path)
    checkpoints = data['checkpoint']
    compression = [
        CompressionStatistics(load_checkpoint(checkpoint_file)).compression_rate()
        for checkpoint_file in checkpoints
    ]
    data['compression_ratio'] = compression
    data.to_csv(path)
    print('[INFO] Finished')


def main():
    base_path = '/home/hoehlein/PycharmProjects/results/fvsrn/multi-variate/single-member/uvw'
    sub_folders = os.listdir(base_path)

    for folder in sub_folders:
        experiment_base_dir = os.path.join(base_path, folder)
        contents = os.listdir(experiment_base_dir)
        if 'stats' in contents:
            print(f'[INFO] Updating run statistics in {experiment_base_dir}')
            update_run_statistics(os.path.join(experiment_base_dir, 'stats', 'run_statistics.csv'))
        else:
            print(f'[INFO] stats not found in {experiment_base_dir}')


def main2():
    experiment_base_dir = '/home/hoehlein/PycharmProjects/results/fvsrn/paper/single_member/grid_params/parameter_interplay'
    update_run_statistics(os.path.join(experiment_base_dir, 'stats', 'run_statistics.csv'))


if __name__ == '__main__':
    main()
