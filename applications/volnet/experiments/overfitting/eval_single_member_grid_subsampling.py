import os
from itertools import product
from typing import List

import numpy as np

from volnet.experiments.multi_run_experiment import MultiRunExperiment
from volnet.experiments.rescaled_ensemble import directories as io

parser = io.build_parser()
args = vars(parser.parse_args())
io.set_debug_mode(args)

EXPERIMENT_NAME = 'rescaled_ensemble/overfitting/grid_sizes'
DATA_FILENAME_PATTERN = 'tk/member0001/t04.cvol'
SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'
SCRIPT_PATH = 'volnet/experiments/overfitting/run_training.py'


def get_grid_sizes(subsampling_hor: List[float], subsampling_vert: List[float]):
    base_resolution = (12, 352, 250)
    min_resolution = 2

    def get_resampled_size(base_size, subsampling):
        out_size = int(np.round(base_size / subsampling))
        return min(max(min_resolution, out_size), base_size)

    def get_resolution(sh, sv):
        return ':'.join([str(get_resampled_size(b, s)) for b, s in zip(base_resolution, [sv, sh, sh])])

    grid_sizes = [get_resolution(sh, sv) for sh, sv in product(subsampling_hor, subsampling_vert)]
    return grid_sizes


PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '256**3',
    '--world-density-data:batch-size': '64*64*128',
    '--world-density-data:validation-share': 0.2,
    '--lossmode': 'density',
    '--network:core:layer-sizes': ['32:32:32:32', '64:64:64'],
    '--network:core:activation': 'SnakeAlt:2',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:volume:mode': 'grid',
    '--network:latent-features:volume:num-channels': [4, 8, 16],
    '--network:latent-features:volume:grid:resolution': get_grid_sizes([1, 2, 4, 8, 16], [1, 2, 4, 8]),
    '--network:output:parameterization-method': 'direct',
    '-l1': 1.,
    '-lr': 0.01,
    '--lr_step': 50,
    '--epochs': 200,
    '--output:save-frequency': 20,
    '--data-storage:filename-pattern': os.path.join(io.get_data_base_path(), DATA_FILENAME_PATTERN),
    '--world-density-data:sub-batching': 8,
}


if __name__ == '__main__':

    output_directory, log_directory = io.get_output_directory(
        EXPERIMENT_NAME,
        return_output_dir=False, return_log_dir=True, overwrite=True
    )
    project_base_path = io.get_project_base_path()
    experiment = MultiRunExperiment(
        io.INTERPRETER_PATH, os.path.join(project_base_path, SCRIPT_PATH), project_base_path, log_directory
    )

    print('[INFO] Processing grid-valued features...')
    parameters_grid_features = {
        **PARAMETERS,
        **{'--output:base-dir': output_directory},
    }
    experiment.process_parameters(parameters_grid_features)

    print('[INFO] Finished')
