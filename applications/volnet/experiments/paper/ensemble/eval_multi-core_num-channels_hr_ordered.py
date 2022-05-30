import os
from itertools import product

import numpy as np

from volnet.experiments.multi_run_experiment import MultiRunExperiment
from volnet.experiments.rescaled_ensemble import directories as io

parser = io.build_parser()
args = vars(parser.parse_args())
io.set_debug_mode(args)

EXPERIMENT_NAME = 'paper/ensemble/multi_core/num_channels/hr_ordered'
DATA_FILENAME_PATTERN = '../level-min-max_scaling/tk/member{member:04d}/t04.cvol'
SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'
SCRIPT_PATH = 'volnet/experiments/ordered_data/run_training.py'


PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '16*12*352*250',
    '--world-density-data:batch-size': '6*352*250',
    '--world-density-data:validation-share': 0.25,
    '--world-density-data:sub-batching': 1,
    '--lossmode': 'density',
    '--network:core:layer-sizes': '32:32:32',
    '--network:core:activation': 'SnakeAlt:1',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:volume:mode': 'grid',
    '--network:latent-features:volume:num-channels': [48],
    '--network:latent-features:volume:grid:resolution': '6:44:31',
    '--network:output:parameterization-method': 'mixed',
    '-l1': 1.,
    '--optimizer:lr': 0.01,
    '--optimizer:hyper-params': '{}',
    '--optimizer:scheduler:mode': 'step-lr',
    '--optimizer:scheduler:gamma': 0.2,
    '--optimizer:scheduler:step-lr:step-size': 20,
    '--optimizer:gradient-clipping:max-norm': 1000.,
    '--epochs': 50,
    '--output:save-frequency': 10,
    '--data-storage:filename-pattern': os.path.join(io.get_data_base_path(), DATA_FILENAME_PATTERN),
    '--dataset-resampling:method': 'random',
    '--dataset-resampling:frequency': 10,
    '--data-storage:ensemble:index-range': '1:65',
}
FLAGS = ['--network:core:split-members']

if __name__ == '__main__':

    project_base_path = io.get_project_base_path()

    output_directory, log_directory = io.get_output_directory(
        EXPERIMENT_NAME,
        return_output_dir=False, return_log_dir=True, overwrite=True
    )
    experiment = MultiRunExperiment(
        io.INTERPRETER_PATH, os.path.join(project_base_path, SCRIPT_PATH), io.get_project_base_path(), log_directory
    )

    print('[INFO] Processing grid-valued features...')
    parameters_grid_features = {
        **PARAMETERS,
        **{'--output:base-dir': output_directory},
    }
    experiment.process_parameters(parameters_grid_features, flags=FLAGS)

    print('[INFO] Finished')
