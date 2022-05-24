import os
from itertools import product

import numpy as np

from volnet.experiments.multi_run_experiment import MultiRunExperiment
from volnet.experiments.rescaled_ensemble import directories as io

parser = io.build_parser()
args = vars(parser.parse_args())
io.set_debug_mode(args)

EXPERIMENT_NAME = 'paper/single_member/grid_params/rh/hres'

DATA_FILENAME_PATTERN = []
for normalization in ['level', 'local', 'global']:
    DATA_FILENAME_PATTERN += ['../{}-min-max_scaling/rh/member{:04d}/t04.cvol'.format(normalization, l) for l in range(1, 9)]

SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'

subsampling = 2 ** np.arange(1, 7)
grid_sizes = [
    '6:{}:{}'.format(int(np.round(352 / s)), int(np.round(250 / s)))
    for s in subsampling
]
print(grid_sizes)


PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '16*12*352*250',
    '--world-density-data:batch-size': '6*352*250',
    '--world-density-data:validation-share': 0.2,
    '--world-density-data:sub-batching': 8,
    '--lossmode': 'density',
    '--network:core:layer-sizes': ['32:32:32'],
    '--network:core:activation': 'SnakeAlt:1',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:volume:mode': 'grid',
    '--network:latent-features:volume:num-channels': 8,
    '--network:latent-features:volume:grid:resolution': grid_sizes,
    '--network:output:parameterization-method': 'mixed',
    '-l1': 1.,
    '--optimizer:lr': 0.01,
    '--optimizer:hyper-params': '{}',
    '--optimizer:scheduler:mode': 'step-lr',
    '--optimizer:scheduler:gamma': 0.2,
    '--optimizer:scheduler:step-lr:step-size': 100,
    '--optimizer:gradient-clipping:max-norm': 1000.,
    '--epochs': 250,
    '--output:save-frequency': 40,
    '--data-storage:filename-pattern': [os.path.join(io.get_data_base_path(), dfp) for dfp in DATA_FILENAME_PATTERN],
    '--dataset-resampling:method': 'random',
    '--dataset-resampling:frequency': 50,
}


if __name__ == '__main__':

    output_directory, log_directory = io.get_output_directory(
        EXPERIMENT_NAME,
        return_output_dir=False, return_log_dir=True, overwrite=True
    )
    experiment = MultiRunExperiment(
        io.INTERPRETER_PATH, io.get_script_path(), io.get_project_base_path(), log_directory
    )

    print('[INFO] Processing grid-valued features...')
    parameters_grid_features = {
        **PARAMETERS,
        **{'--output:base-dir': output_directory},
    }
    experiment.process_parameters(parameters_grid_features)

    print('[INFO] Finished')
