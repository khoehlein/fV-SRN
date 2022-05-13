import os
from itertools import product

from volnet.experiments.multi_run_experiment import MultiRunExperiment
from volnet.experiments.rescaled_ensemble import directories as io

parser = io.build_parser()
args = vars(parser.parse_args())
io.set_debug_mode(args)

EXPERIMENT_NAME = 'rescaled_ensemble/multi_member_multi_core/32c/8m'
DATA_FILENAME_PATTERN = 'tk/member{member:04d}/t04.cvol'
SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'


grid_sizes = [
    '{}:{}:{}'.format(h, int(w), int(d))
    for h, (w, d) in product([4, 8, 12], zip([176, 88, 44, 22, 11, 6], [125, 63, 31, 16, 8, 4]))
]


PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '16*12*352*250',
    '--world-density-data:batch-size': '6*352*250',
    '--world-density-data:validation-share': 0.2,
    '--world-density-data:sub-batching': 24,
    '--lossmode': 'density',
    '--network:core:layer-sizes': '32:32:32',
    '--network:core:activation': 'SnakeAlt:1',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:volume:mode': 'grid',
    '--network:latent-features:volume:num-channels': [4, 8],
    '--network:latent-features:volume:grid:resolution': grid_sizes,
    '--network:output:parameterization-method': 'mixed',
    '-l1': 1.,
    '--optimizer:lr': 0.01,
    '--optimizer:hyper-params': '{}',
    '--optimizer:scheduler:mode': 'step-lr',
    '--optimizer:scheduler:gamma': 0.5,
    '--optimizer:scheduler:step-lr:step-size': 100,
    '--optimizer:gradient-clipping:max-norm': 1000.,
    '--epochs': 200,
    '--output:save-frequency': 40,
    '--data-storage:filename-pattern': os.path.join(io.get_data_base_path(), DATA_FILENAME_PATTERN),
    '--data-storage:ensemble:index-range': '1:9',
    '--dataset-resampling:method': 'random',
    '--dataset-resampling:frequency': 50,
}
FLAGS = ['--network:core:split-members']


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
    experiment.process_parameters(parameters_grid_features, flags=FLAGS)

    print('[INFO] Finished')
