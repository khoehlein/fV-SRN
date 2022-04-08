import os

from volnet.experiments.multi_run_experiment import MultiRunExperiment
from volnet.experiments.rescaled_ensemble import directories as io

parser = io.build_parser()
args = vars(parser.parse_args())
io.set_debug_mode(args)

EXPERIMENT_NAME = 'rescaled_ensemble/overfitting/multi_member_multi_core'
DATA_FILENAME_PATTERN = 'tk/member{member:04d}/t04.cvol'
SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'
SCRIPT_PATH = 'volnet/experiments/overfitting/run_training.py'


PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '12*352*250',
    '--world-density-data:batch-size': '12*352*250',
    '--world-density-data:sub-batching': 12,
    '--world-density-data:validation-share': 0.2,
    '--lossmode': 'density',
    '--network:core:layer-sizes': [
        '32:32:32:32', '64:64:64',
    ],
    '--network:core:activation': ['SnakeAlt:2', 'LeakyReLU'],
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:ensemble:mode': 'multi-grid',
    '--network:latent-features:ensemble:num-channels': 8,
    '--network:latent-features:ensemble:multi-grid:resolution': [
        '6:176:125', '3:88:64', '3:44:32'
    ],
    '--network:latent-features:ensemble:multi-grid:num-grids': [
        1, 2, 4, 8, 16
    ],
    '--network:output:parameterization-method': 'direct',
    '-l1': 1.,
    '--optimizer:lr': 0.001,
    '--optimizer:hyper-params': '{"weight_decay": 0.00001}',
    '--optimizer:scheduler:mode': 'plateau',
    '--optimizer:scheduler:plateau:patience': 8,
    '--optimizer:scheduler:gamma': 0.5,
    '--optimizer:gradient-clipping:max-norm': 10.,
    '--epochs': 400,
    '--output:save-frequency': 20,
    '--data-storage:filename-pattern': os.path.join(io.get_data_base_path(), DATA_FILENAME_PATTERN),
    '--data-storage:ensemble:index-range': '1:65',
}

FLAGS = ['--network:core:split-members']


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
    experiment.process_parameters(parameters_grid_features, flags=FLAGS)

    print('[INFO] Finished')
