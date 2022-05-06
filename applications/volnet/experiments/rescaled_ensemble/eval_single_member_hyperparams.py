import os

from volnet.experiments.multi_run_experiment import MultiRunExperiment
from volnet.experiments.rescaled_ensemble import directories as io

parser = io.build_parser()
args = vars(parser.parse_args())
io.set_debug_mode(args)

EXPERIMENT_NAME = 'rescaled_ensemble/single_member_hyperparams'
DATA_FILENAME_PATTERN = 'tk/member0001/t04.cvol'
SETTINGS_FILE = 'config-files/meteo-ensemble_tk_local-min-max.json'

PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '12*352*250',
    '--world-density-data:batch-size': '12*352*250',
    '--world-density-data:validation-share': 0.2,
    '--world-density-data:sub-batching': 12,
    '--lossmode': 'density',
    '--network:core:layer-sizes': '32:32:32:32',
    '--network:core:activation': 'SnakeAlt:2',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:volume:mode': 'grid',
    '--network:latent-features:volume:num-channels': 4,
    '--network:latent-features:volume:grid:resolution': '4:44:32',
    '--network:output:parameterization-method': 'direct',
    '-l1': 1.,
    '--optimizer:lr': 0.01,
    '--optimizer:hyper-params': '{"weight_decay": 0.00001}',
    '--optimizer:scheduler:mode': 'step-lr',
    '--optimizer:scheduler:step-lr:step-size': [50, 80, 100],
    '--optimizer:scheduler:gamma': [0.1, 0.2, 0.5],
    '--optimizer:gradient-clipping:max-norm': 10.,
    '--epochs': 400,
    '--output:save-frequency': 50,
    '--data-storage:filename-pattern': os.path.join(io.get_data_base_path(), DATA_FILENAME_PATTERN),
    '--dataset-resampling:method': 'random',
    '--dataset-resampling:frequency': [10, 20, 50],
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
