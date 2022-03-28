import os
from volnet.experiments.multi_run_experiment import MultiRunExperiment
from volnet.experiments.ensemble_training import directories as io

parser = io.build_parser()
args = vars(parser.parse_args())
io.set_debug_mode(args)

DATA_FILENAME_PATTERN = 'tk/member{member:04d}/t04.cvol'
SETTINGS_FILE = 'config-files/meteo-ensemble-normalized-anomalies.json'
EXPERIMENT_NAME = 'univariate_linear_encoding_wide'

PARAMETERS = {
    '--renderer:settings-file': os.path.join(io.get_project_base_path(), SETTINGS_FILE),
    '--world-density-data:num-samples-per-volume': '64**3',
    '--world-density-data:batch-size': '64*64*128',
    '--world-density-data:validation-share': 0.2,
    '--lossmode': 'density',
    '--network:core:layer-sizes': '64:64:64',
    '--network:core:activation': 'SnakeAlt:2',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:ensemble:mode': 'vector',
    '--network:latent-features:ensemble:num-channels': [
        4, 16, 64
    ],
    '--network:latent-features:volume:mode': 'grid',
    '--network:latent-features:volume:num-channels': [
        8, 16, 32
    ],
    '--network:latent-features:volume:grid:resolution': '4:88:64',
    '--network:output:parameterization-method': 'direct',
    '-l1': 1.,
    '-lr': 0.01,
    '--lr_step': 50,
    '--epochs': 400,
    '--output:save-frequency': 40,
    '--data-storage:filename-pattern': os.path.join(io.get_data_base_path(), DATA_FILENAME_PATTERN),
    '--data-storage:ensemble:index-range': '1:129',
    '--world-density-data:sub-batching': 8,
    '--dataset-resampling:method': 'random',
    '--dataset-resampling:frequency': 20,
}


if __name__ == '__main__':

    output_directory, log_directory = io.get_output_directory(
        EXPERIMENT_NAME,
        return_output_dir=False, return_log_dir=True, overwrite=True
    )
    experiment = MultiRunExperiment(
        io.INTERPRETER_PATH, io.get_script_path(), io.get_project_base_path(), log_directory
    )

    print('[INFO] Processing vector-valued features...')
    parameters_grid_features = {
        **PARAMETERS,
        **{'--output:base-dir': output_directory}
    }
    experiment.process_parameters(parameters_grid_features)

    print('[INFO] Finished')
