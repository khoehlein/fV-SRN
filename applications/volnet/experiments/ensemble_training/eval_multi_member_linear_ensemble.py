import os

from volnet.experiments.multi_run_experiment import MultiRunExperiment
from volnet.experiments.ensemble_training.directories import (
    INTERPRETER_PATH, SCRIPT_PATH, WORKING_DIRECTORY,
    get_output_directory, get_data_base_path
)

EXPERIMENT_NAME = 'multi_member_linear_ensemble_evaluation'
DATA_FILENAME_PATTERN = os.path.join(get_data_base_path(), 'tk/member{member:04d}/t04.cvol')
SETTINGS_FILE = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/config-files/meteo-ensemble-normalized-anomalies.json'

PARAMETERS = {
    '--renderer:settings-file': SETTINGS_FILE,
    '--world-density-data:num-samples-per-volume': '256**3',
    '--world-density-data:batch-size': '64*64*128',
    '--world-density-data:validation-share': 0.2,
    '--lossmode': 'density',
    '--network:core:layer-sizes': [
        '64:64:64', '128:128:128', '128:128',
    ],
    '--network:core:activation': 'SnakeAlt:2',
    '--network:input:fourier:positions:num-features': 14,
    '--network:input:fourier:method': 'nerf',
    '--network:latent-features:ensemble:num-channels': [
        8, 16, 32
    ],
    '--network:latent-features:volume:num-channels': [
        8, 16, 32
    ],
    '--network:latent-features:volume:grid-size': [
        '4:44:32', '2:44:32', '4:88:64', '2:22:16'
    ],
    '--network:output:parameterization-method': 'direct',
    '-l1': 1.,
    '-lr': 0.01,
    '--lr_step': 50,
    '--epochs': 200,
    '--output:save-frequency': 20,
    '--data-storage:filename-pattern': DATA_FILENAME_PATTERN,
    '--data-storage:ensemble:index-range': [
        '1:3', '1:5', '1:9'
    ],
    '--world-density-data:sub-batching': 8,
    '--dataset-resampling:method': 'importance:grid',
    '--dataset-resampling:loss': 'l1',
    '--dataset-resampling:frequency': 20,
    '--importance-sampler:grid:sub-sampling': 4,
}


if __name__ == '__main__':

    output_directory, log_directory = get_output_directory(
        EXPERIMENT_NAME,
        return_output_dir=False, return_log_dir=True, overwrite=True
    )
    experiment = MultiRunExperiment(
        INTERPRETER_PATH, SCRIPT_PATH, WORKING_DIRECTORY, log_directory
    )

    print('[INFO] Processing vector-valued features...')
    parameters_grid_features = {
        **PARAMETERS,
        **{'--output:base-dir': output_directory}
    }
    experiment.process_parameters(parameters_grid_features)

    print('[INFO] Finished')
