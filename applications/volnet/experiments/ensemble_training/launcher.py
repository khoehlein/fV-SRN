import os
import subprocess
import time
from itertools import chain
from typing import Dict, Any

if __name__ == '__main__':
    interpreter = '/home/hoehlein/anaconda3/envs/fvsrn2/bin/python'
    project_base = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications'
    pyrenderer_directory = '../bin'
    script_path = os.path.join(project_base, 'volnet/experiments/refactored_network/run_training.py')
    output_directory = '/home/hoehlein/PycharmProjects/results/fvsrn/parameter_study'

    options = {
        '--renderer:settings-file': '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/config-files/meteo-ensemble-t_0-6-m_1-10.json',
        '--world-density-data:num-samples-per-volume': '256**3',
        '--world-density-data:batch-size': '64*64*128',
        '--world-density-data:validation-share': 0.2,
        '--lossmode': 'density',
        '--network:core:layer-sizes': '32:32:32',
        '--network:core:activation': 'SnakeAlt:2',
        '--network:input:fourier:positions:num-features': 14,
        '--network:input:fourier:method': 'nerf',
        '--network:latent-features:ensemble:grid-size': '2:44:32',
        '--network:latent-features:ensemble:num-channels': 16,
        '--network:output:parameterization-method': 'direct',
        '-l1': 1., '-lr': 0.01, '--lr_step': 50, '--epochs': 200, '--output:save-frequency': 20,
        '--data-storage:filename-pattern': '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/converted_normal/tk/member{member:04d}/t04.cvol',
        '--data-storage:ensemble:index-range': '1:3:1',
        '--dataset-resampling:method': 'importance:grid',
        '--dataset-resampling:loss': 'l1',
        '--dataset-resampling:frequency': 20,
        '--importance-sampler:grid:sub-sampling': 8,
        '--world-density-data:sub-batching': 8,
        '--output:base-dir': output_directory,
    }

    flags = []

    execution_command = [interpreter, script_path] + [f'{arg}' for arg in chain.from_iterable(options.items())] + flags

    log_file = './log_file.log'
    if log_file is not None:
        log_file = open(log_file, 'w')
        stdout = log_file
        stderr = log_file
    else:
        stdout = subprocess.PIPE
        stderr = subprocess.PIPE
    time_start = time.time()
    process = subprocess.Popen(execution_command, cwd=project_base, stdout=stdout, stderr=stderr)
    time_end = time.time()
    outputs, errors = process.communicate()
    if log_file is not None:
        log_file.close()
    duration = (time_end - time_start)
    print('Output', outputs)
    print('Error', errors)
    print(f'[INFO] Duration: {duration}')
