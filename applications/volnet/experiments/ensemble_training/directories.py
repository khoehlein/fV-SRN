import os

PROJECT_BASE ='/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications'
WORKING_DIRECTORY = PROJECT_BASE
INTERPRETER_PATH = '/home/hoehlein/anaconda3/envs/fvsrn2/bin/python'
SCRIPT = 'volnet/experiments/refactored_network/run_training.py'
SCRIPT_PATH = os.path.join(PROJECT_BASE, SCRIPT)
OUTPUT_BASE_DIR = '/home/hoehlein/PycharmProjects/results/fvsrn'
OUTPUT_DIR_NAME = 'runs'
LOG_DIR_NAME = 'log'

DATA_BASE_PATH = {
    'tuini15-cg05-cg-in-tum-de': '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/converted_normal_anomaly',
    'gpusrv01-cg-in-tum-de': '/home/hoehlein/data/1000_member_ensemble/normalized_anomalies/single_member'
}


def get_data_base_path():
    import socket
    host_name = socket.gethostname()
    return DATA_BASE_PATH[host_name]


def get_output_directory(experiment_name, overwrite=False, return_output_dir=False, return_log_dir=False):
    experiment_dir = os.path.join(OUTPUT_BASE_DIR, experiment_name)
    if not os.path.isdir(experiment_dir):
        os.makedirs(experiment_dir)
    else:
        if not overwrite:
            raise RuntimeError(f'[ERROR] Experiment directory {experiment_dir} exists already.')
    out = [experiment_dir]
    if return_output_dir:
        output_dir = os.path.join(experiment_dir, OUTPUT_DIR_NAME)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        out.append(output_dir)
    if return_log_dir:
        log_dir = os.path.join(experiment_dir, LOG_DIR_NAME)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
        out.append(log_dir)
    return tuple(out)
