import argparse
import os
import socket
from enum import Enum


class DebugMode(Enum):
    DEBUG = 'debug'
    PRODUCTION = 'production'


DEBUG_MODE: DebugMode = None

PROJECT_BASE_PATH ={
    DebugMode.DEBUG: '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications',
    DebugMode.PRODUCTION: '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications',
}
DATA_BASE_PATH = {
    'tuini15-cg05-cg-in-tum-de': '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/converted_normal_anomaly',
    'gpusrv01-cg-in-tum-de': '/home/hoehlein/data/1000_member_ensemble/normalized_anomalies/single_member'
}
INTERPRETER_PATH = '/home/hoehlein/anaconda3/envs/fvsrn2/bin/python'
SCRIPT_PATH = 'volnet/experiments/refactored_network/run_training.py'
OUTPUT_BASE_DIR = '/home/hoehlein/PycharmProjects/results/fvsrn'
OUTPUT_DIR_NAME = 'runs'
LOG_DIR_NAME = 'log'


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, choices=[DebugMode.DEBUG.value, DebugMode.PRODUCTION.value], default=DebugMode.DEBUG.value)
    return parser


def set_debug_mode(args):
    global DEBUG_MODE
    DEBUG_MODE = DebugMode.DEBUG if args['mode'] else DebugMode.PRODUCTION
    print(f'[INFO] Running scripts with path configurations for {DEBUG_MODE.value} mode.')
    return DEBUG_MODE


def get_project_base_path():
    assert DEBUG_MODE is not None, 'Debug mode must be set before path can be obtained!'
    return PROJECT_BASE_PATH[DEBUG_MODE]


def get_data_base_path():
    host_name = socket.gethostname()
    return DATA_BASE_PATH[host_name]


def get_script_path():
    return os.path.join(get_project_base_path(), SCRIPT_PATH)


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
