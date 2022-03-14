import os

from volnet.experiments.ensemble_training.directories import INTERPRETER_PATH
from volnet.experiments.multi_run_experiment import MultiRunExperiment

cwd = os.getcwd()

experiment = MultiRunExperiment(
    INTERPRETER_PATH,
    '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/volnet/experiments/find_device.py',
    cwd,
    os.path.join(cwd, 'log'),
    create_log_dir=True
)

parameters = {'--i': ['1', '2', '3', '4']}
experiment.process_parameters(kwargs=parameters)
