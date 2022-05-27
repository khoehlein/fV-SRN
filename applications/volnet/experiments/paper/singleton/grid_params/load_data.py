import os

import numpy as np
import pandas as pd
import torch

from volnet.modules.datasets import VolumeDataStorage
from volnet.modules.datasets.sampling.meteo_ensemble_data import SingleVariableData
from volnet.modules.datasets.world_dataset import WorldSpaceDensityEvaluator
from volnet.modules.render_tool import RenderTool

ROOT_DIRECTORY = '/home/hoehlein/PycharmProjects/results/fvsrn'
EXPERIMENT_NAME = 'paper/single_member/grid_params'
PARAMETER_NAME = 'qv/vres'
CHECKPOINT_NAME = 'model_epoch_250.pth'

directory = os.path.join(ROOT_DIRECTORY, EXPERIMENT_NAME, PARAMETER_NAME)
checkpoint_directory = os.path.join(directory, 'results', 'model')
run_names = list(sorted(f for f in os.listdir(checkpoint_directory) if f.startswith('run')))
device = torch.device('cuda:0')

data = []


def load_scale_data(args):
    file_pattern = args['data_storage:filename_pattern']
    normalization, variable_name, member_key = file_pattern.split('/')[-4:-1]
    return SingleVariableData(variable_name,  normalization.split('_')[0], device=device).load_scales(return_volume=True), normalization, member_key


for run_name in run_names:
    checkpoint = torch.load(
        os.path.join(checkpoint_directory, run_name, CHECKPOINT_NAME),
        map_location='cpu'
    )

    args = checkpoint['parameters']
    network = checkpoint['model']
    network_evaluator = WorldSpaceDensityEvaluator(network, 0, 0, 0)
    scale_data, normalization, member_key = load_scale_data(args)

    print('[INFO] Initializing rendering tool.')
    render_tool = RenderTool.from_dict(args, device)
    volume_evaluator = render_tool.get_volume_evaluator()
    if not volume_evaluator.interpolator.grid_resolution_new_behavior:
        volume_evaluator.interpolator.grid_resolution_new_behavior = True
    image_evaluator = render_tool.get_image_evaluator()

    print('[INFO] Initializing volume data storage.')
    volume_data_storage = VolumeDataStorage.from_dict(args)
    volume = volume_data_storage.load_volume(0, 0, index_access=True)
    resolution = volume.get_feature(0).get_level(0).to_tensor().shape[1:]
    positions = np.meshgrid(*[(np.arange(r) + 0.5) / r for r in resolution], indexing='ij')
    positions = np.stack([p.astype(np.float32).ravel() for p in positions], axis=-1)
    positions = torch.from_numpy(positions).to(volume_evaluator.device)

    with torch.no_grad():
        volume_evaluator.set_source(volume_data=volume, mipmap_level=0, feature=None)
        ground_truth = volume_evaluator.evaluate(positions)

        volume_evaluator.set_source(volume_data=scale_data, mipmap_level=0, feature='offset')
        offset = volume_evaluator.evaluate(positions)

        volume_evaluator.set_source(volume_data=scale_data, mipmap_level=0, feature='scale')
        scale = volume_evaluator.evaluate(positions)

        predictions = network_evaluator.evaluate(positions).to(device)

        deviation = torch.abs(predictions - ground_truth)

        data.append({
            'run_name': run_name,
            'normalization': normalization,
            'member': member_key,
            'l1': torch.mean(deviation).item(),
            'l2': torch.mean(deviation ** 2.).item(),
            'l1r': torch.mean(deviation * torch.abs(scale)).item(),
            'l2r': torch.mean((deviation * torch.abs(scale)) ** 2.).item(),
            'num_parameters': sum([p.numel() for p in network.parameters()])
        })

data = pd.DataFrame(data)
data.to_csv(os.path.join(directory, 'accuracies.csv'))

