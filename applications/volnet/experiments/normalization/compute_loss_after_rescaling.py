import os

import numpy as np
import pandas as pd
import torch

from data.necker_ensemble.single_variable import load_scales, update_file_pattern_base_path
from volnet.analysis.deviation_statistics import DeviationStatistics
from volnet.modules.datasets import VolumeDataStorage
from volnet.modules.datasets.world_dataset import WorldSpaceDensityEvaluator
from volnet.modules.render_tool import RenderTool


results_root_path = '/home/hoehlein/PycharmProjects/results/fvsrn'
variable_name = 'tk'
device = torch.device('cuda:0')


for variable_name in ['tk', 'rh']:
    data = []
    for normalization in ['global', 'level', 'local']:

        experiment_directory = os.path.join(results_root_path, 'normalization', 'single_member', variable_name, normalization)
        checkpoint_directory = os.path.join(experiment_directory, 'results', 'model')
        run_names = list(sorted(f for f in os.listdir(checkpoint_directory) if f.startswith('run')))
        scales = load_scales(f'{normalization}-min-max', variable_name)

        for run_name in run_names:
            print(f'[INFO] Processing variable {variable_name}, normalization {normalization}, {run_name}')
            checkpoint = torch.load(
                os.path.join(checkpoint_directory, run_name, 'model_epoch_200.pth'),
                map_location='cpu'
            )
            args = checkpoint['parameters']
            network = checkpoint['model']
            network_evaluator = WorldSpaceDensityEvaluator(network, 0, 0, 0)

            print('[INFO] Initializing rendering tool.')
            render_tool = RenderTool.from_dict(args, device)
            volume_evaluator = render_tool.get_volume_evaluator()
            if not volume_evaluator.interpolator.grid_resolution_new_behavior:
                volume_evaluator.interpolator.grid_resolution_new_behavior = True
            image_evaluator = render_tool.get_image_evaluator()

            print('[INFO] Initializing volume data storage.')
            args['verify_files_exist'] = False
            volume_data_storage = VolumeDataStorage.from_dict(args)
            volume_data_storage.file_pattern = update_file_pattern_base_path(volume_data_storage.file_pattern)
            volume = volume_data_storage.load_volume(0, 0, index_access=True)
            ground_truth = volume.get_feature(0).get_level(0).to_tensor().data.cpu().numpy()
            resolution = ground_truth.shape[1:]

            positions = np.meshgrid(*[(np.arange(r) + 0.5) / r for r in resolution], indexing='ij')
            positions = np.stack([p.astype(np.float32).ravel() for p in positions], axis=-1)
            positions = torch.from_numpy(positions).to(volume_evaluator.device)

            with torch.no_grad():
                predictions = torch.clip(network_evaluator.evaluate(positions).to(device), min=0., max=1.)
                predictions = predictions.view(ground_truth.shape).data.cpu().numpy()
                stats = DeviationStatistics(ground_truth, predictions, scales=scales)
                data.append({
                    'run_name': run_name,
                    'normalization': normalization,
                    'member': volume_data_storage.file_pattern,
                    **args,
                    **stats.to_dict(),
                    'num_parameters': sum([p.numel() for p in network.parameters()])
                })

    data = pd.DataFrame(data)
    data.to_csv(os.path.join(results_root_path, 'normalization', 'single_member', variable_name, 'accuracies.csv'))
