import argparse
from itertools import product

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset, DataLoader

from common.mathparser import BigInteger
from volnet.modules.datasets.dataset_type import DatasetType
from volnet.modules.datasets.evaluation import LossEvaluator
from volnet.modules.datasets.evaluation.network_evaluator import NetworkEvaluator
from volnet.modules.datasets.resampling.interface import IImportanceSampler
from volnet.modules.helpers import parse_slice_string
from volnet.modules.datasets.position_sampler import PositionSampler
from volnet.modules.datasets.resampling.fixed_grid._fixed_grid_legacy import FixedGridImportanceSampler
from volnet.modules.datasets.evaluation.volume_evaluator import VolumeEvaluator
from volnet.training_data import _cat_collate as cat_collate


class WorldSpaceDensityData(Dataset):

    OUTPUT_KEYS = ['positions', 'targets', 'tf', 'timestep', 'ensemble']

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):

        def add_arguments(group_: argparse._ArgumentGroup, prefix: str, require_or_default):

            def add_argument_with_prefix(arg, default=None, **kwargs):
                if require_or_default and default is None:
                    kwargs.update({'required': True})
                else:
                    kwargs.update({'default': default if require_or_default else None})
                group_.add_argument('--world-density-data:' + prefix + arg, **kwargs)

            add_argument_with_prefix(
                'batch-size', type=BigInteger, help="""
                number of samples per batch
                """
            )
            add_argument_with_prefix(
                'ensemble-index-slice', type=str, default=':', help="""
                Slice of the available ensemble index range to use for data generation.
                    
                Examples: ":", "0::2", "1::2", "0:10:2" (Default: ":")
                """
            )
            add_argument_with_prefix(
                'timestep-index-slice', type=str, default=':', help="""
                Slice of the available timestep index range to use for data generation
                
                Examples: ":", "0::2", "1::2", "0:10:2" (Default: ":")
                """
            )
            # TODO: Add support for weight-based reweighting of data samples
            # add_argument_with_prefix(
            #     'reweighting-mode', type=str, default='none', choices=['resampling', 'weighting'],
            #     help="""
            #     weighting mode for loss-based reweighting of samples
            #     """
            # )

        group = parser.add_argument_group('WorldDensityData')
        add_arguments(group, '', True) # flags affect both training and validation
        add_arguments(group, DatasetType.TRAINING.value + ':', False) # flags affect only training and overwrite previous settings
        add_arguments(group, DatasetType.VALIDATION.value + ':', False) # flags affect only validation and overwrite previous settings
        group.add_argument('--world-density-data:num-samples-per-volume', type=BigInteger, default=256**3, help="""
        number of sample positions to draw per timestep and ensemble member
        """)
        group.add_argument('--world-density-data:validation-share', type=float, default=0.2, help="""
        fraction of sample points to use for validation
        """)
        group.add_argument('--world-density-data:validation-mode', type=str, default='sample', help="""
        mode specification for generating validation data: 
        - sample if validation data is to be sampled using a position sampler
        - grid if validation data is to be taken from the original feature grid  
        """, choices=['sample', 'grid'])
        group.add_argument('--world-density-data:sub-batching', type=int, default=None, help="""
        fraction of sample points to use for validation
        """)

    @classmethod
    def from_dict(
            cls,
            args, volume_data_storage, mode=DatasetType.TRAINING,
            volume_evaluator=None, position_sampler=None, dtype=None
    ):
        prefix = 'world_density_data:'

        def get_argument(option):
            argument = args[prefix + option] # default argument
            mode_argument = args[prefix + f'{mode.value}:{option}'] # mode argument
            if mode_argument is not None:
                # overwrite default if mode argument is given
                argument = mode_argument
            return argument

        # get mode-related arguments
        batch_size = get_argument('batch_size')
        ensemble_index_slice = get_argument('ensemble_index_slice')
        timestep_index_slice = get_argument('timestep_index_slice')
        # get number of samples
        total_samples_per_volume = args[prefix + 'num_samples_per_volume']
        validation_share = args[prefix + 'validation_share']
        assert 0 <= validation_share < 1, '[ERROR] Validation share must satisfy 0 <= x < 1'
        num_samples_per_volume = int(total_samples_per_volume * validation_share)
        if mode == DatasetType.TRAINING:
            num_samples_per_volume = total_samples_per_volume - num_samples_per_volume
        if mode == DatasetType.VALIDATION and args[prefix + 'validation_mode'] == 'grid':
            data = cls(
                volume_data_storage, num_samples_per_volume, batch_size,
                sub_batching=args[prefix + 'sub_batching'],
                volume_evaluator=None, position_sampler=None,
                ensemble_index_slice=ensemble_index_slice, timestep_index_slice=timestep_index_slice,
                dtype=dtype
            )
            return data.sample_original_grid(volume_evaluator)
        return cls(
            volume_data_storage, num_samples_per_volume, batch_size,
            sub_batching=args[prefix + 'sub_batching'],
            volume_evaluator=volume_evaluator, position_sampler=position_sampler,
            ensemble_index_slice=ensemble_index_slice, timestep_index_slice=timestep_index_slice,
            dtype=dtype
        )

    def __init__(
            self,
            volume_data_storage,
            num_samples_per_volume, batch_size, sub_batching=None,
            volume_evaluator=None, position_sampler=None,
            # reweighting_mode='none',
            ensemble_index_slice=None, timestep_index_slice=None,
            dtype=None
    ):
        super(WorldSpaceDensityData, self).__init__()
        self.volume_data_storage = volume_data_storage
        self.num_samples_per_volume = num_samples_per_volume
        if ensemble_index_slice is None:
            ensemble_index_slice = ':'
        self.ensemble_index = volume_data_storage.ensemble_index[slice(*parse_slice_string(ensemble_index_slice))]
        if timestep_index_slice is None:
            timestep_index_slice = ':'
        self.timestep_index = volume_data_storage.timestep_index[slice(*parse_slice_string(timestep_index_slice))]
        self.batch_size = batch_size
        self.sub_batching = sub_batching
        if sub_batching is not None:
            assert batch_size % sub_batching == 0, \
                f'[ERROR] Number of sub-batches ({sub_batching}) does not divide batch size ({batch_size})'
        if dtype is None:
            dtype = torch.float32
        self.dtype = dtype
        self._reset_data()
        if volume_evaluator is not None and position_sampler is not None:
            self.sample_data(volume_evaluator, position_sampler)

    def _reset_data(self):
        self.data = {key: [] for key in WorldSpaceDensityData.OUTPUT_KEYS}

    def _add_samples_to_data(self, positions, targets, tf_index, timestep_index, ensemble_index):
        for key, data in zip(WorldSpaceDensityData.OUTPUT_KEYS,
                             [positions, targets, tf_index, timestep_index, ensemble_index]):
            self.data[key].append(data)

    def _finalize_data(self):
        self.data = {key: np.concatenate(self.data[key], axis=0) for key in WorldSpaceDensityData.OUTPUT_KEYS}
        if self.sub_batching is None:
            self.data = [values for values in zip(*[self.data[key] for key in WorldSpaceDensityData.OUTPUT_KEYS])]
        else:
            sub_batch_size = np.ceil(self.batch_size / self.sub_batching)
            num_samples = len(self.data[self.OUTPUT_KEYS[0]])
            num_sections = np.ceil(num_samples / sub_batch_size)
            self.data = [
                values for values in zip(
                    *[np.array_split(self.data[key], int(num_sections)) for key in WorldSpaceDensityData.OUTPUT_KEYS]
                )
            ]

    def sample_data(self, volume_evaluator: VolumeEvaluator, position_sampler: PositionSampler):
        assert volume_evaluator.interpolator.grid_resolution_new_behavior, \
            '[ERROR] Data should only be sampled with new resolution behavior in volume interpolation.'
        position_sampler.reset()
        self._reset_data()
        for (timestep_index, timestep), (ensemble_index, ensemble) in product(enumerate(self.timestep_index), enumerate(self.ensemble_index)):
            volume_data = self.volume_data_storage.load_volume(timestep=timestep, ensemble=ensemble, index_access=False)
            volume_evaluator.set_source(volume_data)
            positions = position_sampler.sample(self.num_samples_per_volume)
            positions = torch.from_numpy(positions).to(dtype=self.dtype, device=volume_evaluator.device)
            targets = volume_evaluator.evaluate(positions)
            print(f"[INFO] Targets: min={targets.min().item()}, max={targets.max().item()}, mean={targets.mean().item()}")
            positions = positions.data.cpu().numpy()
            targets = targets.data.cpu().numpy()
            tf_index_data, timestep_index_data, ensemble_index_data = self._build_index_data(0, ensemble_index, timestep_index, self.num_samples_per_volume)
            self._add_samples_to_data(positions, targets, tf_index_data, timestep_index_data, ensemble_index_data)
        self._finalize_data()
        volume_evaluator.restore_defaults()
        return self

    def sample_original_grid(self, volume_evaluator: VolumeEvaluator, feature=0, new_behavior=True, _clamp_for_old_behavior=None):
        assert volume_evaluator.interpolator.grid_resolution_new_behavior == new_behavior
        self._reset_data()
        for (timestep_index, timestep), (ensemble_index, ensemble) in product(enumerate(self.timestep_index), enumerate(self.ensemble_index)):
            volume_data = self.volume_data_storage.load_volume(timestep=timestep, ensemble=ensemble, index_access=False)
            resolution = tuple(volume_data.get_feature(feature).get_level(0).to_tensor().shape[1:])
            if new_behavior:
                positions = np.meshgrid(*[(np.arange(r) + 0.5) / r for r in resolution], indexing='ij')
            else:
                assert _clamp_for_old_behavior is not None
                def get_normalized_positions(num_nodes):
                    num_segments = 2 * (num_nodes - 1)
                    p = torch.arange(1, num_segments + 2, 2) / num_segments
                    if _clamp_for_old_behavior:
                        p[-1] = 1.
                    return p
                positions = np.meshgrid(*[get_normalized_positions(r) for r in resolution], indexing='ij')
            positions = np.stack([p.astype(np.float32).ravel() for p in positions], axis=-1)
            volume_evaluator.set_source(volume_data)
            targets = volume_evaluator.evaluate(torch.from_numpy(positions).to(volume_evaluator.device))
            targets = targets.data.cpu().numpy()
            print(f"[INFO] Targets: min={targets.min()}, max={targets.max()}, mean={targets.mean()}")
            tf_index_data, timestep_index_data, ensemble_index_data = self._build_index_data(0, ensemble_index, timestep_index, len(targets))
            self._add_samples_to_data(positions, targets, tf_index_data, timestep_index_data, ensemble_index_data)
        self._finalize_data()
        return self

    def _build_index_data(self, tf_index, timestep_idx, ensemble_idx, num_samples):
        tf_index = np.full((num_samples,), tf_index, dtype=np.int32)  # dataset doesnt support tf
        timestep_index = np.full((num_samples,), timestep_idx, dtype=np.float32)
        ensemble_index = np.full((num_samples,), ensemble_idx, dtype=np.float32)
        return tf_index, timestep_index, ensemble_index

    def resample_data(self, network, volume_evaluator: VolumeEvaluator, loss_evaluator: LossEvaluator, importance_sampler:IImportanceSampler):
        volume_evaluator.restore_defaults()
        loss_evaluator.set_source(volume=volume_evaluator)
        self._reset_data()
        for (timestep_index, timestep), (ensemble_index, ensemble) in product(enumerate(self.timestep_index), enumerate(self.ensemble_index)):
            volume_data = self.volume_data_storage.load_volume(timestep=timestep, ensemble=ensemble, index_access=False)
            network_evaluator = WorldSpaceDensityEvaluator(network, 0, timestep_index, ensemble_index)
            volume_evaluator.set_source(volume_data)
            loss_evaluator.set_source(network=network_evaluator)
            positions = importance_sampler.generate_samples(self.num_samples_per_volume, loss_evaluator)
            targets = volume_evaluator.evaluate(positions)
            positions = positions.cpu().numpy()
            targets = targets.cpu().numpy()
            tf_index_data, timestep_index_data, ensemble_index_data = self._build_index_data(0, timestep_index, ensemble_index, self.num_samples_per_volume)
            self._add_samples_to_data(positions, targets, tf_index_data, timestep_index_data, ensemble_index_data)
        self._finalize_data()
        volume_evaluator.restore_defaults()
        return self

    def num_timesteps(self):
        return len(self.timestep_index)

    def num_members(self):
        return len(self.ensemble_index)

    def num_ensembles(self):
        return self.num_members()

    def get_dataloader(self, batch_size=None, shuffle=False, drop_last=False, num_workers=0):
        if batch_size is None:
            batch_size = self.batch_size
        collate_fn = None
        if self.sub_batching is not None:
            sub_batch_size = np.ceil(self.batch_size / self.sub_batching)
            assert batch_size % sub_batch_size == 0, \
                f'[ERROR] Batch size {batch_size} is inconsistent with pre-processed sub-batch size ({sub_batch_size})'
            batch_size = self.sub_batching
            collate_fn = cat_collate
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers, collate_fn=collate_fn)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]


class WorldSpaceDensityEvaluator(NetworkEvaluator):

    def __init__(self, network, tf_index, timestep_index, ensemble_index):
        super(WorldSpaceDensityEvaluator, self).__init__(network)
        self.tf_index = tf_index
        self.timestep_idx = timestep_index
        self.ensemble_idx = ensemble_index

    def forward(self, positions: Tensor) -> Tensor:
        num_samples = positions.shape[0]
        tf_index, timestep_index, ensemble_index = self._build_index_data(num_samples)
        positions = positions.to(dtype=torch.float32)
        if positions.device != self.device:
            positions = positions.to(device=self.device)
        predictions = self.network(positions, tf_index, timestep_index, ensemble_index,'world')
        return predictions

    def _build_index_data(self, num_samples):
        tf_index_data = torch.full((num_samples,), self.tf_index, dtype=torch.int32, device=self.device)
        timestep_index_data = torch.full((num_samples,), self.timestep_idx, dtype=torch.float32, device=self.device)
        ensemble_index_data = torch.full((num_samples,), self.ensemble_idx, dtype=torch.float32, device=self.device)
        return tf_index_data, timestep_index_data, ensemble_index_data
