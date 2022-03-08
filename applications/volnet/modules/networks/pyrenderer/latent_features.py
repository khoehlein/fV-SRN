import argparse
from typing import Dict, Any, Optional, Union, Tuple, List

import numpy as np
import pyrenderer
import torch

from volnet.modules.networks.latent_features.init import DefaultInitializer
from volnet.modules.networks.latent_features.marginal import (
    MarginalLatentFeatures,
    TemporalFeatureGrid, TemporalFeatureVector,
    EnsembleFeatureGrid, EnsembleFeatureVector,
    FeatureVector, FeatureGrid
)


class PyrendererLatentFeatures(MarginalLatentFeatures):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('LatentFeatures')
        prefix = '--network:latent-features:'
        group.add_argument(
            prefix + 'time:num-channels', type=int, default=0,
            help="""
            number of channels for time-related latent-features
            """
        )
        group.add_argument(
            prefix + 'time:grid-size', type=str, default=None,
            help="""
            grid size for volumetric temporal features 
            (Default: None, ie. don't use position in temporal features)
            """
        )
        group.add_argument(
            prefix + 'time:key-frames', type=str, default=None,
            help="""
            Key frame specification for temporal features as "min_time:max_time:num_steps" 
            (Default: None, ie. use frames as obtained from the dataset)
            """
        )
        group.add_argument(
            prefix + 'ensemble:num-channels', type=int, default=0,
            help="""
            number of channels for ensemble-related latent-features
            """
        )
        group.add_argument(
            prefix + 'ensemble:grid-size', type=str, default=None,
            help="""
            grid size for volumetric ensemble features 
            (Default: None, ie. don't use position in ensemble features)
            """
        )
        group.add_argument(
            prefix + 'volume:num-channels', type=int, default=0,
            help="""
            number of channels for position-related latent-features
            """
        )
        group.add_argument(
            prefix + 'volume:grid-size', type=str, default=None,
            help="""
            grid size for purely volumetric features 
            (Default: None, ie. use feature vector without position dependence)
            """
        )

    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            member_keys: Optional[List[int]] = None, dataset_key_times: Optional[List[float]] = None
    ):
        prefix = 'network:latent_features:'

        def get_arg(name):
            return args[prefix + name]

        def read_grid_specs(gs: str):
            gs = gs.split(':')
            assert len(gs) in {1, 3}
            if len(gs) == 1:
                gs = [gs[0]] * 3
            out = (int(gs[0]), int(gs[1]), int(gs[2]))
            return out

        temporal_channels = get_arg('time:num_channels')
        if temporal_channels > 0:
            key_time_specs = get_arg('time:key_frames')
            if key_time_specs is None:
                assert dataset_key_times is not None
                key_times = torch.tensor(dataset_key_times)
            else:
                min_time, max_time, num_steps = key_time_specs.split(':')
                key_times = torch.linspace(float(min_time), float(max_time), int(num_steps))
            grid_specs = get_arg('time:grid_size')
            if grid_specs is None:
                temporal_features = TemporalFeatureVector(key_times, temporal_channels)
            else:
                grid_size = read_grid_specs(grid_specs)
                temporal_features = TemporalFeatureGrid(key_times, temporal_channels, grid_size)
        else:
            temporal_features = None

        ensemble_channels = get_arg('ensemble:num_channels')
        if ensemble_channels > 0:
            assert member_keys is not None, \
                '[ERROR] member keys must be provided if ensemble features are to be used.'
            grid_specs = get_arg('ensemble:grid_size')
            if grid_specs is None:
                ensemble_features = EnsembleFeatureVector(member_keys, ensemble_channels)
            else:
                grid_size = read_grid_specs(grid_specs)
                ensemble_features = EnsembleFeatureGrid(member_keys, ensemble_channels, grid_size)
        else:
            ensemble_features = None

        volumetric_channels = get_arg('volume:num_channels')
        if volumetric_channels > 0:
            grid_specs = get_arg('volume:grid_size')
            initializer = DefaultInitializer()
            if grid_specs is None:
                volumetric_features = FeatureVector.from_initializer(initializer, volumetric_channels)
            else:
                grid_size = read_grid_specs(grid_specs)
                volumetric_features = FeatureGrid.from_initializer(initializer, grid_size, volumetric_channels)
        else:
            volumetric_features = None
        if temporal_features is not None or ensemble_features is not None or volumetric_features is not None:
            return cls(temporal_features, ensemble_features, volumetric_features)
        else:
            return None

    def export_to_pyrenderer(
            self,
            grid_encoding,
            network: Optional[pyrenderer.SceneNetwork] = None,
            return_grid_encoding_error=False
    ) -> Union[pyrenderer.SceneNetwork, Tuple[pyrenderer.SceneNetwork, float]]:
        if self.uses_linear_features():
            raise RuntimeError('[ERROR] Use of linear features is not supported in pyrenderer export!')
        if network is None:
            network = pyrenderer.SceneNetwork()
        encoding_error = 0
        encoding_error_count = 0
        if self.uses_positions():
            if self.uses_time() or self.uses_member():
                time_key_frames = self.temporal_features.get_time_key_frames().tolist() if self.uses_time() else [0]
                if len(time_key_frames) > 1:
                    delta = time_key_frames[0] - time_key_frames[1]
                    expected = delta * np.arange(len(time_key_frames)) + time_key_frames[0]
                    assert np.all(np.array(time_key_frames) == expected), \
                        '[ERROR] Pyrenderer doesnot support irregular time grids.'
                ensemble_keys = self.ensemble_features.get_ensemble_keys() if self.uses_member() else [0]
                if len(ensemble_keys) > 1:
                    expected = np.arange(len(ensemble_keys)) + ensemble_keys[0]
                    assert np.all(np.array(ensemble_keys) == expected), \
                        '[ERROR] Pyrenderer does not support irregular member keys.'
                grid_info = pyrenderer.SceneNetwork.LatentGridTimeAndEnsemble(
                    time_min=time_key_frames[0],
                    time_num=self.temporal_features.num_key_times() if self.uses_time() else 0,
                    time_step=time_key_frames[1] - time_key_frames[0] if len(time_key_frames) > 1 else 1,
                    ensemble_min=min(ensemble_keys),
                    ensemble_num=self.ensemble_features.num_members() if self.uses_member() else 0)
                if self.uses_time():
                    grid = self.temporal_features.get_grid()
                    for i in range(self.temporal_features.num_key_times()):
                        e = grid_info.set_time_grid_from_torch(i, grid[i:i + 1], grid_encoding)
                        encoding_error += e
                        encoding_error_count += 1
                if self.uses_member():
                    grid = self.ensemble_features.get_grid()
                    for i in range(self.ensemble_features.num_members()):
                        e = grid_info.set_ensemble_grid_from_torch(i, grid[i:i + 1], grid_encoding)
                        encoding_error += e
                        encoding_error_count += 1
                network.latent_grid = grid_info
            else:
                grid = self.volumetric_features.get_grid()
                grid_info = pyrenderer.SceneNetwork.LatentGridTimeAndEnsemble(
                    time_min=0, time_num=1, time_step=1,
                    ensemble_min=0, ensemble_num=0)
                e = grid_info.set_time_grid_from_torch(0, grid, grid_encoding)
                encoding_error += e
                encoding_error_count += 1
                network.latent_grid = grid_info
            if not network.latent_grid.is_valid():
                raise RuntimeError('[ERROR] Exported latent grid is invalid')
        if return_grid_encoding_error:
            out = encoding_error / encoding_error_count if encoding_error_count > 0 else 0
            return network, out
        return network
