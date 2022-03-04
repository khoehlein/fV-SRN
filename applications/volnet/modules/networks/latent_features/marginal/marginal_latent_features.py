from typing import Any, Optional, Union, Tuple

import numpy as np
import pyrenderer
import torch
from torch import Tensor

from volnet.modules.networks.latent_features.indexing.features import FeatureGrid
from volnet.modules.networks.latent_features.interface import ILatentFeatures
from volnet.modules.networks.latent_features.marginal.ensemble_features import IEnsembleFeatures
from volnet.modules.networks.latent_features.marginal.temporal_features import ITemporalFeatures


class MarginalLatentFeatures(ILatentFeatures):

    def uses_member(self) -> bool:
        return self.ensemble_features is not None

    def uses_time(self) -> bool:
        return self.temporal_features is not None

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

    def __init__(
            self,
            temporal_features: Optional[ITemporalFeatures] = None,
            ensemble_features: Optional[IEnsembleFeatures] = None,
            volumetric_features: Optional[FeatureGrid] = None,
    ):
        dimension = None
        if dimension is None and temporal_features is not None:
            dimension = temporal_features.dimension
        if dimension is None and ensemble_features is not None:
            dimension = ensemble_features.dimension
        if dimension is None and volumetric_features is not None:
            dimension = volumetric_features.dimension
        assert dimension is not None, '[ERROR] At least one of temporal, ensemble or volumetric features must not be None.'
        num_features = 0
        debug = False
        if temporal_features is not None:
            assert temporal_features.dimension == dimension
            assert temporal_features.num_channels() > 0
            num_features = num_features + temporal_features.num_channels()
            debug = temporal_features.is_debug() or debug
            temporal_features.set_debug(False)
        if ensemble_features is not None:
            assert ensemble_features.dimension == dimension
            assert ensemble_features.num_channels() > 0
            num_features = num_features + ensemble_features.num_channels()
            debug = ensemble_features.is_debug() or debug
            ensemble_features.set_debug(False)
        if volumetric_features is not None:
            assert volumetric_features.dimension == dimension
            assert volumetric_features.num_channels() > 0
            num_features = num_features + volumetric_features.num_channels()
            debug = volumetric_features.is_debug() or debug
            volumetric_features.set_debug(False)
        super(MarginalLatentFeatures, self).__init__(dimension, num_features, debug)
        self.temporal_features = temporal_features
        self.ensemble_features = ensemble_features
        self.volumetric_features = volumetric_features

    def forward(self, positions: Tensor, time: Tensor, member: Tensor) -> Tensor:
        features = []
        if self.temporal_features is not None:
            features.append(self.temporal_features.evaluate(positions, time))
        if self.ensemble_features is not None:
            features.append(self.ensemble_features.evaluate(positions, member))
        if self.volumetric_features is not None:
            features.append(self.volumetric_features.evaluate(positions))
        out = torch.cat(features, dim=-1)
        return out

    def reset_member_features(self, *member_keys: Any) -> 'MarginalLatentFeatures':
        self.ensemble_features.reset_member_features(*member_keys)
        return self

    def num_key_times(self) -> int:
        if self.temporal_features is None:
            return 0
        return self.temporal_features.num_key_times()

    def num_members(self) -> int:
        if self.ensemble_features is None:
            return 0
        return self.ensemble_features.num_members()

    def uses_positions(self):
        if self.volumetric_features is not None:
            return True
        if self.temporal_features is not None and self.temporal_features.uses_positions():
            return True
        if self.ensemble_features is not None and self.ensemble_features.uses_positions():
            return True
        return False

    def uses_linear_features(self):
        if self.temporal_features is not None and not self.temporal_features.uses_positions():
            return True
        if self.ensemble_features is not None and not self.ensemble_features.uses_positions():
            return True
        return False
