from typing import List, Any, Optional, Tuple

import torch
from torch import Tensor, nn

from volnet.modules.networks.latent_features.marginal.features import FeatureVector, FeatureGrid
from volnet.modules.networks.latent_features.indexing.key_indexer import KeyIndexer
from volnet.modules.networks.latent_features.init import IInitializer, DefaultInitializer
from volnet.modules.networks.latent_features.interface import IFeatureModule


class IEnsembleFeatures(IFeatureModule):

    def uses_member(self) -> bool:
        return True

    def uses_time(self) -> bool:
        return False

    def __init__(
            self,
            member_keys: List[Any], num_channels: int,
            initializer: Optional[IInitializer] = None,
            debug=False, device=None, dtype=None
    ):
        super(IEnsembleFeatures, self).__init__(3, num_channels, debug)
        if initializer is None:
            initializer = DefaultInitializer()
        self.initializer = initializer
        # self.member_index = KeyIndexer()
        self.key_mapping = {}
        self.feature_mapping = nn.ModuleDict({})
        self.device = device
        self.dtype = dtype
        self.reset_member_features(*member_keys)

    def evaluate(self, positions: Tensor, member: Tensor) -> Tensor:
        if self.is_debug():
            self._verify_inputs(positions, member)
        out = self.forward(positions, member)
        if self.is_debug():
            self._verify_outputs(positions, out)
        return out

    def forward(self, positions: Tensor, member: Tensor) -> Tensor:
        # unique_members, segments = self.member_index.query(member)
        # if len(unique_members) == 1:
        #     feature = self.feature_mapping[int(member[0].item())]
        #     return feature.evaluate(positions)
        # out = torch.empty(len(positions), self.num_channels(), device=positions.device, dtype=positions.dtype)
        # for member, segment in zip(unique_members, segments):
        #     feature = self.feature_mapping[int(member.item())]
        #     out[segment] = feature.evaluate(positions[segment])
        unique_members = torch.unique(member)
        if len(unique_members) == 1:
            feature = self.feature_mapping[int(unique_members[0].item())]
            return feature.evaluate(positions)
        out = torch.empty(len(positions), self.num_channels(), device=positions.device, dtype=positions.dtype)
        for umem in unique_members:
            feature = self.feature_mapping[int(umem.item())]
            locations = torch.eq(umem, member)
            out[locations] = feature.evaluate(positions[locations])
        return out

    def reset_member_features(self, *member_keys: Any) -> 'IEnsembleFeatures':
        raise NotImplementedError()

    def num_members(self) -> int:
        return len(self.key_mapping)

    def uses_positions(self) -> bool:
        raise NotImplementedError()


class EnsembleFeatureVector(IEnsembleFeatures):

    def reset_member_features(self, *member_keys: Any):
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        self.feature_mapping = nn.ModuleList([
            FeatureVector.from_initializer(
                self.initializer, self.num_channels(),
                device=self.device, dtype=self.dtype, debug=False
            ) for _ in range(len(member_keys))
        ])
        return self

    def uses_positions(self) -> bool:
        return False


class EnsembleFeatureGrid(IEnsembleFeatures):

    def __init__(
            self,
            member_keys: List[Any], num_channels: int, grid_size: Tuple[int, int, int],
            initializer: Optional[IInitializer] = None,
            debug=False, device=None, dtype=None
    ):
        self._grid_size = grid_size
        super(EnsembleFeatureGrid, self).__init__(
            member_keys, num_channels, initializer=initializer,
            debug=debug, device=device, dtype=dtype
        )

    def grid_size(self):
        return self._grid_size

    def reset_member_features(self, *member_keys: Any):
        self.key_mapping = {key: i for i, key in enumerate(member_keys)}
        self.feature_mapping = nn.ModuleList([
            FeatureGrid.from_initializer(
                self.initializer, self.grid_size(), self.num_channels(),
                device=self.device, dtype=self.dtype, debug=False
            ) for i in range(len(member_keys))
        ])
        return self

    def uses_positions(self) -> bool:
        return True
