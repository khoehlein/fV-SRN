from typing import Dict, Any, Optional

import pyrenderer
from torch import nn, Tensor

from volnet.modules.networks.evaluation_mode import EvaluationMode


class ISceneRepresentationNetwork(nn.Module):

    def generalize_to_new_ensembles(self, num_members: int):
        raise NotImplementedError()

    def export_to_pyrenderer(self, grid_encoding, return_grid_encoding_error=False, network: Optional[pyrenderer.SceneNetwork] = None):
        raise NotImplementedError()

    def supports_mixed_latent_spaces(self):
        raise NotImplementedError()

    def backend_output_mode(self):
        raise NotImplementedError()

    def uses_direction(self):
        raise NotImplementedError()

    def num_latent_features_time(self):
        raise NotImplementedError()

    def num_latent_features_ensemble(self):
        raise NotImplementedError()

    def start_epoch(self):
        raise NotImplementedError()

    def forward(
            self,
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member:Tensor,
            evaluation_mode: EvaluationMode
    ):
        raise NotImplementedError()