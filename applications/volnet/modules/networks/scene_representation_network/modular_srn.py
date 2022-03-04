from typing import Optional, Tuple, Union

import pyrenderer
from torch import Tensor

from volnet.modules.networks.latent_features import ILatentFeatures
from volnet.modules.networks.evaluation_mode import EvaluationMode
from volnet.modules.networks.postprocessing.interface import IOutputParameterization
from volnet.modules.networks.preprocessing import IInputParameterization
from volnet.modules.networks.processing import ICoreNetwork
from volnet.modules.networks.scene_representation_network.interface import ISceneRepresentationNetwork


class ModularSRN(ISceneRepresentationNetwork):

    def __init__(
            self,
            input_parameterization: IInputParameterization,
            core_network: ICoreNetwork,
            output_parameterization: IOutputParameterization,
            latent_features: Optional[ILatentFeatures] = None
    ):
        super(ModularSRN, self).__init__()
        self.input_parameterization = input_parameterization
        self.core_network = core_network
        self.output_parameterization = output_parameterization
        self.latent_features = latent_features

    def forward(
            self,
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member:Tensor,
            evaluation_mode: EvaluationMode
    ) -> Tensor:
        data_input = self.input_parameterization.forward(positions, transfer_functions, time, member)
        if self.latent_features is not None:
            latent_inputs = self.latent_features.evaluate(positions, time, member)
        else:
            latent_inputs = None
        network_output = self.core_network.forward(data_input, latent_inputs)
        prediction = self.output_parameterization.forward(network_output, evaluation_mode)
        return prediction

    def export_to_pyrenderer(
            self,
            grid_encoding, return_grid_encoding_error=False,
            network: Optional[pyrenderer.SceneNetwork]=None
    ) -> Union[pyrenderer.SceneNetwork, Tuple[pyrenderer.SceneNetwork, float]]:
        if self.input_parameterization.uses_time() and (self.latent_features is None or not self.latent_features.uses_time()):
            raise RuntimeError(
                "[ERROR] Time input for pyrenderer.SceneNetwork() works only for time-dependent latent grids (for now).")
        network = self.input_parameterization.export_to_pyrenderer(network=network)
        network = self.output_parameterization.export_to_pyrenderer(network=network)
        if self.latent_features is not None:
            network, error = self.latent_features.export_to_pyrenderer(grid_encoding, network, return_grid_encoding_error=True)
        else:
            error = 0.
        network = self.core_network.export_to_pyrenderer(network=network)
        if not network.valid():
            raise RuntimeError('[ERROR] Failed to convert scene representation network to tensor cores.')
        if return_grid_encoding_error:
            return network, error
        return network
