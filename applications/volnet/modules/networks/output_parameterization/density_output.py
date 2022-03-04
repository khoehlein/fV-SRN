import torch
from torch import Tensor

from volnet.modules.networks.postprocessing.constant_channel_parameterization import ConstantChannelParameterization
from volnet.modules.networks.postprocessing.backend_output_mode import BackendOutputMode
from volnet.modules.datasets.output_mode import OutputMode


class _DensityOutput(ConstantChannelParameterization):

    def __init__(self):
        super(_DensityOutput, self).__init__(1, OutputMode.DENSITY)


class SoftClampDensityOutput(_DensityOutput):

    @staticmethod
    def _transform(network_output: Tensor) -> Tensor:
        return torch.sigmoid(network_output)

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return self._transform(network_output)

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return self._transform(network_output)

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.DENSITY


class DirectDensityOutput(_DensityOutput):

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return torch.clamp(network_output, min=0, max=1)

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.DENSITY_DIRECT


CHOICES = {
    'soft-clamp': SoftClampDensityOutput,
    'direct': DirectDensityOutput,
}