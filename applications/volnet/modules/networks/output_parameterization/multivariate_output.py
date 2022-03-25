from torch import Tensor

from volnet.modules.datasets import OutputMode
from volnet.modules.networks.output_parameterization import BackendOutputMode
from volnet.modules.networks.output_parameterization.constant_channel_parameterization import \
    ConstantChannelParameterization


class MultivariateOutput(ConstantChannelParameterization):

    def __init__(self, channels):
        super(MultivariateOutput, self).__init__(channels, OutputMode.MULTIVARIATE)

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        return network_output

    def backend_output_mode(self) -> BackendOutputMode:
        return BackendOutputMode.MULTIVARIATE
