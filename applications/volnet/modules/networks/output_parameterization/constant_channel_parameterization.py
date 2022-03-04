from typing import Callable, Dict

from torch import Tensor

from volnet.modules.networks.evaluation_mode import EvaluationMode
from volnet.modules.networks.postprocessing.interface import IOutputParameterization
from volnet.modules.networks.postprocessing.backend_output_mode import BackendOutputMode
from volnet.modules.datasets.output_mode import OutputMode


class ConstantChannelParameterization(IOutputParameterization):

    def __init__(self, channels: int, output_mode: OutputMode):
        super(ConstantChannelParameterization, self).__init__()
        self._channels = channels
        self._output_mode = output_mode
        self._parameterization_mapping: Dict[EvaluationMode, Callable[[Tensor], Tensor]] = {
            EvaluationMode.RENDERING: self.rendering_parameterization,
            EvaluationMode.TRAINING: self.training_parameterization,
        }

    def input_channels(self) -> int:
        return self._channels

    def output_channels(self) -> int:
        return self._channels

    def output_mode(self) -> OutputMode:
        return self._output_mode

    def backend_output_mode(self) -> BackendOutputMode:
        raise NotImplementedError()

    def forward(self, network_output: Tensor, evaluation_mode: EvaluationMode) -> Tensor:
        assert network_output.shape[-1] == self.input_channels()
        out = self._parameterization_mapping[evaluation_mode](network_output)
        assert network_output.shape[1] == self.output_channels()
        return out

    def rendering_parameterization(self, network_output: Tensor) -> Tensor:
        raise NotImplementedError()

    def training_parameterization(self, network_output: Tensor) -> Tensor:
        raise NotImplementedError()
