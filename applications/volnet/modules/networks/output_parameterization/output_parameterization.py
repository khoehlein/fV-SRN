import argparse
from typing import Dict, Any, Optional

from torch import Tensor

from volnet.modules.networks.postprocessing.density_output import CHOICES as density_outputs_by_name
from volnet.modules.networks.postprocessing.rgbo_output import CHOICES as rgbo_outputs_by_name
from volnet.modules.networks.evaluation_mode import EvaluationMode
from volnet.modules.networks.postprocessing.interface import IOutputParameterization
from volnet.modules.networks.postprocessing.backend_output_mode import BackendOutputMode
from volnet.modules.datasets.output_mode import OutputMode

_choices_by_output_mode = {
    OutputMode.DENSITY: density_outputs_by_name,
    OutputMode.RGBO: rgbo_outputs_by_name,
    OutputMode.MULTIVARIATE: {} # currently not supported
}


class OutputParameterization(IOutputParameterization):

    OUTPUT_MODE: OutputMode = None # will be set during parser initialization

    @classmethod
    def set_output_mode(cls, output_mode: OutputMode):
        assert output_mode != OutputMode.MULTIVARIATE, '[ERROR] Multivariate output is currently not supported in OutputParameterization'
        cls.OUTPUT_MODE = output_mode

    @classmethod
    def init_parser(cls, parser: argparse.ArgumentParser, output_mode: Optional[OutputMode] = None):
        group = parser.add_argument_group('OutputParameterization')
        prefix = '--network:output:'
        if cls.OUTPUT_MODE is None:
            assert output_mode is not None
            cls.set_output_mode(output_mode)
        choices = list(_choices_by_output_mode[cls.OUTPUT_MODE].keys())
        group.add_argument(
            prefix + 'parameterization-mode', choices=choices, type=str, default='',
            help="""
            The possible outputs of the network:
            - density: a scalar density is produced that is then mapped to color via the TF.
                * soft-clamp: Sigmoid clamp to [0, 1]
                * direct: noop during training, clamp to [0,1] during rendering
            - rgbo: the network directly estimates red, green, blue, opacity/absorption. The TF is fixed during training and inference.                      
                * soft-clamp: Sigmoid clamp to [0, 1] for color, softplus clamping to [0, infty] for absorption
                * direct: noop for training, clamp to [0,1] for color, [0, infty] for absorption for rendering
                * exp: Sigmoid clamp to [0, 1] for color, exponential clamping to [0, infty] for absorption
            """
        )

    @classmethod
    def from_dict(cls, args: Dict[str, Any], output_mode: Optional[OutputMode]=None):
        if output_mode is None:
            assert cls.OUTPUT_MODE is not None, '[ERROR] OutputParameterization output mode must be set before instance can be created from arguments.'
            output_mode = cls.OUTPUT_MODE
        parameterization_mode = args['network:output:parameterization-mode']
        parameterization_class = _choices_by_output_mode[output_mode][parameterization_mode]
        return cls(parameterization_class())

    def __init__(self, parameterization: IOutputParameterization):
        self._parameterization = parameterization

    def input_channels(self) -> int:
        return self._parameterization.input_channels()

    def output_channels(self) -> int:
        return self._parameterization.output_channels()

    def output_mode(self) -> OutputMode:
        return self._parameterization.output_mode()

    def backend_output_mode(self) -> BackendOutputMode:
        return self._parameterization.backend_output_mode()

    def forward(self, network_output: Tensor, evaluation_mode: EvaluationMode) -> Tensor:
        return self._parameterization.forward(network_output, evaluation_mode)
