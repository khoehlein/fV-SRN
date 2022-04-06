import argparse
from typing import Union, Optional, Dict, Any

import pyrenderer
import torch
from torch import Tensor

from volnet.modules.datasets.output_mode import OutputMode
from volnet.modules.networks.core_network import ICoreNetwork
from .modulated_sine import ModulatedSineProcessor
from .residual_sine import ResidualSineProcessor
from .simple_mlp import SimpleMLP
from ...input_parameterization import IInputParameterization
from ...latent_features import ILatentFeatures
from ...output_parameterization import IOutputParameterization


class PyrendererCoreNetwork(ICoreNetwork):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('CoreNetwork')
        prefix = '--network:core:'
        group.add_argument(prefix + 'layer-sizes', default='32:32:32', type=str,
                           help="The size of the hidden layers, separated by colons ':'")
        group.add_argument(prefix + 'activation', default="ReLU", type=str, help="""
                                The activation function for the hidden layers.
                                This is the class name for activations in torch.nn.** .
                                The activation for the last layer is fixed by the output mode.
                                To pass extra arguments, separate them by colons, e.g. 'Snake:2'""")
        group.add_argument(prefix + 'split-members', action='store_true', dest='network_core_split_members')
        group.set_defaults(network_core_split_members=False)


    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            input_parameterization: IInputParameterization,
            latent_features: ILatentFeatures,
            output_parameterization: IOutputParameterization,
            member_keys=None
    ):
        if args['network_core_split_members']:
            assert member_keys is not None
            return PyrendererMultiCoreNetwork.from_dict(args, input_parameterization, latent_features, output_parameterization, member_keys)
        else:
            return PyrendererSingleCoreNetwork.from_dict(args, input_parameterization, latent_features, output_parameterization)


class PyrendererSingleCoreNetwork(PyrendererCoreNetwork):

    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            input_parameterization: IInputParameterization,
            latent_features: ILatentFeatures,
            output_parameterization: IOutputParameterization,
            member_keys=None,
    ):
        prefix = 'network:core:'
        layer_sizes = list(map(int, args[prefix + 'layer_sizes'].split(':')))
        activation, *activation_params = args[prefix + 'activation'].split(':')
        activation_params = [float(p) for p in activation_params]
        data_input_channels = input_parameterization.output_channels()
        latent_input_channels = latent_features.output_channels() if latent_features is not None else 0
        output_channels = output_parameterization.output_channels()
        if activation == "ModulatedSine":
            processor = ModulatedSineProcessor(
                data_input_channels, latent_input_channels, output_channels,
                layer_sizes,
            )
        elif activation == "ResidualSine":
            processor = ResidualSineProcessor(
                data_input_channels, latent_input_channels, output_channels,
                layer_sizes
            )
        else:
            processor = SimpleMLP(
                data_input_channels, latent_input_channels, output_channels,
                layer_sizes, activation, activation_params
            )
        if output_parameterization.output_mode() == OutputMode.RGBO: #rgba
            last_layer = processor.last_layer()
            last_layer.bias.sample_summary = torch.abs(last_layer.bias.sample_summary) + 1.0 # positive output to see something
        #else:
        #    last_layer.weight.data = 100 * last_layer.weight.data
        return cls(processor)

    def __init__(self, processor: ICoreNetwork):
        """
        :param data_input_channels: InputParametrization.num_output_channels()
        :param output_channels: OutputParametrization.num_input_channels()
        :param layers: colon-separated list of hidden layer sizes
        :param activation: activation function, torch.nn.**
        :param latent_input_channels: the size of the latent vector (for modulated sine)
        """
        super(PyrendererSingleCoreNetwork, self).__init__()
        self.processor = processor

    def forward(
            self, data_input: Tensor, latent_input: Union[Tensor, None],
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member: Tensor
    ) -> Tensor:
        return self.processor(data_input, latent_input, positions, transfer_functions, time, member)

    def data_input_channels(self) -> int:
        return self.processor.data_input_channels()

    def latent_input_channels(self) -> int:
        return self.processor.latent_input_channels()

    def output_channels(self) -> int:
        return self.processor.data_input_channels()

    def last_layer(self):
        return self.processor.last_layer()

    def export_to_pyrenderer(self, network: Optional[pyrenderer.SceneNetwork] = None) -> pyrenderer.SceneNetwork:
        return self.processor.export_to_pyrenderer(network=network)


class PyrendererMultiCoreNetwork(PyrendererCoreNetwork):

    @classmethod
    def from_dict(
            cls, args: Dict[str, Any],
            input_parameterization: IInputParameterization,
            latent_features: ILatentFeatures,
            output_parameterization: IOutputParameterization,
            member_keys=None
    ):
        return PyrendererCoreNetwork._single_model_from_dict(args, input_parameterization, latent_features, output_parameterization)