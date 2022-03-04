import argparse
from typing import Union, Optional

import pyrenderer
import torch
from torch import Tensor

from volnet.modules.networks.processing.interface import ICoreNetwork
from volnet.modules.networks.processing.core_network.modulated_sine import ModulatedSineProcessor
from volnet.modules.networks.processing.core_network.residual_sine import ResidualSineProcessor
from volnet.modules.networks.processing.core_network.simple_mlp import SimpleMLP


class CoreNetwork(ICoreNetwork):

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

    @classmethod
    def from_dict(cls, args, data_input_channels: int, latent_input_channels: int, output_channels: int):
        prefix = 'network:core:'
        layer_sizes = list(map(int, args[prefix + 'layer_sizes'].split(':')))
        activation, *activation_params = args[prefix + 'activation'].split(':')
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
        if output_channels == 4: #rgba
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
        super(CoreNetwork, self).__init__()
        self.processor = processor

    def forward(self, data_input: Tensor, latent_input: Union[Tensor, None]) -> Tensor:
        return self.processor(data_input, latent_input)

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
