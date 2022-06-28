import collections
from typing import List, Tuple, Union, Optional

import pyrenderer
import torch
from torch import nn, Tensor

from volnet.modules.networks.core_network import ICoreNetwork


class ProcessorSequentialWrapper(ICoreNetwork):

    def export_to_pyrenderer(self, network: Optional[pyrenderer.SceneNetwork] = None, time=None) -> pyrenderer.SceneNetwork:
        if network is None:
            network = pyrenderer.SceneNetwork()
        activation_param = float(self._activation_params[0]) if len(self._activation_params) >= 1 else 1
        activation = pyrenderer.SceneNetwork.Layer.ActivationFromString(self._activation)
        for i, s in enumerate(self._layer_sizes):
            layer = getattr(self.layers, f'linear{i}')
            assert isinstance(layer, nn.Linear)
            network.add_layer(layer.weight, layer.bias, activation, activation_param)
        last_layer = getattr(self.layers, f'linear{len(self._layer_sizes)}' )
        network.add_layer(last_layer.weight, last_layer.bias, pyrenderer.SceneNetwork.Layer.Activation.NONE)
        return network

    def __init__(
            self,
            data_input_channels, latent_input_channels, output_channels,
            layers: List[Tuple[str, nn.Module]], layer_sizes: List[int],
            activation: str, activation_params: List[str],
    ):
        super(ProcessorSequentialWrapper, self).__init__()
        self._data_input_channels = data_input_channels
        self._latent_input_channels = latent_input_channels
        self._output_channels = output_channels
        self._layer_sizes = layer_sizes
        self._activation = activation
        self._activation_params = activation_params
        self.layers = nn.Sequential(collections.OrderedDict(layers))

    def forward(
            self, data_input: Tensor, latent_input: Union[Tensor, None],
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member:Tensor
    ) -> Tensor:
        if latent_input is None:
            assert self.latent_input_channels() == 0
            joint_input = data_input
        else:
            joint_input = torch.cat([data_input, latent_input], dim=-1)
        output = self.layers(joint_input)
        return output

    def last_layer(self):
        return self.layers[-1]

    def data_input_channels(self) -> int:
        return self._data_input_channels

    def latent_input_channels(self) -> int:
        return self._latent_input_channels

    def output_channels(self) -> int:
        return self._output_channels
