from typing import Union, Optional

import pyrenderer
from torch import nn, Tensor


class ICoreNetwork(nn.Module):

    def forward(
            self,
            data_input: Tensor, latent_input: Union[Tensor, None],
            positions: Tensor, transfer_functions: Tensor, time: Tensor, member: Tensor
    ) -> Tensor:
        raise NotImplementedError()

    def data_input_channels(self) -> int:
        raise NotImplementedError()

    def latent_input_channels(self) -> int:
        raise NotImplementedError()

    def output_channels(self) -> int:
        raise NotImplementedError()

    def last_layer(self):
        raise NotImplementedError()
