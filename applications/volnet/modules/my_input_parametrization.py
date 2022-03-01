import numpy as np
from numpy import pi as PI 
import torch
from torch import nn


class InputParametrization(nn.Module):
    # in earlier checkpoints, the factor 2pi was applied to the fourier matrix during 'forward'
    # In newer version, this is done directly in the constructor (more efficient).
    # For old checkpoints, this variable is set during loading to false
    PREMULTIPLY_2_PI = True

    def __init__(self, *,
                 has_direction = False,
                 num_fourier_features: int = 0,
                 fourier_std: float = 1,
                 disable_direction_in_fourier: bool = True,
                 fourier_position_direction_split: int = -1,
                 use_time_direct: bool = False,
                 num_time_fourier: int = 0):
        super().__init__()
        self._has_direction = has_direction
        self._num_fourier_features = num_fourier_features
        self._disable_direction_in_fourier = disable_direction_in_fourier if (disable_direction_in_fourier is not None) else True
        self._use_time_direct = use_time_direct or False
        self._num_time_fourier = num_time_fourier or 0
        self._fourier_position_direction_split = fourier_position_direction_split or -1
        self._premultiply2pi = InputParametrization.PREMULTIPLY_2_PI
        if num_fourier_features > 0:
            self._initialize_spatial_b_matrix(fourier_std)
            if self._num_time_fourier > 0:
                self._initialize_temporal_b_matrix(fourier_std)
        assert has_direction or not disable_direction_in_fourier #disable_direction_in_fourier implies has_direction

    def _initialize_spatial_b_matrix(self, fourier_std):
        out = 6 if (self._has_direction and not self._disable_direction_in_fourier) else 3
        num_fourier_features = self._num_fourier_features
        if self._num_time_fourier > 0:
            num_position_fourier = num_fourier_features - self._num_time_fourier
        else:
            num_position_fourier = num_fourier_features

        if fourier_std > 0:
            # random gaussian
            B = torch.normal(0, fourier_std, (num_position_fourier, out))
        else:
            # scaled block-identity, based on NeRF
            assert self._fourier_position_direction_split < 0, "fourier-split not compatible with NeRF-position-matrix"
            num_blocks = int(np.ceil(num_position_fourier / out))
            Bx = []
            for i in range(num_blocks):
                Bx.append(((2 ** i)) * torch.eye(out, out))
            B = torch.cat(Bx, dim=0)[:num_position_fourier, :]
        if self._premultiply2pi:
            B = B * (2 * PI)
        if self._fourier_position_direction_split >= 0:
            assert self._has_direction and not self._disable_direction_in_fourier
            assert self._fourier_position_direction_split < num_position_fourier
            # set directional component for [:fourier_position_direction_split] to zero
            B[:self._fourier_position_direction_split, 3:].zero_()
            # set positional component for [fourier_position_direction_split:] to zero
            B[self._fourier_position_direction_split:, :3].zero_()
        self.register_buffer('B', B.t())

    def _initialize_temporal_b_matrix(self, fourier_std):
        if fourier_std > 0:
            B_time = torch.normal(0, fourier_std, (self._num_time_fourier, 1))
        else:
            B_time = torch.tensor([2 * PI * (2 ** i) for i in range(self._num_time_fourier)], dtype=torch.float32)
            B_time = B_time.unsqueeze(1)
        self.register_buffer('B_time', B_time.t())


    def has_direction(self):
        return self._has_direction

    def has_position(self):
        return True

    def is_premultiplied(self):
        return self._premultiply2pi

    def has_time(self):
        return self._use_time_direct or self._num_time_fourier>0

    def num_input_channels(self):
        """
        Returns the number of input channels:
        3 for position (x,y,z)
        3 for direction if enabled (dx, dy, dz)
        :return: the number of input channels
        """
        return 3 + (3 if self._has_direction else 0) + (1 if self.has_time() else 0)

    def _num_direct_output_channels(self):
        """
        Returns the number of input channels that are directly passed on to the output
        """
        return 3 + (3 if self._has_direction else 0) + (1 if self._use_time_direct else 0)

    def num_output_channels(self):
        """
        :return: the number of output channels
        """
        out = 3 + (3 if self._has_direction else 0) + (1 if self._use_time_direct else 0)
        return out + 2*self._num_fourier_features

    def get_fourier_feature_matrix(self):
        """
        :return: the fourier feature matrix of shape B*3 or None if no fourier features are here
        """
        return self.B.t() if self._num_fourier_features > 0 else None

    def forward(self, x):
        """
        Input parametrization from (B, Cin) to (B, Cout)
        where Cin=self.num_input_channels(), Cout=self.num_output_channels().
        Any additional channels are simply added to the end,
        use this for latent vectors for timestep, ensemble, TF.
        """
        assert len(x.shape)==2, \
            "input is not of shape (B,Cin), but " + str(x.shape)

        extra_channels = x.shape[1] - self.num_input_channels()
        assert extra_channels >= 0, f"At least {self.num_input_channels()} channels expected, but got {x.shape[1]}"

        if self._num_fourier_features > 0:
            x_base = x[:, :self._num_direct_output_channels()]
            x_extra = x[:, self.num_input_channels():]
            if self._has_direction and self._disable_direction_in_fourier:
                x_fourier = x[:,:3]
            elif self._has_direction and not self._disable_direction_in_fourier:
                x_fourier = x[:, :6]
            else: # not self._has_direction:
                x_fourier = x[:, :3]
            f = torch.matmul(x_fourier, self.B)
            if not self._premultiply2pi:
                f = 2 * PI * f

            x_parts = [
                x_base,
                torch.cos(f),
                torch.sin(f)]

            if self._num_time_fourier > 0:
                x_time = x[:, 3:4]
                ftime = torch.matmul(x_time, self.B_time) # B_time has the factor 2*pi backed in already
                x_parts.append(torch.cos(ftime))
                x_parts.append(torch.sin(ftime))

            x_parts.append(x_extra)
            x = torch.cat(x_parts, dim=1)

        return x
