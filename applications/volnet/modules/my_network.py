import argparse
from itertools import product

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from common import utils
from volnet.input_data import TrainingInputData
from volnet.network import InputParametrization, OutputParametrization, InnerNetworkMeta, InnerNetwork


def pyrenderer_interp1D(fp, x):
    # fp: Tensor of shape (B, C, N)
    # x: Tensor of shape (B, M)
    x = x[..., None] * 2. / (fp.shape[-1] - 1.) - 1.
    grid = torch.stack([torch.zeros_like(x), x], dim=-1)
    out = F.grid_sample(fp[..., None], grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out[..., 0]


class _TimevariateFeaturesNd(nn.Module):

    def __init__(self, num_timesteps, num_features, resolution, dim):
        super(_TimevariateFeaturesNd, self).__init__()
        assert len(resolution) == dim
        self.num_timesteps = num_timesteps
        self.resolution = resolution
        self._initialize_fetaures(num_features)

    def _initialize_features(self, num_features):
        shape = (self.num_timesteps, num_features, *self.resolution)
        p = self.get_initial_feature_tensor(shape)
        self.register_parameter('features', nn.Parameter(p, requires_grad=True))

    def reset_features(self, num_features):
        if hasattr(self, 'features'):
            del self._parameters['features']
        self._initialize_features(num_features)
        return self

    def get_initial_feature_tensor(self, shape):
        raise NotImplementedError()

    def num_params(self):
        return self.features.sample_summary.numel()

    def num_timesteps(self):
        return self.features.shape[0]

    def num_features(self):
        return self.features.shape[1]


class TimevariateGaussianFeatures3d(_TimevariateFeaturesNd):

    def __init__(self, num_timesteps, num_features, resolution, mu=0., std=1.):
        self.mu = mu
        self.std = std
        super(TimevariateGaussianFeatures3d, self).__init__(num_timesteps, num_features, resolution, 3)

    def get_initial_feature_tensor(self, shape):
        return torch.randn(*shape) * self.std + self.mu

    def interpolate(self, x, idx=None):
        grid = x.unsqueeze(0).unsqueeze(1).unsqueeze(1)  # 1,N,1,1,3
        grid = grid * 2 - 1
        if idx is None:
            out = self._grid_sample(self.features, grid)
        else:
            idx = idx.item()
            num_timesteps = self.num_timesteps()
            idx_low = np.clip(int(np.floor(idx)), 0, num_timesteps - 1)
            idx_high = min(idx_low + 1, num_timesteps - 1)
            idx_f = idx - idx_low
            # interpolation in space
            latent_low = self._grid_sample(self.features[idx_low:idx_low + 1, ...], grid)
            latent_high = self._grid_sample(self.features[idx_high:idx_high + 1, ...], grid)
            # interpolate in time
            out = (1 - idx_f) * latent_low + idx_f * latent_high
        return out

    def _grid_sample(self, features, grid):
        return F.grid_sample(features, grid, align_corners=False, padding_mode='border')[0, :, 0, 0, :].t()


class TimevariateUniformFeatures1d(_TimevariateFeaturesNd):

    def __init__(self, num_timesteps, num_features, resolution, a=0, b=1):
        self.a = a
        self.b = b
        super(TimevariateUniformFeatures1d, self).__init__(num_timesteps, num_features, (resolution,), 1)

    def get_initial_feature_tensor(self, shape):
        return torch.rand(*shape) * (self.b - self.a) + self.a

    def interpolate(self, x):
        return pyrenderer_interp1D(self.features, x)


class MetaPretrainData():

    def __init__(self, config, device, dtype):
        super(MetaPretrainData, self).__init__()
        self._device = device
        self._dtype = dtype
        s = config.split(':')
        assert len(s) == 2
        self.epoch1 = int(s[0])
        self.epoch2 = int(s[1])
        self.current_epoch = 0
        self.tfs = None
        self.ensembles = None
        self.times = None

    def precompute_latent_variables(self, input_data):
        """
        To not store the input data (can't be pickled), precompute the inputs to
        the latent variables already.
        This will later be needed in self.start_epoch()
        :param input_data:
        """
        # query all latent variables
        tfs = []
        ensembles = []
        times = []

        num_tfs = input_data.num_tfs()
        num_timesteps = input_data.num_timesteps('train')
        num_ensembles = input_data.num_ensembles()
        for tf, timestep, ensemble in product(range(num_tfs), range(num_timesteps), range(num_ensembles)):
            actual_timestep, actual_ensemble = input_data.compute_actual_time_and_ensemble(
                timestep, ensemble, 'train')
            tfs.append(tf)
            ensembles.append(actual_ensemble)
            times.append(actual_timestep)

        self.tfs = torch.tensor(tfs, device=self._device, dtype=self._dtype)
        self.ensembles = torch.tensor(ensembles, device=self._device, dtype=self._dtype)
        self.times = torch.tensor(times, device=self._device, dtype=self._dtype)


class MySceneRepresentationNetwork(nn.Module):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        parser_group = parser.add_argument_group("Network")
        parser_group.add_argument('-om', '--outputmode',
                                  choices=["density", "density:direct", "rgbo", "rgbo:direct", "rgbo:exp"],
                                  type=str, default="density", help="""
                        The possible outputs of the network:
                        - density: a scalar density is produced that is then mapped to color via the TF.
                          This allows to use multiple TFs, see option 'randomizeTF'.
                        - density:direct: noop for world-space, clamp to [0,1] for screen-space
                        - rgbo: the network directly estimates red,green,blue,opacity/absorption. 
                          The TF is fixed during training and inference.                      
                        - rgbo:direct: noop for world-space, clamp to [0,1] for color, [0,infty] for absorption
                          for screen-space
                        """)
        parser_group.add_argument('-l', '--layers', default='32:32:32', type=str,
                                  help="The size of the hidden layers, separated by colons ':'")
        parser_group.add_argument('-a', '--activation', default="ReLU", type=str, help="""
                        The activation function for the hidden layers.
                        This is the class name for activations in torch.nn.** .
                        The activation for the last layer is fixed by the output mode.
                        To pass extra arguments, separate them by colons, e.g. 'Snake:2'""")
        parser_group.add_argument('-fn', '--fouriercount', default=0, type=int,
                                  help="Number of fourier features")
        parser_group.add_argument('-fs', '--fourierstd', default=1, type=float, help="""
            Standard Deviation of the fourier features, a positive value.
            If a negative value, the special NeRF-compatibility mode is used where the fourier features
            are block-identity matrices scaled by power of twos.
            """)
        parser_group.add_argument('--time_features', default=0, type=int,
                                  help="Feature size for timestep encoding")
        parser_group.add_argument('--ensemble_features', default=0, type=int,
                                  help="Feature size for ensemble encoding")
        parser_group.add_argument('--volumetric_features_channels', default=0, type=int,
                                  help="For volumetric latent spaces, specify the channels per voxel here")
        parser_group.add_argument('--volumetric_features_resolution', default=0, type=int,
                                  help="For volumetric latent spaces, specify the grid resolution here")
        parser_group.add_argument('--volumetric_features_std', default=0.01, type=float,
                                  help="Standard deviation for sampling the initial volumetric features")
        parser_group.add_argument('--volumetric_features_time_dependent', action='store_true', help="""
            If specified, the volumetric feature grid is time+ensemble-dependent.
            The split between time features and ensemble features is controlled by
            '--time_features' and '--ensemble_features', they must sum up to 
            '--volumetric_features_channels'.
            This results in two 5D grids:
             - time grid of shape (time_features, num_timesteps, volumetric_features_resolution^3)
             - ensemble grid of shape (ensemble_features, num_timesteps, volumetric_features_resolution^3)                 
            """)
        parser_group.add_argument('--use_direction', action='store_true')
        parser_group.add_argument('--disable_direction_in_fourier_features', action='store_true')
        parser_group.add_argument('--fourier_position_direction_split', default=-1, type=int, help="""
            If specified with a value in {0,1,...,fouriercount-1},
            the fourier matrix is split between positional and directional part.
            The first 'fourier_position_direction_split' fourier features only act
            on the position, the other fourier features only act on the direction. 
            """)
        parser_group.add_argument('--use_time_direct', action='store_true', help="""
            Uses time as a direct (scalar) input to the network""")
        parser_group.add_argument('--num_time_fourier', type=int, default=0, help="""
            Allocates that many inputs from 'fouriercount' to time encoding instead of position.""")
        parser_group.add_argument('--meta_network', default=None, type=str, help="""
            Alternative way how TF/Ensemble/Time is encoded:
            The default, if this parameter is not specified, is to send the latent vectors
            for the TF/Ensemble/Time as additional input to the network.
            With this parameter, a second meta-network instead is first called that
            predicts the weights of the actual scene representation network from the latent vectors.
            This parameter specifies the hidden layers of that meta network, e.g. '64:64:64'.
            """)
        parser_group.add_argument('--meta_activation', default="ReLU", type=str, help="""
                        The activation function for the hidden layers in the meta network.
                        This is the class name for activations in torch.nn.** .
                        The activation for the last layer is fixed by the output mode.""")
        parser_group.add_argument('--meta_pretrain', default=None, type=str, help="""
            To improve stability, the meta-network can use a pre-training method.
            To enable this, specify this parameter with two integers "e1:e2".
            First, only the inner network is trained for e1 epochs with a set of parameters
            that is independent of the ensemble, time or TF. This allows the network
            to find a first coarse match.
            Then, the meta-network is trained to predict that set of parameters for
            all latent variables for e2 epochs.
            Only after that is the full pipeline trained end-to-end.
            """)

    def __init__(self, opt: dict, input: TrainingInputData, dtype, device):
        """
        Initializes the scene reconstruction network with the dictionary obtained from
        the ArgumentParser
        :param opt: the dictionary with the results from the ArgumentParser
        """
        super().__init__()
        self._dtype = dtype
        self._device = device

        self._input_parametrization = InputParametrization(
            has_direction=opt['use_direction'],
            fourier_std=opt['fourierstd'], num_fourier_features=opt['fouriercount'],
            disable_direction_in_fourier=opt['disable_direction_in_fourier_features'],
            fourier_position_direction_split=opt['fourier_position_direction_split'],
            use_time_direct=opt['use_time_direct'], num_time_fourier=opt['num_time_fourier'])

        self._output_parametrization = OutputParametrization(opt['output_mode'])

        self._build_feature_modules(opt, input)

        self._build_inner_network(opt, input)

    def _build_feature_modules(self, opt, input):
        self._time_features = opt['time_features']
        self._ensemble_features = opt['ensemble_features']
        self._volumetric_features_channels = opt['volumetric_features_channels'] or 0

        volumetric_features_resolution = opt['volumetric_features_resolution'] or 0
        volumetric_features_time_dependent = opt['volumetric_features_time_dependent'] or False
        
        has_volumetric_features = self._volumetric_features_channels > 0 and volumetric_features_resolution > 0

        if not has_volumetric_features:
            self._volumetric_features_channels = 0  # so that the feature computation below is accurate
 
        if volumetric_features_time_dependent:
            assert has_volumetric_features, "A time-dependent volumetric feature grid is requested, but the resolution or channel count is zero"
            assert self._volumetric_features_channels == self._time_features + self._ensemble_features, \
                "A time-dependent volumetric feature grid is requested, but volumetric_features_channels!=time_features+ensemble_features"

        latent_space_params = 0
        self.latent_features = nn.ModuleDict({})
        
        if has_volumetric_features:
            self.latent_features['3d'] = nn.ModuleDict({key: None for key in ['time', 'ensemble', 'joint']})
            std = opt['volumetric_features_std']
            if volumetric_features_time_dependent:
                num_params = self._build_marginal_volumetric_feature_space(input, std, volumetric_features_resolution)
            else:
                num_params = self._build_joint_volumetric_feature_space(std, volumetric_features_resolution)
            latent_space_params += num_params
        
        if not volumetric_features_time_dependent:
            self.latent_features['1d'] = nn.ModuleDict({key: None for key in ['time', 'ensemble']})
            num_params = self._build_marginal_linear_feature_space(input)
            latent_space_params += num_params

        print("Latent space memory:", utils.humanbytes(latent_space_params * 4))

    def _build_marginal_linear_feature_space(self, input):
        num_params = 0
        if self._time_features > 0:
            self.latent_features['1d']['time'] = TimevariateUniformFeatures1d(
                1, self._time_features, input.num_timekeyframes()
            )
            num_params += self.latent_features['1d']['time'].num_params()
        if self._ensemble_features > 0:
            self.latent_features['1d']['ensemble'] = TimevariateUniformFeatures1d(
                1, self._ensemble_features, input.num_ensembles()
            )
            num_params += self.latent_features['1d']['time'].num_params()
        return num_params

    def _build_joint_volumetric_feature_space(self, std, volumetric_features_resolution):
        self.latent_features['3d']['joint'] = TimevariateGaussianFeatures3d(
            1, self._volumetric_features_channels, [volumetric_features_resolution] * 3, std=std
        )
        num_params = self.latent_features['3d']['joint'].num_params()
        return num_params

    def _build_marginal_volumetric_feature_space(self, input, std, volumetric_features_resolution):
        num_params = 0
        if self._time_features > 0:
            self.latent_features['3d']['time'] = TimevariateGaussianFeatures3d(
                input.num_timekeyframes(), self._time_features, [volumetric_features_resolution] * 3,
                std=std
            )
            num_params += self.latent_features['3d']['time'].num_params()
        if self._ensemble_features > 0:
            self.latent_features['3d']['ensemble'] = TimevariateGaussianFeatures3d(
                input.num_ensembles(), self._ensemble_features, [volumetric_features_resolution] * 3,
                std=std
            )
            num_params += self.latent_features['3d']['time'].num_params()
        return num_params

    def _total_latent_size(self):
        channels = self._time_features + self._ensemble_features
        if '3d' in self.latent_features and self.latent_features['3d']['joint'] is not None:
            channels = channels + self._volumetric_features_channels
        return channels

    def _build_inner_network(self, opt, input):
        meta_pretrain_config = opt['meta_pretrain']
        if meta_pretrain_config is not None:
            self._meta_pretrain_data = MetaPretrainData(meta_pretrain_config, self._device, self._dtype)
            self._meta_pretrain_data.precompute_latent_variables(input)
        else:
            self._meta_pretrain_data = None
        meta_network_config = opt['meta_network']
        self._has_meta_network = self._meta_network_config is not None
        if meta_network_config is not None:
            self._hidden_layers = InnerNetworkMeta(
                self._input_parametrization.num_output_channels(),
                self.output_channels(),
                opt['layers'], opt['activation'],
                opt['meta_network'], opt['meta_activation'],
                self._total_latent_size(),
                (self._meta_pretrain_data is not None)
            )
        else:
            self._hidden_layers = InnerNetwork(
                self._input_parametrization.num_output_channels() + self._total_latent_size(),
                self.output_channels(),
                opt['layers'], opt['activation'],
                self._total_latent_size())

    def generalize_to_new_ensembles(self, num_members: int):
        """
        Prepares for generalization-training:
        Replaces the ensemble latent space grid with a new grid for
        the desired number of ensembles.
        :param num_members: the number of ensemble members
        """
        if '3d' not in self.latent_features or self.latent_features['3d']['ensemble'] is None:
            raise ValueError("Network wasn't loaded/initialized with ensemble-dependent volumentric latent grids")
        self.latent_features['3d']['ensemble'].reset_features(num_members)
        return self.latent_features['3d']['ensemble'].num_channels

    # def export_to_pyrenderer(self, opt,
    #                          grid_encoding, return_grid_encoding_error=False): # pyrenderer.SceneNetwork.LatentGrid.Encoding):
    #     """
    #     Exports this network to the pyrenderer TensorCore implementation
    #     :param opt: the opt dictionary. Used keys:
    #         layers, activation, ensembles, time_keyframes
    #     :param grid_encoding:
    #     :return:
    #     """
    #     n = pyrenderer.SceneNetwork()
    #
    #     # input
    #     B = self._input_parametrization.get_fourier_feature_matrix()
    #     n.input.has_direction = self.use_direction()
    #     if B is not None:
    #         n.input.set_fourier_matrix_from_tensor(B, self._input_parametrization.is_premultiplied())
    #     else:
    #         n.input.disable_fourier_features()
    #     if self._input_parametrization.has_time():
    #         if self.volumetric_features is None and self.latent_space is not None:
    #             raise ValueError("time input only possible (for now) for time-dependent latent grids")
    #         n.input.has_time = True
    #     else:
    #         n.input.has_time = False
    #
    #     #output
    #     n.output.output_mode = pyrenderer.SceneNetwork.OutputParametrization.OutputModeFromString(
    #         self.output_mode())
    #
    #     # grid
    #     encoding_error = 0
    #     encoding_error_count = 0
    #     if self.volumetric_features is not None:
    #         if self.volumetric_features['time'] is not None:
    #             time_keyframes = list(range(*map(int, opt['time_keyframes'].split(':'))))
    #             ensemble_range = list(range(*map(int, opt['ensembles'].split(':'))))
    #             time_num = len(time_keyframes) if self._time_features > 0 else 0
    #             ensemble_num = len(ensemble_range) if self._ensemble_features > 0 else 0
    #             grid_info = pyrenderer.SceneNetwork.LatentGridTimeAndEnsemble(
    #                 time_min=time_keyframes[0],
    #                 time_num=time_num,
    #                 time_step=time_keyframes[1] - time_keyframes[0] if len(time_keyframes)>1 else 1,
    #                 ensemble_min=ensemble_range[0],
    #                 ensemble_num=ensemble_num)
    #
    #             if self._time_features > 0:
    #                 grid_time = self._volumetric_latent_space_time
    #                 assert grid_time.shape[0] == len(time_keyframes)
    #                 for i in range(len(time_keyframes)):
    #                     e = grid_info.set_time_grid_from_torch(i, grid_time[i:i + 1], grid_encoding)
    #                     encoding_error += e
    #                     encoding_error_count += 1
    #             else:
    #                 assert len(time_keyframes)<=1, "time features disabled, but there were time keyframes in the dataset"
    #
    #             if self._ensemble_features > 0:
    #                 grid_ensemble = self._volumetric_latent_space_ensemble
    #                 assert grid_ensemble.shape[0] == len(ensemble_range)
    #                 for i in range(len(ensemble_range)):
    #                     e = grid_info.set_ensemble_grid_from_torch(i, grid_ensemble[i:i+1], grid_encoding)
    #                     encoding_error += e
    #                     encoding_error_count += 1
    #             else:
    #                  assert len(ensemble_range)<=1, "ensemble features disabled, but there were ensemble frames in the dataset"
    #
    #             n.latent_grid = grid_info
    #         else:
    #             grid_static = self._volumetric_latent_space
    #             # save static grid as time grid with one timestep
    #             grid_info = pyrenderer.SceneNetwork.LatentGridTimeAndEnsemble(
    #                 time_min=0, time_num=1, time_step=1,
    #                 ensemble_min=0, ensemble_num=0)
    #             e = grid_info.set_time_grid_from_torch(0, grid_static, grid_encoding)
    #             encoding_error += e
    #             encoding_error_count += 1
    #             n.latent_grid = grid_info
    #         if not n.latent_grid.is_valid():
    #             raise ValueError("LatentGrid is invalid")
    #
    #     #hidden
    #     assert isinstance(self._hidden_layers, InnerNetwork)
    #     layers = opt['layers']
    #     activation = opt['activation']
    #     activationX = activation.split(':')
    #     activation_param = float(activationX[1]) if len(activationX)>=2 else 1
    #     activation = pyrenderer.SceneNetwork.Layer.ActivationFromString(activationX[0])
    #     layer_sizes = list(map(int, layers.split(':')))
    #     for i, s in enumerate(layer_sizes):
    #         layer = getattr(self._hidden_layers, 'linear%d' % i)
    #         assert isinstance(layer, nn.Linear)
    #         n.add_layer(layer.weight, layer.bias, activation, activation_param)
    #     last_layer = getattr(self._hidden_layers, 'linear%d'%len(layer_sizes))
    #     n.add_layer(last_layer.weight, last_layer.bias, pyrenderer.SceneNetwork.Layer.Activation.NONE)
    #
    #     if not n.valid():
    #         raise ValueError("Failed to convert network to TensorCores")
    #
    #     if return_grid_encoding_error:
    #         return n, encoding_error/encoding_error_count if encoding_error_count>0 else 0
    #     return n

    def supports_mixed_latent_spaces(self):
        """
        Returns if the network supports mixed latent spaces.
        True -> tf, time, ensemble are torch tensors in the forward method
        False -> tf, time, ensemble are still torch tensors, but only the first entry is used,
            i.e. it is assumed that all latent vectors are identical
        :return: if mixed latent space is supported
        """
        if self._has_meta_network: return False
        if '1d' in self.latent_features: return False
        return True

    def output_mode(self):
        return self._output_parametrization.output_mode()

    def use_direction(self):
        return self._input_parametrization.has_direction()

    def num_time_features(self):
        return self._time_features

    def num_ensemble_features(self):
        return self._ensemble_features

    def num_volumetric_features(self):
        return self._volumetric_features_channels

    def base_input_channels(self):
        return self._input_parametrization.num_input_channels()

    def total_input_channels(self):
        return self.base_input_channels() + self._total_latent_size()

    def output_channels(self):
        return self._output_parametrization.num_output_channels()

    def start_epoch(self) -> bool:
        """
        Called when an epoch is started.
        This is used to control the pretraining, so that the main script does not
        need to know the details
        :return: true if the optimizer should be reset (i.e. a new phase is entered)
        """
        if self._meta_pretrain_data is not None:
            self._meta_pretrain_data.current_epoch += 1
            if (self._meta_pretrain_data.current_epoch > self._meta_pretrain_data.epoch1):
                print("Pretraining of inner network done, now match the meta-network with the parameters")
                self._match_meta_network_with_parameters()
                return True
            return False
        else:
            return False # no pretraining -> nothing to do

    def _match_meta_network_with_parameters(self):
        with torch.no_grad():
            features = []
            if self.latent_features['1d']['ensemble'] is not None:
                ensembles = self._meta_pretrain_data.ensembles.unsqueeze(1)
                features.append(self.latent_features['1d']['ensemble'].interpolate(ensembles)[..., 0])
            if self.latent_features['1d']['time'] is not None:
                times = self._meta_pretrain_data.times.unsqueeze(1)
                features.append(self.latent_features['1d']['time'].interpolate(times)[..., 0])
            z = torch.cat(features, dim=1)
        # finish network
        assert isinstance(self._hidden_layers, InnerNetworkMeta)
        self._hidden_layers.finish_pretraining(z, self._meta_pretrain_data.epoch2)
        self._meta_pretrain_data = None

    def forward(self, x, tf, time, ensemble, mode: str):
        """
        'x' of shape (B,3)
        'tf', 'time', 'ensemble' of shape (B)
        if self.supports_mixed_latent_spaces() == False only the first entry of 'tf', 'time' and 'ensemble' are used.
        :param x: N,3
        :param tf:
        :param time:
        :param ensemble:
        :return:
        """

        assert mode in ['screen', 'world']

        x2 = [x]
        if self._input_parametrization.has_time():
            x2.append(time.unsqueeze(1))

        if not self.supports_mixed_latent_spaces():
            assert torch.all(ensemble == ensemble[:1])
            assert torch.all(time == time[:1])
            tf = tf[:1]
            ensemble = ensemble[:1]
            time = time[:1]

        features = []
        # Don't change order for compatibility with Sebastian!
        if '3d' in self.latent_features:
            if self.latent_features['3d']['time'] is not None:
                features.append(self.latent_features['3d']['time'](x, idx=time))
            if self.latent_features['3d']['ensemble'] is not None:
                features.append(self.latent_features['3d']['ensemble'](x, idx=ensemble))
        if '1d' in self.latent_features:
            if self.latent_features['1d']['ensemble'] is not None:
                features.append(self.latent_features['1d']['ensemble'].interpolate(ensemble.unsqueeze(1))[..., 0])
            if self.latent_features['1d']['time'] is not None:
                features.append(self.latent_features['1d']['time'].interpolate(time.unsqueeze(1))[..., 0])
        if '3d' in self.latent_features:
            if self.latent_features['3d']['joint'] is not None:
                features.append(self.latent_features['3d']['joint'].interpolate(x[..., :3]))

        if self._has_meta_network:
            x = torch.cat(x2, dim=1)
            y = self._input_parametrization(x)
            z = torch.cat(features, dim=1)
            y = self._hidden_layers(z, y)
        else:
            x2 = x2 + features
            x = torch.cat(x2, dim=1)
            y = self._input_parametrization(x)
            y = self._hidden_layers(y)
        
        return self._output_parametrization(y, mode=mode)
