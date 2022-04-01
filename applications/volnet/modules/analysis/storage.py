import math
from typing import Dict, Any, Tuple

import numpy as np

from volnet.modules.helpers import parse_range_string

FLOAT_SIZE= 32


def compute_latent_channels(args: Dict[str, Any]):
    ensemble_channels = args['network:latent_features:ensemble:num_channels']
    volume_channels = args['network:latent_features:volume:num_channels']
    time_channels = args['network:latent_features:time:num_channels']
    return ensemble_channels + volume_channels + time_channels


def compute_decoder_memory_size(args: Dict[str, Any]):
    fourier_channels = args['network:input:fourier:positions:num_features']
    latent_channels = compute_latent_channels(args)
    layer_sizes = [int(s) for s in args['network:core:layer_sizes'].split(':') if s.isnumeric()]
    input_channels = 3 + 2 * fourier_channels + latent_channels
    output_channels = 1
    current_channels = input_channels
    layer_num_params = []
    for new_channels in [*layer_sizes, output_channels]:
        weight_size = current_channels * new_channels
        bias_size = new_channels
        layer_num_params.append(weight_size + bias_size)
        current_channels = new_channels
    memory = np.sum(layer_num_params) * FLOAT_SIZE
    return memory


def compute_latent_feature_memory_size(args: Dict[str, Any]):
    ensemble_memory = compute_ensemble_feature_memory(args)
    volume_memory = compute_volume_feature_memory(args)
    time_memory = compute_time_feature_memory(args)
    return ensemble_memory + volume_memory + time_memory


def compute_ensemble_feature_memory(args: Dict[str, Any]):
    ensemble_channels = args['network:latent_features:ensemble:num_channels']
    if ensemble_channels == 0:
        return 0
    mode = args['network:latent_features:ensemble:mode']
    num_members = len(parse_range_string(args['data_storage:ensemble:index_range']))
    if mode == 'vector':
        return ensemble_channels * num_members * FLOAT_SIZE
    elif mode == 'grid':
        resolution = _read_grid_specs(args['network:latent_features:ensemble:grid:resolution'])
        return ensemble_channels * num_members * _compute_grid_numelem(resolution) * FLOAT_SIZE
    elif mode == 'multi-res':
        coarsest = _read_grid_specs(args['network:latent_features:ensemble:multi_res:coarsest'])
        finest = _read_grid_specs(args['network:latent_features:ensemble:multi_res:finest'])
        num_levels = args['network:latent_features:ensemble:multi_res:num_levels']
        table_size = args['network:latent_features:ensemble:multi_res:table_size']
        return num_members * compute_multires_memory(ensemble_channels, coarsest, finest, num_levels, table_size)
    elif mode == 'multi-grid':
        num_grids = args['network:latent_features:ensemble:multi_grid:num_grids']
        resolution = _read_grid_specs(args['network:latent_features:ensemble:multi_grid:resolution'])
        return ensemble_channels * num_grids * _compute_grid_numelem(resolution) * FLOAT_SIZE + num_grids * num_members * FLOAT_SIZE
    else:
        raise NotImplementedError()


def compute_volume_feature_memory(args: Dict[str, Any]):
    volume_channels = args['network:latent_features:volume:num_channels']
    if volume_channels == 0:
        return 0
    mode = args['network:latent_features:volume:mode']
    if mode == 'vector':
        return volume_channels * FLOAT_SIZE
    elif mode == 'grid':
        resolution = _read_grid_specs(args['network:latent_features:volume:grid:resolution'])
        return volume_channels * _compute_grid_numelem(resolution) * FLOAT_SIZE
    elif mode == 'multi-res':
        coarsest = _read_grid_specs(args['network:latent_features:volume:multi_res:coarsest'])
        finest = _read_grid_specs(args['network:latent_features:volume:multi_res:finest'])
        num_levels = args['network:latent_features:volume:multi_res:num_levels']
        table_size = args['network:latent_features:volume:multi_res:table_size']
        return compute_multires_memory(volume_channels, coarsest, finest, num_levels, table_size)
    else:
        raise NotImplementedError()


def compute_time_feature_memory(args: Dict[str, Any]):
    time_channels = args['network:latent_features:time:num_channels']
    if time_channels == 0:
        return 0
    mode = args['network:latent_features:time:mode']
    num_key_frames = _read_num_key_frames(args)
    if mode == 'vector':
        return num_key_frames * time_channels * FLOAT_SIZE
    elif mode == 'grid':
        resolution = _read_grid_specs(args['network:latent_features:time:grid:resolution'])
        return time_channels * num_key_frames * _compute_grid_numelem(resolution) * FLOAT_SIZE
    else:
        raise NotImplementedError()


def _read_num_key_frames(args: Dict[str, Any]):
    key_time_specs = args['network:latent_features:time:key_frames']
    assert key_time_specs is not None
    min_time, max_time, num_steps = key_time_specs.split(':')
    import torch
    key_times = torch.linspace(float(min_time), float(max_time), int(num_steps))
    return len(key_times)


def _read_grid_specs(gs: str):
    gs = gs.split(':')
    assert len(gs) in {1, 3}
    if len(gs) == 1:
        gs = [gs[0]] * 3
    out = (int(gs[0]), int(gs[1]), int(gs[2]))
    return out


def _compute_grid_numelem(resolution: Tuple[int, ...]):
    return np.prod(np.array(resolution))


def compute_multires_memory(num_channels, coarsest, finest, num_levels, table_size):
    grid_channels = int(num_channels // num_levels)
    grid_memory = []

    for l in range(num_levels):

        def compute_resolution(c, f):
            b = math.exp((math.log(f) - math.log(c)) / (num_levels - 1))
            return int(math.floor(c * b ** l))

        resolution = tuple([compute_resolution(c, f) for c, f in zip(coarsest, finest)])
        num_elem = _compute_grid_numelem(resolution)
        if num_elem <= table_size:
            grid_memory.append(num_elem * grid_channels)
        else:
            grid_memory.append(table_size * grid_channels)

    return np.sum(grid_memory) * FLOAT_SIZE


def _test():
    params = {
    "--world-vis-data:enable_image_caching": False,
    "absorption_weighting": 0.1,
    "data_storage:base_path": None,
    "data_storage:ensemble:index_range": "1:129",
    "data_storage:filename_pattern": "/home/hoehlein/data/1000_member_ensemble/cvol/single_variable/local-min-max_scaling/tk/member{member:04d}/t04.cvol",
    "data_storage:timestep:index_range": None,
    "dataset_resampling:frequency": 20,
    "dataset_resampling:loss": "l1",
    "dataset_resampling:method": "random",
    "dssim": 0,
    "epochs": 400,
    "global_seed": 124,
    "importance_sampler:grid:batch_size": None,
    "importance_sampler:grid:grid_size": None,
    "importance_sampler:grid:min_density": 0.01,
    "importance_sampler:grid:num_samples_per_voxel": 8,
    "importance_sampler:grid:sub_sampling": None,
    "importance_sampler:tree:alpha": 0.05,
    "importance_sampler:tree:batch_size": None,
    "importance_sampler:tree:max_depth": 12,
    "importance_sampler:tree:max_ratio": 10,
    "importance_sampler:tree:min_density": 0.01,
    "importance_sampler:tree:min_depth": 4,
    "importance_sampler:tree:num_samples_per_node": 128,
    "importance_sampler:warp_net:entropy_regularization": 0.0001,
    "importance_sampler:warp_net:hidden_channels": 32,
    "importance_sampler:warp_net:log_p_regularization": 0.0001,
    "importance_sampler:warp_net:lr": 0.001,
    "importance_sampler:warp_net:num_batches": 8,
    "importance_sampler:warp_net:num_samples_per_batch": 262144,
    "importance_sampler:warp_net:num_transforms": 4,
    "importance_sampler:warp_net:spline_order": 8,
    "importance_sampler:warp_net:weight_decay": 0.0001,
    "l1": 1.0,
    "l2": 0,
    "lossmode": "density",
    "lpips": 0,
    "lr": 0.01,
    "lr_gamma": 0.5,
    "lr_step": 50,
    "multiply_alpha": False,
    "network:core:activation": "SnakeAlt:2",
    "network:core:layer_sizes": "32:32:32:32",
    "network:input:fourier:method": "nerf",
    "network:input:fourier:positions:method": None,
    "network:input:fourier:positions:num_features": 14,
    "network:input:fourier:random:std": 0.01,
    "network:input:fourier:time:method": None,
    "network:input:fourier:time:num_features": 0,
    "network:latent_features:ensemble:grid:resolution": None,
    "network:latent_features:ensemble:mode": "multi-grid",
    "network:latent_features:ensemble:multi_grid:num_grids": 16,
    "network:latent_features:ensemble:multi_grid:resolution": "4:88:64",
    "network:latent_features:ensemble:multi_res:coarsest": None,
    "network:latent_features:ensemble:multi_res:finest": None,
    "network:latent_features:ensemble:multi_res:num_levels": 2,
    "network:latent_features:ensemble:multi_res:table_size": None,
    "network:latent_features:ensemble:num_channels": 16,
    "network:latent_features:time:grid:resolution": None,
    "network:latent_features:time:key_frames": None,
    "network:latent_features:time:mode": "vector",
    "network:latent_features:time:num_channels": 0,
    "network:latent_features:volume:grid:resolution": None,
    "network:latent_features:volume:mode": "vector",
    "network:latent_features:volume:multi_res:coarsest": None,
    "network:latent_features:volume:multi_res:finest": None,
    "network:latent_features:volume:multi_res:num_levels": 2,
    "network:latent_features:volume:multi_res:table_size": None,
    "network:latent_features:volume:num_channels": 0,
    "network:output:parameterization_method": "direct",
    "network_core_split_members": False,
    "network_inputs_use_direct_time": False,
    "optim_params": "{}",
    "optimizer": "Adam",
    "output:base_dir": "/home/hoehlein/PycharmProjects/results/fvsrn/rescaled_ensemble/multi_member_multi_grid",
    "output:checkpoint_dir": "/home/hoehlein/PycharmProjects/results/fvsrn/rescaled_ensemble/multi_member_multi_grid/results/model",
    "output:experiment_name": None,
    "output:hdf5_dir": "/home/hoehlein/PycharmProjects/results/fvsrn/rescaled_ensemble/multi_member_multi_grid/results/hdf5",
    "output:log_dir": "/home/hoehlein/PycharmProjects/results/fvsrn/rescaled_ensemble/multi_member_multi_grid/results/log",
    "output:save_frequency": 40,
    "profile": False,
    "renderer:settings_file": "/home/hoehlein/PycharmProjects/production/fvsrn/applications/config-files/meteo-ensemble_tk_local-min-max.json",
    "renderer:tf_dir": None,
    "sampling:cache": None,
    "sampling:method": "random",
    "sampling:training:method": None,
    "sampling:validation:method": None,
    "verify_files_exist": True,
    "world_density_data:batch_size": 524288,
    "world_density_data:ensemble_index_slice": ":",
    "world_density_data:num_samples_per_volume": 262144,
    "world_density_data:sub_batching": 8,
    "world_density_data:timestep_index_slice": ":",
    "world_density_data:training:batch_size": None,
    "world_density_data:training:ensemble_index_slice": None,
    "world_density_data:training:timestep_index_slice": None,
    "world_density_data:validation:batch_size": None,
    "world_density_data:validation:ensemble_index_slice": None,
    "world_density_data:validation:timestep_index_slice": None,
    "world_density_data:validation_mode": "sample",
    "world_density_data:validation_share": 0.2,
    "world_vis_data:enable_image_caching": False,
    "world_vis_data:ensemble_index_slice": ":",
    "world_vis_data:refinements": None,
    "world_vis_data:resolution": 256,
    "world_vis_data:resolution:X": None,
    "world_vis_data:resolution:Y": None,
    "world_vis_data:step_size": 0.005,
    "world_vis_data:timestep_index_slice": ":"
}
    print('Decoder memory:', compute_decoder_memory_size(params))
    print('Latent feature memory:', compute_latent_feature_memory_size(params))


if __name__ == '__main__':
    _test()
