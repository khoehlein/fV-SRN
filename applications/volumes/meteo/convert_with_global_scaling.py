import argparse
import os

import pyrenderer
import torch
import tqdm
import xarray as xr

from volumes.meteo.data_specs import Axis

BASE_PATH = '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/raw'
MEMBER_FILE_PATTERN = 'member{member:04d}.nc'
MEMBER_DIM_NAME = 'member'
DIMENSION_ORDER = [Axis.LONGITUDE.value, Axis.LATITUDE.value, Axis.LEVEL.value]
VARIABLE_NAMES = ['tk', 'rh', 'qv', 'z', 'dbz', 'qhydro']
OUTPUT_PATH = '/mnt/hdd10tb/Datasets/1k_member_ensemble_201606110000/cvol/single_variable'


def int_or_none(value: str):
    try:
        return int(value)
    except ValueError:
        return None


def load_data(variable_name: str, min_member: int, max_member: int, level_slice='-12:'):

    def _load_member(member_id):
        file_name = os.path.join(BASE_PATH, MEMBER_FILE_PATTERN.format(member=member_id))
        dataset = xr.open_dataset(file_name)
        variable = dataset[variable_name].isel({Axis.LEVEL.value: slice(*map(int_or_none, level_slice.split(':')))})
        return variable.expand_dims(dim={MEMBER_DIM_NAME: [member_id]}, axis=0)

    member_ids = list(range(min_member, max_member + 1))
    member_data = []

    with tqdm.tqdm(total=len(member_ids)) as pbar:
        for member_id in member_ids:
            member_data.append(_load_member(member_id))
            pbar.update(1)

    all_data = xr.concat(member_data, dim=MEMBER_DIM_NAME,)

    return all_data


def get_global_min_max_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME, Axis.LEVEL.value, Axis.LATITUDE.value, Axis.LONGITUDE.value]
    local_min = all_data.min(dim=summation_dims)
    local_max = all_data.max(dim=summation_dims)
    return local_min, (local_max - local_min)


def get_global_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME, Axis.LEVEL.value, Axis.LATITUDE.value, Axis.LONGITUDE.value]
    mu = all_data.mean(dim=summation_dims)
    sigma = all_data.std(dim=summation_dims, ddof=1)
    return mu, sigma


def get_level_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME, Axis.LATITUDE.value, Axis.LONGITUDE.value]
    mu = all_data.mean(dim=summation_dims)
    sigma = all_data.std(dim=summation_dims, ddof=1)
    return mu, sigma


def get_local_min_max_normalization(all_data: xr.DataArray):
    summation_dims = [MEMBER_DIM_NAME]
    local_min = all_data.min(dim=summation_dims)
    local_max = all_data.max(dim=summation_dims)
    return local_min, (local_max - local_min)


def convert_variable(variable_name: str, min_member: int, max_member: int, norm='global'):
    data = load_data('tk', min_member, max_member)
    if norm == 'global':
        mu, sigma = get_global_normalization(data)
    elif norm == 'level':
        mu, sigma = get_level_normalization(data)
    elif norm == 'local-min-max':
        mu, sigma = get_local_min_max_normalization(data)
    elif norm == 'global-min-max':
        mu, sigma = get_global_min_max_normalization(data)
    else:
        raise NotImplementedError()

    normalized_data = (data - mu) / sigma

    if not os.path.isdir(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for member_id in normalized_data.coords[MEMBER_DIM_NAME]:

        output_path = os.path.join(OUTPUT_PATH, f'{norm}_scaling', variable_name, 'member{:04d}'.format(member_id.values))
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        for timestep_idx, timestep in enumerate(normalized_data.coords[Axis.TIME.value]):
            snapshot = normalized_data.sel({MEMBER_DIM_NAME: member_id.values, Axis.TIME.value: timestep.values})
            snapshot = snapshot.transpose(*DIMENSION_ORDER)
            file_name = 't{:02d}.cvol'.format(timestep_idx)

            vol = pyrenderer.Volume()
            vol.worldX = 10.
            vol.worldY = 10.
            vol.worldZ = 1.
            vol.add_feature_from_tensor(variable_name, torch.from_numpy(snapshot.values)[None, ...])
            vol.save(os.path.join(output_path, file_name), compression=0)

    print('Finished')


def convert_ensemble(min_member: int, max_member: int, norm='global'):
    for variable_name in VARIABLE_NAMES:
        print(f'[INFO] Converting variable {variable_name}')
        convert_variable(variable_name, min_member, max_member, norm=norm)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-member', type=int, default=1)
    parser.add_argument('--max-member', type=int, default=128)
    parser.add_argument('--norm', type=str, default='global-min-max',choices=['level', 'global', 'local-min-max', 'global-min-max'])
    args = vars(parser.parse_args())
    convert_ensemble(args['min_member'], args['max_member'], norm=args['norm'])


if __name__ == '__main__':
    main()
