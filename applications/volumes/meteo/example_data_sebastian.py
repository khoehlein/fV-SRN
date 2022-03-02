import argparse
import os
import re
from enum import Enum

import numpy as np
import torch
import xarray as xr

import common.utils as commut # required to properly find pyrenderer

import pyrenderer


class Axis(Enum):
    TIME = 'time'
    LEVEL = 'lev'
    LATITUDE = 'lat'
    LONGITUDE = 'lon'


class EnsembleConverter(object):

    MEMBER_PATTERN = re.compile(r"member\d{4}.nc")
    DIMENSION_ORDER = [Axis.LONGITUDE, Axis.LATITUDE, Axis.LEVEL]
    VELOCITY_NAMES = {
        Axis.LONGITUDE: 'u',
        Axis.LATITUDE: 'v',
        Axis.LEVEL: 'w',
    }

    def __init__(self, source_dir, target_dir,):
        self.source_dir = source_dir
        self.target_dir = target_dir
        self._member_files = None

    def read_source_directory(self):
        print('[INFO] Reading source directory')
        assert os.path.isdir(self.source_dir), f'[ERROR] {self.source_dir} is not a valid directory'
        member_files = set()
        other_files = set()
        for f in os.listdir(self.source_dir):
            (member_files if self.MEMBER_PATTERN.match(f) else other_files).add(f)
        assert len(member_files) > 0, '[ERROR] Source directory does not contain any member files'
        print(f'[INFO] Found source directory with {len(member_files)} member files and {len(other_files)}')
        self._member_files = sorted(member_files)
        return self._member_files

    def validate_create_target_directory(self, overwrite=False, create_target_dir=False):
        if os.path.isdir(self.target_dir):
            contents = os.listdir(self.target_dir)
            if len(contents) > 0:
                if overwrite:
                    print('[INFO] Found non-empty target directory. Contents will be overwritten')
                else:
                    raise Exception('[INFO] Found non-empty target directory')
        elif os.path.exists(self.target_dir):
            raise Exception('[ERROR] Target directory path exists, but is not a valid directory')
        else:
            print('[INFO] Target directory does not exist.')
            if create_target_dir:
                os.makedirs(self.target_dir)
            else:
                raise Exception('[INFO] Target directory does not exist')

    @staticmethod
    def _prepare_subdirectory(directory, folder):
        sub_directory = os.path.join(directory, folder)
        if not os.path.isdir(sub_directory):
            os.makedirs(sub_directory)
        return sub_directory

    def convert_variable(self, var_name, drop_nan_levels=False, attach_velocity=False, clamp_min=None, clamp_max=None, max_member=None, split_to_member_files=True):
        self._prepare_subdirectory(self.target_dir, var_name)
        if max_member is None:
            max_member = len(self._member_files)
        if split_to_member_files:
            for file_name in self._member_files[:max_member]:
                self._convert_variable_in_file(file_name, var_name, drop_nan_levels, attach_velocity, clamp_min, clamp_max)
        else:
            self._export_multi_member_files(attach_velocity, clamp_max, clamp_min, drop_nan_levels, max_member, var_name)

    def _export_multi_member_files(self, attach_velocity, clamp_max, clamp_min, drop_nan_levels, max_member, var_name):
        volumes = None
        for file_name in self._member_files[:max_member]:
            dataset = xr.open_dataset(os.path.join(self.source_dir, file_name))
            variable = self._select_variable(dataset, var_name)
            velocity = self._select_velocity(dataset) if attach_velocity else None
            member_id = os.path.splitext(file_name)[0]
            if volumes is None:
                def get_volume():
                    vol = pyrenderer.Volume()
                    vol.worldX = 10
                    vol.worldY = 10
                    vol.worldZ = 1
                    return vol

                volumes = [get_volume() for _ in range(len(variable.coords[Axis.TIME.value]))]
            for i, t in enumerate(variable.coords[Axis.TIME.value]):
                variable_snapshot, velocity_snapshot = self._select_snapshot(t, variable, velocity)
                if drop_nan_levels:
                    variable_snapshot, velocity_snapshot = self._drop_nan_levels(variable_snapshot,
                                                                                 velocity_snapshot)
                variable_snapshot, velocity_snapshot = self._transpose_dimensions(variable_snapshot,
                                                                                  velocity_snapshot)
                variable_data, velocity_data = self._extract_data(variable_snapshot, velocity_snapshot)
                print(f'[INFO] '
                      f'Data shape: {variable_data.shape}, ' +
                      f'Dtype: {variable_data.dtype}, ' +
                      f'Min: {np.min(variable_data)}, ' +
                      f'Max: {np.max(variable_data)}, ' +
                      f'Mean: {np.mean(variable_data)}, ' +
                      f'Std: {np.std(variable_data)}'
                      )
                variable_data = self._normalize_data(variable_data, clamp_min, clamp_max)
                full_data = variable_data
                if velocity_data is not None:
                    full_data = np.concatenate([full_data, velocity_data], axis=0)
                print(
                    f'[INFO] Made it to the conversion step! Data shape: {full_data.shape}')
                volumes[i].add_feature_from_tensor(member_id, torch.from_numpy(full_data))
        for i, vol in enumerate(volumes):
            storage_path = os.path.join(self.target_dir, var_name, 't{:02d}.cvol'.format(var_name, i))
            vol.save(os.path.abspath(storage_path), compression=0)

    def _convert_variable_in_file(self, file_name, var_name, drop_nan_levels, attach_velocity, clamp_min, clamp_max):
        print(f'[INFO] Processing file {file_name}')
        dataset = xr.open_dataset(os.path.join(self.source_dir, file_name))
        variable = self._select_variable(dataset, var_name)
        velocity = self._select_velocity(dataset) if attach_velocity else None
        member_id = os.path.splitext(file_name)[0]
        self._prepare_subdirectory(os.path.join(self.target_dir, var_name), member_id)
        for i, t in enumerate(variable.coords[Axis.TIME.value]):
            variable_snapshot, velocity_snapshot = self._select_snapshot(t, variable, velocity)
            if drop_nan_levels:
                variable_snapshot, velocity_snapshot = self._drop_nan_levels(variable_snapshot, velocity_snapshot)
            variable_snapshot, velocity_snapshot = self._transpose_dimensions(variable_snapshot, velocity_snapshot)
            variable_data, velocity_data = self._extract_data(variable_snapshot, velocity_snapshot)
            print(f'[INFO] '
                  f'Data shape: {variable_data.shape}, ' +
                  f'Dtype: {variable_data.dtype}, ' +
                  f'Min: {np.min(variable_data)}, ' +
                  f'Max: {np.max(variable_data)}, ' +
                  f'Mean: {np.mean(variable_data)}, ' +
                  f'Std: {np.std(variable_data)}'
            )
            variable_data = self._normalize_data(variable_data, clamp_min, clamp_max)
            self._store_snapshot(var_name, member_id, i, variable_data, velocity_data)

    @staticmethod
    def _select_variable(dataset, var_name):
        return dataset[var_name]

    def _select_velocity(self, dataset):
        return dataset[self._velocity_names()]

    def _velocity_names(self):
        return [self.VELOCITY_NAMES[dimension] for dimension in self.DIMENSION_ORDER]

    def _select_snapshot(self, t, variable, velocity):
        velocity_snapshot = velocity.sel({Axis.TIME.value: t}) if velocity is not None else None
        return variable.sel({Axis.TIME.value: t}), velocity_snapshot

    def _drop_nan_levels(self, variable_snapshot, velocity_snapshot):
        valid = self._find_valid_levels(variable_snapshot)
        if velocity_snapshot is not None:
            for vel_name in self._velocity_names():
                valid = np.logical_and(valid, self._find_valid_levels(velocity_snapshot[vel_name]))
            velocity_snapshot = velocity_snapshot.isel({Axis.LEVEL.value: valid})
        variable_snapshot = variable_snapshot.isel({Axis.LEVEL.value: valid})
        return variable_snapshot, velocity_snapshot

    @staticmethod
    def _find_valid_levels(snapshot):
        # find all levels within and above which no nans occur
        nan_counts = np.isnan(snapshot).sum(dim=Axis.LATITUDE.value).sum(dim=Axis.LONGITUDE.value)
        return np.flip(np.flip(nan_counts).cumsum(dim=Axis.LEVEL.value)) == 0

    def _dimension_order(self):
        return [dimension.value for dimension in self.DIMENSION_ORDER]

    def _transpose_dimensions(self, variable_snapshot, velocity_snapshot):
        if velocity_snapshot is not None:
            velocity_snapshot = velocity_snapshot.transpose(*self._dimension_order())
        return variable_snapshot.transpose(*self._dimension_order()), velocity_snapshot

    def _extract_data(self, variable_snapshot, velocity_snapshot):
        if velocity_snapshot is not None:
            velocity_snapshot = np.stack([
                velocity_snapshot[vel_name].values
                for vel_name in self._velocity_names()
            ], axis=0)
        return variable_snapshot.values[None, ...], velocity_snapshot

    def _store_snapshot(self, var_name, member_id, i, variable_data, velocity_data):
        full_data = variable_data
        if velocity_data is not None:
            full_data = np.concatenate([full_data, velocity_data], axis=0)
        storage_path = self._get_storage_path(var_name, member_id, i)
        print(f'[INFO] Made it to the conversion step! Data shape: {full_data.shape}, storage path: {storage_path}')
        vol = pyrenderer.Volume()
        vol.worldX = 10
        vol.worldY = 10
        vol.worldZ = 1
        vol.add_feature_from_tensor("vel+density", torch.from_numpy(full_data))
        vol.save(os.path.abspath(storage_path), compression=0)

    def _get_storage_path(self, var_name, member_id, i):
        return os.path.join(self.target_dir, var_name, member_id, 't{:02d}.cvol'.format(i))

    def _normalize_data(self, data, clamp_min, clamp_max):
        if clamp_min is not None:
            data = np.clip(data, a_min=clamp_min)
            norm_min = clamp_min
        else:
            norm_min = np.min(data)
        if clamp_max is not None:
            data = np.clip(data, a_max=clamp_max)
            norm_max = clamp_max
        else:
            norm_max = np.max(data)
        data = (data - norm_min) / (norm_max - norm_min)
        return data


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('source_dir', type=str, help="base directory of 1000 member simulation ensemble")
    parser.add_argument('target_dir', type=str, help="output directory for storage of converted files")
    parser.add_argument('--overwrite', '-o', dest='overwrite', action='store_true', help="overwrite files in target directory if necessary")
    parser.add_argument('--create', '-c', dest='create', action='store_true', help="create target directory if necessary")
    parser.add_argument('--drop-nan-levels', '-d', dest='drop_nan_levels', action='store_true', help='drop nan levels upon conversion')
    parser.add_argument('--attach-velocity', '-v', dest='attach_velocity', action='store_true', help='attach velocity to ouput file upon conversion')
    parser.add_argument('--split-members', '-s', dest='split_members', action='store_true', help='attach velocity to ouput file upon conversion')
    parser.set_defaults(overwrite=False, create=False, drop_nan_levels=False, normalize=False, attach_velocity=False, split_members=False)
    parser.add_argument('--variable', type=str, help='variable to convert', choices=['tk', 'rh', 'qv', 'z', 'dbz', 'qhydro'], default=None)
    parser.add_argument('--clamp-min', type=float, help='minimum for data clamping', default=None)
    parser.add_argument('--clamp-max', type=float, help='maximum for data clamping', default=None)
    parser.add_argument('--max-member', type=int, help='maximum member index', default=None)
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    source = os.path.abspath(args.source_dir)
    target = os.path.abspath(args.target_dir)
    print('[INFO] Converting 1000 member simulation ensemble')
    print(f'[INFO] Source directory: {source}')
    print(f'[INFO] Target directory: {target}')
    assert args.variable is not None
    converter = EnsembleConverter(source, target)
    converter.read_source_directory()
    converter.validate_create_target_directory(overwrite=args.overwrite, create_target_dir=args.create)
    converter.convert_variable(
        args.variable,
        drop_nan_levels=args.drop_nan_levels, attach_velocity=args.attach_velocity,
        clamp_min=args.clamp_min, clamp_max=args.clamp_max,
        max_member=args.max_member,
        split_to_member_files=args.split_members
    )


if __name__ == '__main__':
    main()


