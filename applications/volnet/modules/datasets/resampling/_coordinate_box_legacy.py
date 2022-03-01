from typing import Union

import numpy as np
import torch
from torch import Tensor


class CoordinateBox(object):

    def __init__(self, dimension: int, bounds: np.ndarray):
        assert dimension > 0, '[ERROR] Dimensions must be positive.'
        self.dimension = dimension
        assert bounds.shape == (2, dimension), \
            f'[ERROR] Expecting bounds array to have shape (2, {dimension}). Got {tuple(bounds.shape)} instead'
        assert np.all(bounds[0] < bounds[1]), \
            f'[ERROR] Expecting lower bounds to be strictly smaller than upperbounds. Got lower bounds {tuple(bounds[0].tolist())} and upper bounds {tuple(bounds[1].tolist())} instead.'
        self.bounds = bounds

    def center(self, keepdims=True):
        return np.mean(self.bounds, axis=0, keepdims=keepdims)

    def lower_bounds(self, keepdims=True):
        bounds = self.bounds[0]
        if keepdims:
            return bounds[None, ...]
        return bounds

    def upper_bounds(self, keepdims=True):
        bounds = self.bounds[1]
        if keepdims:
            return bounds[None, ...]
        return bounds

    def size(self, keepdims=True):
        box_size = np.diff(self.bounds, axis=0)
        if not keepdims:
            box_size = box_size[0]
        return box_size

    def volume(self):
        return np.prod(self.size(keepdims=False))

    def max_aspect(self):
        size = self.size(keepdims=False)
        return np.max(size) / np.min(size)

    def contains(self, coordinates: np.ndarray, include_boundary=True):
        assert coordinates.shape[-1] == self.dimension, \
            f'[ERROR] Expecting last axis of coordinate array to have dimension {self.dimension}. Got {coordinates.shape[-1]} instead.'
        leading_axes = len(coordinates.shape) - 1
        bounds_shape = [1 for _ in range(leading_axes)]
        lower_bounds = self.lower_bounds(keepdims=False).reshape(*bounds_shape, self.dimension)
        upper_bounds = self.upper_bounds(keepdims=False).reshape(*bounds_shape, self.dimension)
        if include_boundary:
            satisfies_lower_bound = lower_bounds <= coordinates
            satisfies_upper_bound = coordinates <= upper_bounds
        else:
            satisfies_lower_bound = lower_bounds < coordinates
            satisfies_upper_bound = coordinates < upper_bounds
        return np.logical_and(np.all(satisfies_lower_bound, axis=-1), np.all(satisfies_upper_bound, axis=-1))

    def rescale(self, coordinates: Union[np.ndarray, Tensor]):
        assert coordinates.shape[-1] == self.dimension, \
            f'[ERROR] Expecting last axis of coordinate array to have dimension {self.dimension}. Got {coordinates.shape[-1]} instead.'
        leading_axes = len(coordinates.shape) - 1
        bounds_shape = [1 for _ in range(leading_axes)]
        lower_bounds = self.lower_bounds(keepdims=False).reshape(*bounds_shape, self.dimension)
        upper_bounds = self.upper_bounds(keepdims=False).reshape(*bounds_shape, self.dimension)
        delta = upper_bounds - lower_bounds
        if type(coordinates).__module__ != 'numpy':
            delta = torch.from_numpy(delta).to(coordinates.device)
            lower_bounds = torch.from_numpy(lower_bounds).to(coordinates.device)
        return coordinates * delta + lower_bounds


class UnitCube(CoordinateBox):

    def __init__(self, dimension: int):
        assert dimension > 0,'[ERROR] Dimensions must be positive.'
        bounds = np.array([[0., 1.]] * dimension).T
        super(UnitCube, self).__init__(dimension, bounds)
