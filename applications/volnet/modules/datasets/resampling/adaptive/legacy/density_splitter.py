from typing import Optional, List, Union

import numpy as np

from volnet.modules.datasets.resampling.adaptive.legacy.data_statistics import SampleSummary, MergerSummary


class DensitySplitter(object):

    def __init__(self, min_density: Optional[float] = None, max_ratio: Optional[float] = None, seed: Optional[int] = None):
        self._min_density = min_density
        if max_ratio is not None:
            max_ratio = np.exp(np.abs(np.log(max_ratio)))
        self._max_ratio = max_ratio
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def min_density(self):
        return self._min_density

    def max_ratio(self):
        return self._max_ratio

    def distribute_density(self, density: float, data: List[Union[SampleSummary, MergerSummary]], index: Optional[List[np.ndarray]] = None):
        data = np.array([node_data.mean() for node_data in data])
        p = self._apply_constraints(data / np.sum(data), density)
        densities = 2. * density * p
        if index is None:
            return densities
        child_indices = self._distribute_indices(index, p)
        return densities, child_indices

    def _apply_constraints(self, p_raw, density):
        constraints = []
        if self._min_density is not None:
            constraints.append(self._min_density / (2. * density))
        if self._max_ratio is not None:
            constraints.append(1. / (1 + self._max_ratio))
        p_min = np.max(constraints)
        idx_lower = np.argmin(p_raw)
        p = p_raw
        if p_raw[idx_lower] < p_min:
            p[idx_lower] = p_min
            p[1 - idx_lower] = 1. - p_min
        return p

    def _distribute_indices(self, index, p):
        num_samples = len(index)
        classification = self.rng.random(size=num_samples) < p[0]
        return index[classification], index[~classification]
