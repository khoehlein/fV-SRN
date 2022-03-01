from typing import Union

import torch

from volnet.modules.datasets.evaluation.field_evaluator import IFieldEvaluator
from volnet.modules.datasets.resampling.coordinate_box import CoordinateBox, UnitCube


class IImportanceSampler(object):

    def __init__(self, dimension: int, root_box: Union[CoordinateBox, None], device):
        if root_box is None:
            root_box = UnitCube(dimension)
        else:
            assert root_box.dimension == dimension
        self.root_box = root_box
        if device is None:
            device = torch.device('cpu')
        self.device = device

    def generate_samples(self, num_samples: int, evaluator: IFieldEvaluator):
        raise NotImplementedError()
