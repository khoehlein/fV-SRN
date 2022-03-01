from pyrenderer import IVolumeInterpolation
from torch import Tensor

from volnet.modules.datasets.evaluation.field_evaluator import IFieldEvaluator


class VolumeEvaluator(IFieldEvaluator):

    def __init__(self, interpolator: IVolumeInterpolation, device):
        super(VolumeEvaluator, self).__init__(3, 1, device)
        self.interpolator = interpolator
        self._default_volume = interpolator.volume()
        self._default_mipmap_level = interpolator.mipmap_level()

    def forward(self, positions: Tensor) -> Tensor:
        return self.interpolator.evaluate(positions)

    def set_source(self, volume_data, mipmap_level=None):
        if mipmap_level is None:
            mipmap_level = self._default_mipmap_level
        self.interpolator.setSource(volume_data, mipmap_level)
        return self

    def restore_defaults(self):
        return self.set_source(self._default_volume, mipmap_level=self._default_mipmap_level)
