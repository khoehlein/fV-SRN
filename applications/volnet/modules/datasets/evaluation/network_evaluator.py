from torch import Tensor

from volnet.modules.datasets.evaluation.field_evaluator import IFieldEvaluator
from volnet.network import SceneRepresentationNetwork


class NetworkEvaluator(IFieldEvaluator):

    def __init__(self, network: SceneRepresentationNetwork):
        super(NetworkEvaluator, self).__init__(network.base_input_channels(), network.output_channels(), network.device)
        self.network = network

    def forward(self, positions: Tensor) -> Tensor:
        raise NotImplementedError()
