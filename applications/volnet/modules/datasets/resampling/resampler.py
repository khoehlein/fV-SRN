import argparse
from typing import Optional, Dict, Any, Union

from torch.nn import functional as F

from volnet.modules.datasets.evaluation import VolumeEvaluator, LossEvaluator
from volnet.modules.datasets.sampling.position_sampler import PositionSampler
from volnet.modules.datasets.resampling import IImportanceSampler
from volnet.modules.datasets.resampling.adaptive import DensityTreeImportanceSampler
from volnet.modules.datasets.resampling.fixed_grid import FixedGridImportanceSampler
from volnet.modules.datasets.resampling.inn.flow.importance_sampler import WarpingNetworkImportanceSampler
from volnet.modules.datasets.world_dataset import WorldSpaceDensityData
from volnet.network import SceneRepresentationNetwork


class DatasetResampler(object):

    RANDOM_KEY = 'random'
    IMPORTANCE_KEY = 'importance'

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group= parser.add_argument_group('DatasetResampler')
        prefix = '--dataset-resampling:'
        group.add_argument(
            prefix + 'method', type=str, default='random',
            help="""method for resampling dataset Choices: random, importance:grid, importance:tree, importance:warp"""
        )
        group.add_argument(
            prefix + 'loss', type=str, default='l1', choices=['l1', 'l2', 'mse'],
            help="""loss type to compute for loss-importance-based resampling of the dataset"""
        )
        group.add_argument(
            prefix + 'frequency', type=int, default=None,
            help="""number of epochs after which to resample data (periodically) (Default: None -> No resampling)"""
        )
        FixedGridImportanceSampler.init_parser(parser)
        DensityTreeImportanceSampler.init_parser(parser)
        WarpingNetworkImportanceSampler.init_parser(parser)

    @classmethod
    def from_dict(cls, args: Dict[str, Any], device=None):
        prefix = 'dataset_resampling:'

        def get_arg(key: str):
            return args[prefix + key]

        method = get_arg('method')
        if method is None:
            method = 'random'
        method, *specs = method.split(':')
        if method == 'random':
            if len(specs) > 0:
                algo = specs[0]
            else:
                algo = 'random'
            sampler = PositionSampler(method=algo, dimension=3)
        elif method == 'importance':
            if len(specs) > 0:
                algo = specs[0]
            else:
                algo = 'grid'
            if algo == 'grid':
                sampler = FixedGridImportanceSampler.from_dict(args, dimension=3, device=device)
            elif algo == 'tree':
                sampler = DensityTreeImportanceSampler.from_dict(args, dimension=3, device=device)
            elif algo == 'warp-net':
                sampler = WarpingNetworkImportanceSampler.from_dict(args, dimension=3, device=device)
            else:
                raise Exception('[ERROR] Encountered unknown importance sampling algorithm')
        else:
            raise Exception('[ERROR] Encountered unknown sampling ')
        return cls(sampler, frequency=get_arg('frequency'), loss=get_arg('loss'))

    def __init__(
            self,
            sampler: Union[PositionSampler, IImportanceSampler],
            frequency: Optional[int] = None,
            loss: Optional[str] = 'l1',
    ):
        self.sampler = sampler
        self.frequency = frequency
        self.loss_mode = loss
        self.device = sampler.device

    def uses_importance_sampling(self):
        return isinstance(self.sampler, IImportanceSampler)

    def requires_action(self, epoch):
        if self.frequency is None:
            return False
        if epoch % self.frequency == 0:
            return True
        return False

    def resample_dataset(
            self,
            dataset: WorldSpaceDensityData, volume_evaluator: VolumeEvaluator,
            network: Optional[SceneRepresentationNetwork] = None,
    ):
        if self.uses_importance_sampling():
            assert network is not None, '[ERROR] Cannot do remportance sampling without scene representation network.'
            loss_evaluator = LossEvaluator(self._get_loss_function(), volume=volume_evaluator)
            dataset.sample_data_with_loss_importance(network, volume_evaluator, loss_evaluator, self.sampler)
        else:
            dataset.sample_data(volume_evaluator, self.sampler)
        return dataset

    def _get_loss_function(self):
        return {
            'l1': F.l1_loss, 'l2': F.mse_loss, 'mse': F.mse_loss
        }[self.loss_mode]

