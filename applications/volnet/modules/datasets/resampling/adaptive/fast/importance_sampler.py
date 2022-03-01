import argparse
from typing import Optional

from volnet.modules.datasets.evaluation import IFieldEvaluator
from volnet.modules.datasets.resampling.coordinate_box import CoordinateBox
from volnet.modules.datasets.resampling.adaptive.fast.fast_density_tree_sampler import FastDensityTreeSampler
from volnet.modules.datasets.resampling.adaptive.fast.fast_density_tree import FastDensityTree
from volnet.modules.datasets.resampling.adaptive.fast.fast_statistical_tests import FastKolmogorovSmirnovTestNd, \
    FastWhiteHomoscedasticityTest
from volnet.modules.datasets.resampling.interface import IImportanceSampler
from volnet.modules.datasets.sampling.interface import ISampler


class DensityTreeImportanceSampler(IImportanceSampler):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('DensityTreeImportanceSampler')
        group.add_argument('--importance-sampler:min-depth', type=int, default=4, help="""
        minimum tree depth for adaptive loss grid
        """)
        group.add_argument('--importance-sampler:max-depth', type=int, default=12, help="""
        maximum tree depth for adaptive loss grid
        """)
        group.add_argument('--importance-sampler:samples-per-node', type=int, default=128, help="""
        number of samples per node for loss tree refinement
        """)
        group.add_argument('--importance-sampler:alpha', type=float, default=0.05, help="""
        significance threshold for splitting decision
        """)
        group.add_argument('--importance-sampler:batch-size', type=int, default=None, help="""
        batch size for loss evaluation during importance sampling (Default: Dataset batch size)
        """)
        group.add_argument('--importance-sampler:min-density', type=float, default=0.01, help="""
        minimum probability density for sampling per grid box 
        """)
        group.add_argument('--importance-sampler:max-ratio', type=float, default=10, help="""
        maximum ratio of probability densities during node splitting 
        """)
        group.add_argument('--importance-sampler:seed', type=int, default=42, help="""
        seed for importance sampling random number generator
        """)
        group.add_argument('--importance-sampler:loss-mode', type=str, default='l1', choices=['l1', 'l2', 'mse'],
                           help="""loss type to compute for loss-importance-based resampling of the dataset (Default: batch size of dataset)""")

    def __init__(
            self,
            sampler: ISampler, batch_size=None,
            min_depth=4, max_depth=8, num_samples_per_node=128, min_density=0.01, max_ratio=10,
            alpha=0.05, root_box: Optional[CoordinateBox] = None, device=None
    ):
        super(DensityTreeImportanceSampler, self).__init__(sampler.dimension, root_box, device)
        assert sampler.device == self.device
        self.sampler = sampler
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_samples_per_node = num_samples_per_node
        self.min_density = min_density
        self.max_ratio = max_ratio
        self.alpha = alpha
        assert batch_size is not None
        self.batch_size = int(batch_size)

    def generate_samples(self, num_samples: int, evaluator: IFieldEvaluator):
        difference_test = FastKolmogorovSmirnovTestNd(alpha=self.alpha)
        homoscedasticity_test = FastWhiteHomoscedasticityTest(alpha=self.alpha)
        tree = FastDensityTree.from_scalar_field(
            self.root_box, self.sampler, evaluator, difference_test, homoscedasticity_test=homoscedasticity_test,
            min_depth=self.min_depth,max_depth=self.max_depth, num_samples_per_node=self.num_samples_per_node,
            store_sample_summary=True, num_samples_per_batch=self.batch_size,
            device=self.device
        )
        sampler = FastDensityTreeSampler(
            self.sampler, tree,
            min_density=self.min_density, max_ratio=self.max_ratio
        )
        samples = sampler.generate_samples(num_samples)
        if samples.device != self.device:
            samples = samples.to(self.device)
        return samples
