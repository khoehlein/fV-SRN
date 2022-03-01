import argparse
from typing import Tuple, Dict, Any, Optional

import torch
from torch import Tensor
import torch.nn.functional as F

from volnet.modules.datasets.evaluation import IFieldEvaluator
from volnet.modules.datasets.resampling import IImportanceSampler, CoordinateBox
from volnet.modules.datasets.sampling import ISampler, RandomUniformSampler,StratifiedGridSampler


class FixedGridImportanceSampler(IImportanceSampler):

    @staticmethod
    def init_parser(parser: argparse.ArgumentParser):
        group = parser.add_argument_group('FixedGridImportanceSampler')
        group.add_argument('--importance-sampler:grid-size', type=int, default=64, help="""
        grid resolution for loss-importance-based resampling of the dataset
        """)
        group.add_argument('--importance-sampler:num-samples-per-voxel', type=int, default=8, help="""
        number of samples per voxel for loss evaluation in importance-based dataset resampling
        """)
        group.add_argument('--importance-sampler:min-density', type=float, default=0.01, help="""
        minimum density for sampling per voxel 
        """)
        group.add_argument('--importance-sampler:batch-size', type=int, default=None, help="""
        batch size during importance sampling
        """)

    @classmethod
    def from_dict(cls, args: Dict[str, Any], sampler: Optional[ISampler] = None):
        prefix = 'importance_sampler:'
        return cls(
            tuple([args[prefix + 'grid_size']] * 3), sampler=sampler,
            num_samples_per_voxel=args[prefix + 'num_samples_per_voxel'], min_density=args[prefix + 'min_density'],
            batch_size=args[prefix + 'batch_size']
        )

    def __init__(
            self,
            grid_size: Tuple[int, ...], sampler: Optional[ISampler] = None,
            num_samples_per_voxel=8, min_density=0.01,
            batch_size=None, root_box: Optional[CoordinateBox] = None,
            verbose=False, device=None
    ):
        super(FixedGridImportanceSampler, self).__init__(len(grid_size), root_box, device)
        if sampler is None:
            sampler = RandomUniformSampler(len(grid_size), self.device)
        else:
            assert sampler.device == self.device
        self.sampler = StratifiedGridSampler(sampler, grid_size, handle_remainder=False)
        self.root_box = root_box
        self.num_samples_per_voxel = num_samples_per_voxel
        self.min_density = min_density
        self.batch_size = batch_size
        self._grid_numel = self.sampler.grid_numel()
        self.verbose = verbose

    def generate_samples(self, num_samples: int, evaluator: IFieldEvaluator):
        value_grid = self._build_value_grid(evaluator)
        importance_grid = torch.maximum(value_grid / torch.max(value_grid).item(), torch.tensor([self.min_density], device=value_grid.device))
        positions = self._generate_importance_samples(num_samples, importance_grid)
        return positions

    def _generate_importance_samples(self, num_samples: int, importance_grid: Tensor):
        sampler = RandomUniformSampler(self.sampler.dimension, self.sampler.device)
        batch_size = self.batch_size if self.batch_size is not None else num_samples
        all_samples = []
        total_samples = 0
        total_accepted_samples = 0
        current_sample_count = 0
        while current_sample_count < num_samples:
            samples = sampler.generate_samples(batch_size)
            threshold = F.grid_sample(
                importance_grid[None, None, ...],
                2. * samples.view(*[1 for _ in importance_grid.shape], *samples.shape) - 1.,
                align_corners=True, mode='bilinear'
            )
            threshold = threshold.view(-1)
            accepted = (torch.rand(batch_size, device=sampler.device) < threshold)
            samples = samples[accepted]
            num_accepted = len(samples)
            total_samples = total_samples + batch_size
            if num_accepted > 0:
                total_accepted_samples = total_accepted_samples + num_accepted
                if current_sample_count + num_accepted > num_samples:
                    samples = samples[:(num_samples - current_sample_count)]
                all_samples.append(samples)
                current_sample_count = current_sample_count + len(samples)
        if self.verbose:
            self._print_statistics(total_samples, total_accepted_samples)
        samples = torch.cat(all_samples, dim=0)
        return self.root_box.rescale(samples)

    def _print_statistics(self, total_samples, total_accepted_samples):
        print(
            '[INFO] Finished importance sampling after {tot} total samples. Acceptance rate was {frac:.4f}'.format(
                tot=total_samples, frac=total_accepted_samples / total_samples
            )
        )

    def _build_value_grid(self, evaluator: IFieldEvaluator):
        assert evaluator.device == self.sampler.device
        grid_size = self.sampler.grid_size
        loss_grid = torch.zeros(*grid_size, dtype=torch.float32, device=evaluator.device)
        for j in range(self.num_samples_per_voxel):
            positions = self.sampler.generate_samples(self._grid_numel)
            values = evaluator.evaluate(positions)
            loss_grid += values.view(*grid_size)
        return loss_grid


def _test_importance_sampler():
    import matplotlib.pyplot as plt

    class Evaluator(IFieldEvaluator):

        def __init__(self, dimension, device=None):
            super(Evaluator, self).__init__(dimension, 1, device)
            self.direction = 4 * torch.tensor([1] * dimension, device=device)[None, :]# torch.randn(1, dimension, device=device)
            self.offset = torch.tensor([0.5] * dimension, device=device)[None, :] # torch.randn(1, dimension, device=device)

        def forward(self, positions: Tensor) -> Tensor:
            return torch.exp(torch.sum(self.direction * (positions - self.offset), dim=-1))

    dimension = 2
    device = torch.device('cuda:0')
    eval = Evaluator(dimension, device)
    sampler = FixedGridImportanceSampler((16, 16), verbose=True, device=device)
    positions = sampler.generate_samples(100000, eval)
    values = eval.evaluate(positions)[:, 0]
    positions = positions.data.cpu().numpy()
    values = values.data.cpu().numpy()
    plt.scatter(positions[:, 0], positions[:, 1], c=values, alpha=0.05)
    plt.colorbar()
    plt.tight_layout()
    plt.show()
    plt.close()

    print('Finished')


if __name__ == '__main__':
    _test_importance_sampler()
