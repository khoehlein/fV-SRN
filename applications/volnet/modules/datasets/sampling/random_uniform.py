import torch

from volnet.modules.datasets.sampling.interface import ISampler


class RandomUniformSampler(ISampler):

    def __init__(self, dimension, device=None, dtype=None):
        super(RandomUniformSampler, self).__init__(dimension, device, dtype)

    def generate_samples(self, num_samples: int):
        return torch.rand(num_samples, self.dimension, device=self.device, dtype=self.dtype)
