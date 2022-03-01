import torch


def get_normalized_positions(num_nodes, clamp=False):
    num_segments = 2 * (num_nodes - 1)
    positions = torch.arange(1, num_segments + 2, 2) / num_segments
    if clamp:
        positions[-1] = 1.
    return positions
