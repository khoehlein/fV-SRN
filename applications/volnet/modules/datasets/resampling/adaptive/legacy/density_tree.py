import time

import numpy as np
from matplotlib import pyplot as plt, colors, cm, patches
from typing import List, Dict
from itertools import chain


from volnet.modules.datasets.resampling.coordinate_box import CoordinateBox, UnitCube
from volnet.modules.datasets.resampling.adaptive.legacy.data_statistics import SampleSummary, MergerSummary
from volnet.modules.datasets.resampling.adaptive.legacy.density_splitter import DensitySplitter
from volnet.modules.datasets.resampling.adaptive.legacy.statistical_tests import MyWhiteHomoscedasticityTest


class DensityTree(object):

    def __init__(
            self,
            root_node: 'DensityTreeNode',
            parent_map: Dict['DensityTreeNode', 'DensityTreeNode'],
            children_map: Dict['DensityTreeNode', List['DensityTreeNode']],
            nodes_by_depth: Dict[int, List['DensityTreeNode']],
            leaf_nodes: List['DensityTreeNode']
    ):
        self.root_node = root_node
        self._children_map = children_map
        self._parent_map = parent_map
        self._nodes_by_depth = nodes_by_depth
        self._leaf_nodes = leaf_nodes

    class _SingleNodeEvaluationProcess(object):

        def __init__(
                self,
                node, depth, sampler, field_evaluator, difference_test,
                homoscedasticity_test=None,
                min_depth=0, max_depth=10,
                num_samples_per_node=64, store_sample_summary=True

        ):
            self.node = node
            self.depth = depth
            self.sampler = sampler
            self.field_evaluator = field_evaluator
            self.difference_test = difference_test
            self.homoscedasticity_test = homoscedasticity_test
            self.min_depth = min_depth
            self.max_depth = max_depth
            self.num_samples_per_node = num_samples_per_node
            self.store_sample_summary = store_sample_summary

        def run(self):
            positions = self.sampler.generate_samples(self.num_samples_per_node)
            positions = self.node.box.rescale(positions)
            values = self.field_evaluator.evaluate(positions)
            if self.store_sample_summary:
                self.node.store_sample_summary(values)
            children = None
            if self.depth < self.max_depth:
                split_dimension = None
                classification = positions < self.node.box.center()
                difference_result = self.difference_test.compute(values, classification)
                if difference_result.reject() or (self.min_depth is not None and self.depth <= self.min_depth):
                    split_dimension = difference_result.best_split()
                if split_dimension is None and self.homoscedasticity_test is not None:
                    homoscedasticity_result = self.homoscedasticity_test.compute(values, positions)
                    if homoscedasticity_result.reject():
                        split_dimension = difference_result.best_split()
                if split_dimension is not None:
                    children = self.node.split_along_dimension(split_dimension)
            return DensityTree._NodeEvaluationResult(self.node, children)

    class _NodeEvaluationResult(object):

        def __init__(self, parent: 'DensityTreeNode', children: List['DensityTreeNode']):
            self.parent = parent
            self.children = children

        def get_parent(self):
            return self.parent

        def get_children(self):
            return self.children

    class _NodeBatchEvaluationProcess(object):

        def __init__(
                self,
                nodes: List['DensityTreeNode'], depth: int, sampler, field_evaluator, difference_test,
                homoscedasticity_test=None,
                min_depth=0, max_depth=10,
                num_samples_per_node=64, store_sample_summary=True
        ):
            self.nodes = nodes
            self.depth = depth
            self.sampler = sampler
            self.field_evaluator = field_evaluator
            self.difference_test = difference_test
            self.homoscedasticity_test = homoscedasticity_test
            self.min_depth = min_depth
            self.max_depth = max_depth
            self.num_samples_per_node = num_samples_per_node
            self.store_sample_summary = store_sample_summary

        def num_nodes(self):
            return len(self.nodes)

        def num_samples_total(self):
            return self.num_nodes() * self.num_samples_per_node

        def run(self):
            raw_samples = self.sampler.generate_samples(self.num_samples_total())
            positions = self._get_positions_from_samples(raw_samples)
            values = self._evaluate_field_at_positions(positions)
            if self.store_sample_summary:
                [node.store_sample_summary(node_values)for node, node_values in zip(self.nodes, values)]
            split_dimension = np.full((self.num_nodes(),), -1)
            if self.depth < self.max_depth:
                classification = self._classify_positions(positions)
                difference_result = self.difference_test.compute(values, classification)
                best_split = difference_result.best_split()
                difference_reject = difference_result.reject()
                if self.min_depth is not None and self.depth <= self.min_depth:
                    split_dimension = difference_result.best_split()
                elif np.any(difference_reject):
                    split_dimension[difference_reject] = best_split[difference_reject]
                difference_no_reject = ~ difference_reject
                if np.any(difference_no_reject) and self.homoscedasticity_test is not None:
                    homoscedasticity_result = self.homoscedasticity_test.compute(values[difference_no_reject], positions[difference_no_reject])
                    homoscedasticity_reject = homoscedasticity_result.reject()
                    if np.any(homoscedasticity_reject):
                        secondary_reject = np.full_like(difference_no_reject, False)
                        secondary_reject[difference_no_reject] = homoscedasticity_reject
                        split_dimension[secondary_reject] = best_split[secondary_reject]
            results = [
                DensityTree._NodeEvaluationResult(
                    node, node.split_along_dimension(dimension) if dimension >= 0 else None
                )
                for node, dimension in zip(self.nodes, split_dimension)
            ]
            return results

        def _get_positions_from_samples(self, samples:np.ndarray):
            samples = np.reshape(samples, (self.num_nodes(), self.num_samples_per_node, -1))
            bounds = np.stack([node.box.bounds for node in self.nodes], axis=0)
            scales = np.diff(bounds, axis=1)
            offsets = bounds[:, 0, :][:, None, :]
            positions = scales * samples + offsets
            return positions

        def _evaluate_field_at_positions(self, positions: np.ndarray):
            shape = positions.shape
            positions = np.reshape(positions, (-1, shape[-1]))
            values = self.field_evaluator.evaluate(positions)
            values = np.reshape(values, shape[:-1])
            return values

        def _classify_positions(self, positions: np.ndarray):
            centers = np.stack([node.box.center(keepdim=True) for node in self.nodes], axis=0)
            return positions < centers


    @staticmethod
    def _run_process(p: 'DensityTree._SingleNodeEvaluationProcess'):
        return p.run()

    @classmethod
    def from_scalar_field(
            cls, root_box: CoordinateBox,
            sampler, field_evaluator, difference_test,
            homoscedasticity_test=None,
            min_depth=0, max_depth=16,
            num_samples_per_node=64, store_sample_summary=True,
            multi_node_batch_evaluation=False, num_samples_per_batch=64**3
    ):
        root_node = DensityTreeNode(root_box, depth=0)
        parent_map = {}
        children_map = {}
        nodes_by_depth = {0: [root_node]}
        leaf_nodes = []

        current_depth = 0

        while current_depth in nodes_by_depth:
            if multi_node_batch_evaluation:
                num_nodes_per_batch = num_samples_per_batch // num_samples_per_node
                all_nodes = nodes_by_depth[current_depth]
                batches = [all_nodes[i:i+num_nodes_per_batch] for i in range(0, len(all_nodes), num_nodes_per_batch)]
                results = []
                for batch in batches:
                    process = DensityTree._NodeBatchEvaluationProcess(
                        batch, current_depth,
                        sampler, field_evaluator, difference_test,
                        homoscedasticity_test=homoscedasticity_test,
                        min_depth=min_depth, max_depth=max_depth,
                        num_samples_per_node=num_samples_per_node, store_sample_summary=store_sample_summary
                    )
                    results += process.run()
            else:
                processes = [
                    DensityTree._SingleNodeEvaluationProcess(
                        current_node, current_depth,
                        sampler, field_evaluator, difference_test,
                        homoscedasticity_test=homoscedasticity_test,
                        min_depth=min_depth, max_depth=max_depth,
                        num_samples_per_node=num_samples_per_node, store_sample_summary=store_sample_summary
                    )
                    for current_node in nodes_by_depth[current_depth]
                ]
                results = [p.run() for p in processes]
            new_nodes = []
            for result in results:
                parent = result.get_parent()
                children = result.get_children()
                if children is not None:
                    parent_map.update({child_node: parent for child_node in children})
                    children_map[parent] = children
                    new_nodes.append(children)
                else:
                    leaf_nodes.append(parent)
            if len(new_nodes) > 0:
                new_nodes = list(chain.from_iterable(new_nodes))
                nodes_by_depth[current_depth + 1] = new_nodes
            current_depth = current_depth + 1
        return cls(root_node, parent_map, children_map, nodes_by_depth, leaf_nodes)

    def _get_children_of(self, node: 'DensityTreeNode'):
        return self._children_map[node]

    def _get_parent_of(self, node: 'DensityTreeNode'):
        return self._parent_map[node]

    def get_max_depth(self):
        return len(self._nodes_by_depth) - 1

    def query_sample_data(self):
        for depth in reversed(sorted(self._nodes_by_depth.keys())[:-1]):
            nodes = self._nodes_by_depth[depth]
            for current_node in nodes:
                try:
                    children = self._get_children_of(current_node)
                except KeyError:
                    continue
                else:
                    current_node.query_data_from(*children)
        return self

    def distribute_samples(self, raw_samples: np.ndarray, splitter: DensitySplitter):
        num_samples = len(raw_samples)
        final_samples = []
        queue = [(self.root_node, 1. / self.root_node.box.volume(), np.arange(num_samples))]
        while len(queue) > 0:
            current_node, current_density, current_index = queue.pop()
            if current_node.is_leaf_node() or current_density <= splitter.min_density():
                current_samples = raw_samples[current_index]
                current_samples = current_node.box.rescale(current_samples)
                final_samples.append(current_samples)
            else:
                children = self._get_children_of(current_node)
                data = [node.get_sample_data() for node in children]
                densities, indices = splitter.distribute_density(current_density, data, index=current_index)
                for child_node, child_density, child_index in zip(children, densities, indices):
                    if len(child_index) > 0:
                        queue.append((child_node, child_density, child_index))
        final_samples = np.concatenate(final_samples, axis=0)
        assert len(final_samples) == num_samples
        return final_samples

    def plot(self, ax, mode='depth', cmap='YlOrRd'):
        assert self.root_node.coordinate_dimension() == 2
        if mode == 'depth':
            vmax = self.get_max_depth()
            vmin = 0
            def get_value(node):
                return node.depth
        elif mode == 'value':
            assert self.root_node.carries_sample_data()
            data = self.root_node.get_sample_data()
            vmax = data.max()
            vmin = data.min()
            def get_value(node):
                return node.get_sample_data().mean()
        else:
            raise Exception()
        cmap = plt.get_cmap(cmap)
        cnorm = colors.Normalize(vmin,vmax)
        scalar_map = cm.ScalarMappable(norm=cnorm, cmap=cmap)
        for node in self._leaf_nodes:
            node.draw(ax, scalar_map, get_value)

    def leaf_depth_histogram(self):
        depth, counts = np.unique([node.depth for node in self._leaf_nodes], return_counts=True)
        order = np.argsort(depth)
        return depth[order], counts[order]

    def leaf_volume_histogram(self):
        volume, counts = np.unique([node.box.volume() for node in self._leaf_nodes], return_counts=True)
        order = np.argsort(volume)
        return volume[order], counts[order]

    def leaf_max_aspect_histogram(self):
        aspect, counts = np.unique([node.box.max_aspect() for node in self._leaf_nodes], return_counts=True)
        order = np.argsort(aspect)
        return aspect[order], counts[order]

    def num_leaves(self):
        return len(self._leaf_nodes)


class DensityTreeNode(object):

    DATA_KEY_SAMPLED = 'sampled'
    DATA_KEY_QUERIED = 'queried'

    def __init__(self, box: CoordinateBox, depth: int):
        self.box = box
        self.depth = depth
        self.data = {self.DATA_KEY_SAMPLED: None, self.DATA_KEY_QUERIED: None}
        self._split_dimension = None

    def is_leaf_node(self):
        return self._split_dimension is None

    def coordinate_dimension(self):
        return self.box.dimension

    def carries_sample_data(self):
        if self.is_leaf_node():
            return self.data[self.DATA_KEY_SAMPLED] is not None
        else:
            return self.data[self.DATA_KEY_QUERIED] is not None

    def store_sample_summary(self, sample: np.ndarray):
        loss_data = SampleSummary.from_sample(sample)
        existing_samples = self.data[self.DATA_KEY_SAMPLED]
        if existing_samples is not None:
            assert type(existing_samples) == SampleSummary
            loss_data = SampleSummary.from_sample_summaries(existing_samples, loss_data)
        assert loss_data is not None
        self.data.update({self.DATA_KEY_SAMPLED: loss_data})
        return self

    def split_along_dimension(self, dimension):
        if self.is_leaf_node():
            self._split_dimension = dimension
        else:
            raise Exception('[ERROR] DensityTreeNode can only be splitted once.')

        box = self.box

        def make_box(idx):
            bounds = np.concatenate([box.lower_bounds(), box.upper_bounds()], axis=0)
            bounds[idx, dimension] = box.center(keepdim=False)[dimension]
            return CoordinateBox(box.dimension, bounds)

        children = [DensityTreeNode(make_box(i), depth=(self.depth + 1)) for i in [1, 0]]

        return children

    def get_sample_data(self):
        if self.is_leaf_node():
            out_key = self.DATA_KEY_SAMPLED
        else:
            out_key = self.DATA_KEY_QUERIED
        out = self.data[out_key]
        assert out is not None, '[ERROR] Data was requested before it was queried'
        return out

    def query_data_from(self, *children: 'DensityTreeNode'):
        data = [child_node.get_sample_data() for child_node in children]
        weights = [child_node.box.volume() for child_node in children]
        out = MergerSummary.from_summaries(*chain.from_iterable(zip(data, weights)))
        self.data.update({self.DATA_KEY_QUERIED: out})
        return out

    def draw(self, ax, scalar_map, getter):
        value = getter(self)
        color = scalar_map.to_rgba(value)
        xy = self.box.lower_bounds(keepdim=False)
        width = self.box.upper_bounds(keepdim=False) - self.box.lower_bounds(keepdim=False)
        rectangle = patches.Rectangle(xy, *width.tolist(), facecolor=color, edgecolor='white')
        ax.add_patch(rectangle)


class Evaluator(object):

    def __init__(self, dimension, noise_amplitude=0.1, seed=None):
        self.dimension = dimension
        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.v_mu = self.rng.normal(size=(1, dimension))
        self.v_beta = self.rng.random(size=(1, dimension))
        self.noise_amplitude = noise_amplitude

    def evaluate(self, coordinates):
        mu = np.sum(self.v_mu * (coordinates - self.v_beta), axis=-1)
        # log_std = np.sum(self.v_log_std * coordinates, axis=-1)
        z = mu  # + np.random.randn(*mu.shape) * np.exp(log_std) / 20.
        return z ** 2 + self.noise_amplitude * self.rng.normal(size=z.shape)


class RandomUniformSampler(object):

    def __init__(self, dimension, seed=None):
        self.dimension = dimension
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def generate_samples(self, num_samples):
        return self.rng.random((num_samples, self.dimension))


def test(dimension=2, num_samples=256, min_depth=2, max_depth=10, alpha=0.05, num_processes=0):

    from volnet.modules.datasets.resampling.adaptive.legacy.statistical_tests import (
        WhiteHomoscedasticityTest,
        WelchTTestNd, KolmogorovSmirnovTestNd, MyKolmogorovSmirnovTestNd
    )

    box = UnitCube(dimension)
    sampler = RandomUniformSampler(dimension, seed=42)
    volume_evaluator = Evaluator(dimension, noise_amplitude=0.01, seed=42)
    homoscedasticity_test = WhiteHomoscedasticityTest(alpha=alpha, simplify_predictors=True)
    splitter = DensitySplitter(min_density=0.01, max_ratio=7, seed=42)

    print('Building trees')
    difference_test = WelchTTestNd(alpha=alpha)
    t0 = time.time()
    tree1 = DensityTree.from_scalar_field(
        box, sampler, volume_evaluator, difference_test,
        homoscedasticity_test=homoscedasticity_test,
        min_depth=min_depth, max_depth=max_depth,
        num_samples_per_node=num_samples,
        store_sample_summary=True, #num_processes=num_processes,
    )
    t1 = time.time()
    print(f'Completed 1. Time: {t1 - t0} sec')
    difference_test = KolmogorovSmirnovTestNd(alpha=alpha)
    t0 = time.time()
    tree2 = DensityTree.from_scalar_field(
        box, sampler, volume_evaluator, difference_test,
        homoscedasticity_test=homoscedasticity_test,
        min_depth=min_depth, max_depth=max_depth,
        num_samples_per_node=num_samples,
        store_sample_summary=True, #num_processes=num_processes,
    )
    t1 = time.time()
    print(f'Completed 2. Time: {t1 - t0} sec')
    difference_test = MyKolmogorovSmirnovTestNd(alpha=alpha)
    homoscedasticity_test = MyWhiteHomoscedasticityTest(alpha=alpha, simplify_predictors=True)
    t0 = time.time()
    tree3 = DensityTree.from_scalar_field(
        box, sampler, volume_evaluator, difference_test,
        homoscedasticity_test=homoscedasticity_test,
        min_depth=min_depth, max_depth=max_depth,
        num_samples_per_node=num_samples,
        store_sample_summary=True, #num_processes=num_processes,
        multi_node_batch_evaluation=True
    )

    t1 = time.time()
    print(f'Completed 3. Time: {t1 - t0} sec')

    print('Executing data query')

    tree1.query_sample_data()
    tree2.query_sample_data()
    tree3.query_sample_data()

    print('Plotting histograms')

    fig, ax = plt.subplots(3, 3, figsize=(15, 15))
    for i, tree in enumerate([tree1, tree2, tree3]):
        ax[i, 0].bar(*tree.leaf_depth_histogram())
        v, c = tree.leaf_volume_histogram()
        ax[i, 1].bar(np.log(v), c)
        ax[i, 2].bar(*tree.leaf_max_aspect_histogram())
    plt.tight_layout()
    plt.show()
    plt.close()

    if dimension == 2:
        print('Plotting tree structure')

        fig, ax = plt.subplots(3, 3, figsize=(15, 15))
        tree1.plot(ax[0, 0], 'value')
        tree1.plot(ax[0, 1], 'depth')
        tree2.plot(ax[1, 0], 'value')
        tree2.plot(ax[1, 1], 'depth')
        tree3.plot(ax[2, 0], 'value')
        tree3.plot(ax[2, 1], 'depth')

        print('Distributing samples')

        raw_samples = sampler.generate_samples(10000)
        samples1 = tree1.distribute_samples(raw_samples, splitter)
        samples2 = tree2.distribute_samples(raw_samples, splitter)
        samples3 = tree3.distribute_samples(raw_samples, splitter)

        print('Plotting samples')

        def draw_samples_to_axes(samples, ax):
            ax.scatter(samples[:, 0], samples[:, 1], alpha=0.1)

        draw_samples_to_axes(samples1, ax[0, 2])
        draw_samples_to_axes(samples2, ax[1, 2])
        draw_samples_to_axes(samples3, ax[2, 2])
        plt.tight_layout()
        plt.show()
        plt.close()

    print('Finished')


if __name__ == '__main__':
    test(
        dimension=2, num_samples=128,
        min_depth=2, max_depth=14,
        num_processes=0, alpha=0.005
    )