import numpy as np
import torch
import tqdm
from scipy.cluster.vq import kmeans, whiten, vq
from scipy.spatial.distance import cdist

from volnet.modules.datasets import WorldSpaceDensityData
from volnet.modules.networks.latent_features.marginal import EnsembleMultiGridFeatures


class InitialClustering(object):

    def __init__(self, max_iter=20, num_trials=10, seed=None):
        self.max_iter = max_iter
        self.num_trials = num_trials
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def initialize_mixing_features(self, dataset: WorldSpaceDensityData, latent_features: EnsembleMultiGridFeatures):
        num_clusters = latent_features.num_grids()
        print('[INFO] Clustering dataset')
        volumes = dataset.get_volumes().cpu().numpy()
        assert len(volumes) == 1, \
            '[ERROR] Length of volumes must be 1! InitialClustering does currently not support time-variate data!'
        current_weights = latent_features.mixing_features.data
        new_weights = self._compute_weights(volumes, num_clusters)[0]
        new_weights = torch.from_numpy(new_weights).to(device=current_weights.device, dtype=current_weights.dtype)
        assert new_weights.shape == current_weights.shape, \
            '[ERROR] Something went wrong with matching the shape of old and new mixing weights!'
        print('[INFO] Resetting mixing features')
        latent_features.mixing_features.data = new_weights
        return latent_features

    def _compute_weights(self, volumes: np.ndarray, num_clusters: int):
        all_weights = []
        for snapshot in volumes:
            print('[INFO] Computing whitened features')
            features = whiten(np.reshape(snapshot, (len(snapshot), -1)))
            clustering = None
            distortion = None
            print('[INFO] Computing clusters')
            with tqdm.tqdm(total=self.num_trials) as pbar:
                for i in range(self.num_trials):
                    _clustering, _distortion = kmeans(features, num_clusters, iter=self.max_iter, seed=self.rng)
                    if distortion is None or _distortion < distortion:
                        clustering = _clustering
                        distortion = _distortion
                    pbar.update(1)
            print(f'[INFO] Final distortion: {distortion}')
            distances = cdist(features, clustering, metric='seuclidean')
            k_out = distances.shape[-1]
            if k_out < num_clusters:
                print(f'[INFO] Final clustering had {k_out} clusters, instead of {num_clusters}, as requested!')
                largest = np.amax(distances, axis=-1, keepdims=True)
                _distances = np.zeros(distances.shape[0], num_clusters)
                _distances[:, :k_out] = distances
                _distances[:, k_out:] = largest
                distances = _distances
            print('[INFO] computing weights')
            weights = np.exp(- distances / (2. * np.mean(np.sqrt(distances), axis=-1, keepdims=True) ** 2))
            weights = (weights - np.mean(weights, axis=-1, keepdims=True)) / np.std(weights, axis=-1, keepdims=True)
            all_weights.append(weights)
        all_weights = np.stack(all_weights, axis=0)
        return all_weights


def _test_clustering():

    clustering = InitialClustering()
    data = np.random.randn(1, 64, 12, 250, 352)

    weights = clustering._compute_weights(data, 4)

    print('Finished')


if __name__ == '__main__':
    _test_clustering()
