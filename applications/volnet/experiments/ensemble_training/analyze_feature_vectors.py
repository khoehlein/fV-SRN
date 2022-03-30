import os

import numpy as np
import scipy.cluster.hierarchy
import torch
from matplotlib import pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import TSNE

from volnet.experiments.ensemble_training.univariate_encoding_wide import EXPERIMENT_NAME
from volnet.experiments.ensemble_training import directories as io


RUN_NUMBER = 16
CHECKPOINT_EPOCH = 400
TSNE_PERPLEXITY = 30


def load_feature_vectors():
    output_directory = io.get_output_directory(
        EXPERIMENT_NAME,
        return_output_dir=False, return_log_dir=False, overwrite=True
    )[0]

    checkpoint_directory = os.path.join(
        output_directory, 'results', 'model'
    )

    checkpoint_file = os.path.join(
        checkpoint_directory, 'run{:05d}'.format(RUN_NUMBER), 'model_epoch_{}.pth'.format(CHECKPOINT_EPOCH)
    )

    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model = checkpoint['model']
    params = checkpoint['parameters']

    ensemble_features = model.latent_features.ensemble_features
    vectors = [v.data.data.cpu().numpy() for v in ensemble_features.feature_mapping]
    vectors = np.stack(vectors, axis=0)
    return vectors, params


if __name__ == '__main__':
    description = f'(run {RUN_NUMBER}, epoch {CHECKPOINT_EPOCH})'
    vectors, params = load_feature_vectors()
    whitened = (vectors - np.mean(vectors, axis=0, keepdims=True)) / np.std(vectors, axis=0, keepdims=True)

    def show_principle_components(num_components=4):
        u, s, v = np.linalg.svd(whitened, full_matrices=False)
        fig, ax = plt.subplots(num_components, num_components, figsize=(10, 10))
        fig.suptitle(f'Principle components {description}')
        for i in range(num_components):
            for j in range(num_components):
                if i == j:
                    ax[i, j].hist(u[:, i], bins=16)
                elif j > i:
                    ax[i, j].scatter(u[:, i], u[:, j])
                else:
                    ax[i, j].set_axis_off()
        plt.tight_layout()
        plt.show()
        plt.close()

        plt.figure()
        plt.plot(np.arange(len(s)), s ** 2 / np.sum(s ** 2))
        plt.title('Fractional explained variance' + description)
        plt.show()
        plt.close()

    show_principle_components()

    def show_tsne():
        transform = TSNE(perplexity=TSNE_PERPLEXITY, init='random')
        transformed = transform.fit_transform(whitened)
        plt.figure()
        plt.scatter(transformed[:,0], transformed[:, 1])
        plt.title('t-SNE transformed features' + description)
        plt.show()
        plt.close()

    show_tsne()

    def show_tsne_distances(num_projections=100):
        total = 0
        for _ in range(num_projections):
            transform = TSNE(n_components=2, perplexity=TSNE_PERPLEXITY, init='random')
            transformed = transform.fit_transform(whitened)
            total = total + pdist(transformed, metric='seuclidean')
        total = total / num_projections
        total = np.sqrt(total)

        fig, ax = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [2, 8], 'height_ratios': [2, 8]}, dpi=300)
        fig.suptitle(f'Similarities t-SNE features, n = {num_projections} {description}')
        z = scipy.cluster.hierarchy.linkage(total, method='ward', optimal_ordering=True)
        order = scipy.cluster.hierarchy.leaves_list(z)
        scipy.cluster.hierarchy.dendrogram(z, ax=ax[0, 1], no_labels=True)
        scipy.cluster.hierarchy.dendrogram(z, ax=ax[1, 0], orientation='left')
        ax[0, 0].set_axis_off()
        ax[1, 1].pcolor(squareform(total)[order, :][:, order])
        plt.tight_layout()
        plt.show()
        plt.close()

        fig, ax = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'width_ratios': [2, 8], 'height_ratios': [2, 8]})
        total = pdist(whitened, metric='euclidean')
        fig.suptitle(f'Similarities feature vectors, n = {num_projections} {description}')
        z = scipy.cluster.hierarchy.linkage(total, method='ward', optimal_ordering=True)
        order = scipy.cluster.hierarchy.leaves_list(z)
        scipy.cluster.hierarchy.dendrogram(z, ax=ax[0, 1], no_labels=True)
        scipy.cluster.hierarchy.dendrogram(z, ax=ax[1, 0], orientation='left')
        ax[0, 0].set_axis_off()
        ax[1, 1].pcolor(squareform(total)[order, :][:, order])
        plt.tight_layout()
        plt.show()
        plt.close()


    show_tsne_distances()



print('[INFO] Finished')

