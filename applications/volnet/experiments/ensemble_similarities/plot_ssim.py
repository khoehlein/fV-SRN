import matplotlib.pyplot as plt
import numpy as np
import os

import scipy.cluster.hierarchy
from scipy.spatial.distance import squareform

data = np.load('similarities.npz')

similarities = data['similarities']

# dissimilarities = np.mean(similarities, axis=-1) #
dissimilarities = (1. - np.amin(similarities + np.eye(len(similarities))[..., None], axis=-1) ) / 2.

print(np.amin(similarities, axis=-1))

z = scipy.cluster.hierarchy.linkage(squareform(dissimilarities), method='ward', optimal_ordering=True)
order = scipy.cluster.hierarchy.leaves_list(z)
fig, ax = plt.subplots(2, 2, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 8], 'width_ratios': [2, 8]}, dpi=300)
ax[0, 0].set_axis_off()
p = ax[1, 1].pcolor(dissimilarities[order, :][:, order])
plt.colorbar(p)
scipy.cluster.hierarchy.dendrogram(z, orientation='top', ax=ax[0, 1])
scipy.cluster.hierarchy.dendrogram(z, orientation='left', ax=ax[1, 0])
plt.tight_layout()
plt.show()
plt.close()
