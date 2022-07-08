import json
from typing import Dict, Any

import numpy as np
from matplotlib import pyplot as plt, cm
from matplotlib.colors import ListedColormap

reference_path = '/home/hoehlein/PycharmProjects/deployment/delllat94/fvsrn/applications/config-files/meteo-ensemble_tk_local-min-max.json'

with open(reference_path, 'r') as f:
    data = json.load(f)
specs = data['tf']


def read_gaussian_tf(specs: Dict[str, Any], resolution=512):
    points = specs['points']
    out_c = 0
    out_o = 0
    x = np.linspace(0, 1, resolution)
    for point in points:
        c = np.array(point[:3])
        alpha, mu, variance = point[3:]
        opacity = np.exp(-(x - mu) ** 2 / variance ** 2)
        out_c = out_c + c[None, :] * opacity[:, None]
        out_o = out_o + opacity
    # plt.figure(figsize=(10,4))
    # plt.plot(x, out_o)
    # plt.show()
    out = np.concatenate([out_c, out_o[:, None]], axis=-1)
    out = out / np.amax(out, axis=0, keepdims=True)
    return out

colors = read_gaussian_tf(specs['Gaussian'], resolution=512)
cmap = ListedColormap(colors)

fig = plt.figure(figsize=(6, 0.7))
cax = plt.axes()
fig.colorbar(cm.ScalarMappable(cmap=cmap), orientation='horizontal', cax=cax, aspect=50)
plt.tight_layout()
plt.savefig('colorbar_hor.pdf')
plt.show()
plt.close()

fig = plt.figure(figsize=(0.8, 5))
cax = plt.axes()
fig.colorbar(cm.ScalarMappable(cmap=cmap), orientation='vertical', cax=cax, aspect=50)
plt.tight_layout()
plt.savefig('colorbar_vert.pdf')
plt.show()