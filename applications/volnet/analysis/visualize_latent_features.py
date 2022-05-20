import argparse
import os

import numpy as np
import torch
import pyrenderer
from sklearn.decomposition import PCA


def get_feature_data(latent_features):
    return latent_features.volumetric_features.data


def export_features(data, file_name):
    num_channels = data.shape[0]
    vol = pyrenderer.Volume()
    vol.worldX = 10.
    vol.worldY = 10.
    vol.worldZ = 1.
    for c in range(num_channels):
        x = data[c].T
        min_x = torch.amin(x,dim=(0, 1, 2))
        max_x = torch.amax(x,dim=(0, 1, 2))
        x = (x - min_x) / (max_x - min_x + 1.e-6)
        print(x)
        vol.add_feature_from_tensor(f'f{c}', x[None, ...])
    vol.save(file_name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='path to *.pth file', required=True)
    parser.add_argument('--target', type=str, help='path to *.cvol file', required=True)
    args = vars(parser.parse_args())

    file_path = args['input']
    target_folder = os.path.abspath(args['target'])
    if not os.path.isdir(target_folder):
        os.makedirs(target_folder)

    checkpoint = torch.load(file_path, map_location='cpu')
    model = checkpoint['model']

    latent_features = model.latent_features

    data = get_feature_data(latent_features)
    export_features(data, os.path.join(target_folder, 'raw_features.cvol'))

    pca = PCA()
    flattened = data.flatten(start_dim=1).T.data.numpy()
    features = pca.fit_transform((flattened - np.mean(flattened, axis=0, keepdims=True)) / np.std(flattened, axis=0, keepdims=True))
    features = torch.from_numpy(features).T.reshape(data.shape)
    export_features(features, os.path.join(target_folder, 'pca_features.cvol'))

    print('Finished')





if __name__ == '__main__':
    main()