import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mode = 'reverted'

fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharex='all', sharey='row', dpi=300)

for i, compressor in enumerate(['sz3', 'tthresh', 'zfp']):
    for j, norm in enumerate(['global', 'level', 'local']):

        file_name = '/home/hoehlein/PycharmProjects/results/fvsrn/classical_compressors/compressor_stats_{}_{}_ensemble.csv'.format(compressor, norm)
        data = pd.read_csv(file_name)
        data = data.loc[data['stacking'] == 0, :]
        data = data.loc[np.logical_and(data['compression_ratio'] <= 1500, data['compression_ratio'] > 1.), :]
        ax[0, i].plot(data['compression_ratio'], data[f'rmse_{mode}'], label=f'{norm} (ensemble)')
        ax[1, i].plot(data['compression_ratio'], data[f'dssim_{mode}'], label=f'{norm} (ensemble)')

for i, compressor in enumerate(['sz3', 'tthresh', 'zfp']):
    ax[0, i].set_prop_cycle(None)
    ax[1, i].set_prop_cycle(None)
    for j, norm in enumerate(['global', 'level', 'local']):

        file_name = '/home/hoehlein/PycharmProjects/results/fvsrn/classical_compressors/compressor_stats_{}_{}_singleton.csv'.format(
            compressor, norm)
        data = pd.read_csv(file_name)
        data = data.loc[np.logical_and(data['compression_ratio'] <= 1500, data['compression_ratio'] > 1.), :]
        # data = data.loc[data['compression_ratio'] <= 1500, :]
        ax[0, i].plot(data['compression_ratio'], data[f'rmse_{mode}'], label=f'{norm} (singleton)', linestyle='--')
        ax[1, i].plot(data['compression_ratio'], data[f'dssim_{mode}'], label=f'{norm} (singleton)', linestyle='--')

    ax[0, i].set(xscale='log', yscale='log', title=compressor)
    ax[0, i].grid()
    ax[1, i].set(xscale='log')
    ax[1, i].grid()

mode_label = 'original' if mode == 'reverted' else mode
ax[0, 0].set(ylabel=f'RMSE ({mode_label})')
ax[1, 0].set(ylabel=f'Data SSIM ({mode_label})')
ax[1, 1].set(xlabel='Compression ratio')
ax[1, 1].legend()
plt.tight_layout()
plt.show()
