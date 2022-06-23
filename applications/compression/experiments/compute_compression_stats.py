import os.path
from enum import Enum

import numpy as np
import pandas as pd

from compression import CompressedArray
from compression.compressors import SZ3, ZFP, TTHRESH
from data.necker_ensemble.single_variable import load_ensemble, load_scales, revert_scaling
from ldcpy.my_dssim import DSSIM2d


class DeviationStatistics(object):

    class Measure(Enum):
        MSE = 'mse'
        RMSE = 'rmse'
        MAE = 'mae'
        MEDAE = 'medae'
        MAXAE = 'maxae'
        PSNR = 'psnr'
        DSSIM = 'dssim'

    def __init__(self, data: np.ndarray, restored: np.ndarray, scales=None):
        self._measures_rescaled = self.compute_deviation_measures(data, restored)
        if scales is not None:
            data_r = revert_scaling(data, scales)
            restored_r = revert_scaling(restored, scales)
            self._measures_reverted = self.compute_deviation_measures(data_r, restored_r)
        else:
            self._measures_reverted = {}

    def compute_deviation_measures(self, data, restored):
        deviation = np.abs(data - restored)
        mse = np.mean(deviation ** 2)
        range = self.compute_range(data, restored)
        return {
            DeviationStatistics.Measure.MSE: mse,
            DeviationStatistics.Measure.RMSE: np.sqrt(mse),
            DeviationStatistics.Measure.MAE: np.mean(deviation),
            DeviationStatistics.Measure.MEDAE: np.median(deviation),
            DeviationStatistics.Measure.MAXAE: np.max(deviation),
            DeviationStatistics.Measure.PSNR: 10 * (np.log(range) - np.log10(mse)),
            DeviationStatistics.Measure.DSSIM: self.compute_dssim(data, restored),
        }

    def compute_range(self, *data):
        min_val = min([np.min(x) for x in data])
        max_val = max([np.max(x) for x in data])
        return max_val - min_val

    def compute_dssim(self, data, restored):
        data = np.transpose(data, (0, 3, 2, 1))
        restored = np.transpose(restored, (0, 3, 2, 1))
        assert data.shape[1:] == (12, 352, 250)
        assert restored.shape == data.shape
        dssim = DSSIM2d()
        values = dssim(data, restored)
        out = np.mean(np.min(values, axis=-1))
        return out

    def to_dict(self):
        return {
            **{key.value + '_rescaled': self._measures_rescaled[key] for key in self._measures_rescaled},
            **{key.value + '_reverted': self._measures_reverted[key] for key in self._measures_reverted},
        }


def compute_ensemble_stats(data, scales, compressor, axis):
    data = np.stack(data, axis=axis)
    compressed = CompressedArray.from_numpy(data, compressor)
    restored = compressed.restore_numpy()
    if axis != 0:
        data = np.moveaxis(data, axis, 0)
        restored = np.moveaxis(restored, axis, 0)
    return DeviationStatistics(data, restored, scales), compressed.compression_ratio()


def compute_single_member_stats(data, scales, compressor):
    total_size = 0
    compressed_size = 0
    restored = []
    for member in data:
        compressed = CompressedArray.from_numpy(member, compressor)
        total_size = total_size + compressed.data_size()
        compressed_size = compressed_size + compressed.code_size()
        restored.append(compressed.restore_numpy())
    restored = np.stack(restored, axis=0)
    data = np.stack(data, axis=0)
    return DeviationStatistics(data, restored, scales=scales), total_size / compressed_size


def evaluate_compressor(data, scales, accuracies, get_compressor):
    singleton = []
    ensemble = []
    for accuracy in accuracies:
        print(accuracy)
        compressor = get_compressor(accuracy)
        for axis in [0,-1]:
            stats, compression_ratio = compute_ensemble_stats(data, scales, compressor, axis)
            ensemble.append({
                'accuracy': accuracy,
                'stacking': axis,
                'compression_ratio': compression_ratio,
                **stats.to_dict()
            })
        stats, compression_ratio = compute_single_member_stats(data, scales, compressor)
        singleton.append({
            'accuracy': accuracy,
            'compression_ratio': compression_ratio,
            **stats.to_dict()
        })
    singleton = pd.DataFrame(singleton)
    ensemble = pd.DataFrame(ensemble)
    return singleton, ensemble


def evaluate_sz3(data, scales):
    accuracies = np.linspace(-9, 0, 30)
    def get_sz3_compressor(accuracy):
        return SZ3(SZ3.CompressionMode.ABS, threshold=10. ** accuracy, verbose=True)
    return evaluate_compressor(data, scales, accuracies, get_sz3_compressor)


def evaluate_tthresh(data,scales):
    accuracies = np.linspace(-7, -1, 30)
    def get_tthresh_compressor(accuracy):
        return TTHRESH(TTHRESH.CompressionMode.RMSE, threshold=10. ** accuracy, verbose=True)
    return evaluate_compressor(data, scales, accuracies, get_tthresh_compressor)


def evaluate_zfp(data, scales):
    accuracies = np.linspace(-9, 3, 30)
    def get_zfp_compressor(accuracy):
        return ZFP(ZFP.CompressionMode.ABS, threshold=10. ** accuracy)
    return evaluate_compressor(data, scales, accuracies, get_zfp_compressor)


def export_compressor_stats(singleton, ensemble, compressor_name, norm_name):
    out_directory = '/home/hoehlein/PycharmProjects/results/fvsrn/classical_compressors'
    if not os.path.isdir(out_directory):
        os.makedirs(out_directory)
    singleton.to_csv(os.path.join(out_directory, f'compressor_stats_{compressor_name}_{norm_name}_singleton.csv'))
    ensemble.to_csv(os.path.join(out_directory, f'compressor_stats_{compressor_name}_{norm_name}_ensemble.csv'))


def main():
    for norm in ['local', 'level', 'global']:
        ensemble = load_ensemble(f'{norm}-min-max', 'tk', min_member=1, max_member=65, time=4)
        scales = load_scales(f'{norm}-min-max', 'tk')
        print('[INFO] Data shape:', ensemble[0].shape)
        stats = evaluate_sz3(ensemble, scales)
        export_compressor_stats(*stats, 'sz3', norm)
        stats = evaluate_tthresh(ensemble, scales)
        export_compressor_stats(*stats, 'tthresh', norm)
        stats = evaluate_zfp(ensemble, scales)
        export_compressor_stats(*stats, 'zfp', norm)


if __name__ == '__main__':
    main()
