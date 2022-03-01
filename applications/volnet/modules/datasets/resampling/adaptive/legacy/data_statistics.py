import numpy as np


class _SummaryStatistics(object):

    def mean(self):
        raise NotImplementedError()

    def var(self, unbiased=True):
        raise NotImplementedError()

    def std(self, unbiased=True):
        return np.sqrt(self.var(unbiased=unbiased))

    def num_samples(self):
        raise NotImplementedError()

    def var_of_mean(self):
        raise NotImplementedError()

    def std_of_mean(self):
        return np.sqrt(self.var_of_mean())

    def min(self):
        raise NotImplementedError()

    def max(self):
        raise NotImplementedError()


class SampleSummary(_SummaryStatistics):

    @classmethod
    def from_sample(cls, sample):
        mu = np.mean(sample)
        sum_of_squares = np.sum(np.abs(sample - mu) ** 2)
        return cls(len(sample), mu, sum_of_squares, np.min(sample), np.max(sample))

    @classmethod
    def from_sample_summaries(cls, stats1: 'SampleSummary', stats2: 'SampleSummary'):
        num_samples = stats1._num_samples + stats2._num_samples
        f = stats1._num_samples * stats2._num_samples / num_samples
        mu1, mu2 = stats1.mean(), stats2.mean()
        delta = np.abs(mu2 - mu1)
        mu = (stats1._num_samples * mu1 + stats2._num_samples * mu2) / num_samples
        sum_of_squares = stats1._sum_of_squares + stats2._sum_of_squares + f * delta ** 2
        min_ = np.minimum(stats1.min(), stats2.max())
        max_ = np.maximum(stats1.max(), stats2.max())
        return cls(num_samples, mu, sum_of_squares, min_, max_)

    def __init__(self, num_samples, mu, sum_of_squares, min, max):
        self._num_samples = num_samples
        self._mu = mu
        self._sum_of_squares = sum_of_squares
        self._min = min
        self._max = max

    def mean(self):
        return self._mu

    def var(self, unbiased=True):
        norm = self._num_samples
        if unbiased:
            norm = norm - 1
        return self._sum_of_squares / norm

    def var_of_mean(self):
        return self.var(unbiased=True) / self._num_samples

    def min(self):
        return self._min

    def max(self):
        return self._max

    def num_samples(self):
        return self._num_samples


class _MergerSummary(_SummaryStatistics):

    def __init__(self, sum_of_weights, sum_of_squared_weights, mu, squared_deviation, var_of_mean, min, max):
        self._sum_of_weights = sum_of_weights
        self._sum_of_squared_weights = sum_of_squared_weights
        self._mu = mu
        self._squared_deviation = squared_deviation
        self._var_of_mean = var_of_mean
        self._min = min
        self._max = max

    @classmethod
    def _from_summaries(cls, stats1: _SummaryStatistics, weight1: float, stats2: _SummaryStatistics, weight2: float):
        sum_of_weights = weight1 + weight2
        sum_of_squared_weights = weight1 ** 2 + weight2 ** 2
        f = weight1 * weight2 / sum_of_weights
        mu1, mu2 = stats1.mean(), stats2.mean()
        delta = np.abs(mu2 - mu1)
        mu = (mu1 * weight1 + mu2 * weight2) / sum_of_weights
        squared_deviation = weight1 * stats1.var(unbiased=False) + weight2 * stats2.var(unbiased=False) + f * delta ** 2
        var_of_mean = (weight1 ** 2 * stats1.var_of_mean() + weight2 ** 2 * stats1.var_of_mean()) / sum_of_weights ** 2
        min_ = np.minimum(stats1.min(), stats2.min())
        max_ = np.maximum(stats1.max(), stats2.max())
        return cls(sum_of_weights, sum_of_squared_weights, mu, squared_deviation, var_of_mean, min_, max_)

    def mean(self):
        return self._mu

    def var(self, unbiased=True):
        norm = self._sum_of_weights
        if unbiased:
            norm = norm - self._sum_of_squared_weights / self._sum_of_weights
        return self._squared_deviation / norm

    def num_samples(self):
        # effective sample size according to weights
        return self._sum_of_weights ** 2 / self._sum_of_squared_weights

    def var_of_mean(self):
        return self._var_of_mean

    def min(self):
        return self._min

    def max(self):
        return self._max


class MergerSummary(_MergerSummary):

    @classmethod
    def from_summaries(cls, stats1: _SummaryStatistics, weight1: float, stats2: _SummaryStatistics, weight2: float):
        return cls._from_summaries(stats1, weight1, stats2, weight2)


class DensityData(_MergerSummary):

    @classmethod
    def from_uniform_volume(cls, density: float, volume: float):
        return cls(volume, volume ** 2, density, 0., 0., density, density)

    @classmethod
    def from_density_data(cls, stats1: 'DensityData', weight1: float, stats2: 'DensityData', weight2: float):
        return cls._from_summaries(stats1, weight1, stats2, weight2)


class LossData(object):

    @classmethod
    def from_sample(cls, losses: np.ndarray):
        loss_summary = SampleSummary.from_sample(losses)
        log_loss_summary = SampleSummary.from_sample(np.log(losses))
        return cls(loss_summary, log_loss_summary)

    @classmethod
    def from_loss_data(cls, stats1: 'LossData', weight1: float, stats2: 'LossData', weight2: float):
        ls1, ls2 = stats1._loss_summary, stats2._loss_summary
        lls1, lls2 = stats1._log_loss_summary, stats2._log_loss_summary
        loss_summary = MergerSummary.from_summaries(ls1, weight1, ls2, weight2)
        log_loss_summary = MergerSummary.from_summaries(lls1, weight1, lls2, weight2)
        return cls(loss_summary, log_loss_summary)

    def __init__(self, loss_summary: _SummaryStatistics, log_loss_summary: _SummaryStatistics):
        self._loss_summary = loss_summary
        self._log_loss_summary = log_loss_summary

    def _get_summary(self, log_scale):
        return self._log_loss_summary if log_scale else self._loss_summary

    def mean(self, log_scale=False):
        return self._get_summary(log_scale).mean()

    def var(self, log_scale=False, unbiased=True):
        return self._get_summary(log_scale).var(unbiased=unbiased)

    def std(self, log_scale=False, unbiased=True):
        return self._get_summary(log_scale).std(unbiased=unbiased)

    def min(self, log_scale=False):
        return self._get_summary(log_scale).min()

    def max(self, log_scale=False):
        return self._get_summary(log_scale).max()

    def var_of_mean(self, log_scale=False):
        return self._get_summary(log_scale).var_of_mean()

    def std_of_mean(self, log_scale=False):
        return self._get_summary(log_scale).std_of_mean()