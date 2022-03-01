import math

import numpy as np
from scipy import stats
from statsmodels import api as sm


class TestResult(object):

    def best_split(self):
        raise NotImplementedError()

    def split_ranks(self):
        raise NotImplementedError()

    def reject(self):
        raise NotImplementedError()


class MyKolmogorovSmirnovTestNd(object):

    def __init__(self, alpha=0.05):
        assert alpha > 0
        self.alpha = alpha

    def critical_factor(self):
        return math.sqrt(-math.log(self.alpha / 2.) / 2.)

    def threshold(self, n1, n2):
        return self.critical_factor() * np.sqrt((n1 + n2) / (n1 * n2))

    def compute(self, samples, classification):
        assert classification.shape[:-1] == samples.shape, f'[ERROR] Expected shape {samples.shape},got {classification.shape[:-1]} instead.'
        sample_order = np.argsort(samples, axis=0)
        leading_dim_grid = tuple([ax.ravel() for ax in np.meshgrid(*[np.arange(s) for s in samples.shape], indexing='ij')[1:]])
        ordered_grid = (sample_order.ravel(),*leading_dim_grid)
        ordered_classification = np.reshape(classification[ordered_grid], classification.shape)
        sample_counts = np.stack([
            np.cumsum(ordered_classification.astype(int), axis=0),
            np.cumsum((~ordered_classification).astype(int), axis=0)
        ], axis=0) # shape (left/right, num_samples, nodes, dim)
        total_samples = sample_counts[:, -1] # shape (l/r, nodes, dim)
        assert np.all(total_samples) > 0
        cdfs = sample_counts / total_samples[:, None, ...]
        test_statistics = np.max(np.abs(np.diff(cdfs, axis=0)[0]), axis=0) # shape (nodes, dim)
        thresholds = self.threshold(total_samples[0], total_samples[1])
        significance_ratios = test_statistics / thresholds
        return MyKolmogorovSmirnovTestNd.Result(test_statistics, significance_ratios)

    class Result(TestResult):

        def __init__(self, test_statistics, significance_ratios):
            self.test_statistics = test_statistics
            self.significance_ratios = significance_ratios

        def reject(self):
            return np.any(self.significance_ratios > 1., axis=-1)

        def best_split(self):
            return np.argmax(self.significance_ratios, axis=-1)

        def split_ranks(self):
            return np.argsort(self.significance_ratios.shape[-1] - 1 - np.argsort(self.significance_ratios, axis=-1), axis=-1)


class WelchTTestNd(object):

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def compute(self, samples, classification):
        assert len(samples.shape) == 1
        num_samples = len(samples)
        assert len(classification) == num_samples
        results = [
            stats.ttest_ind(samples[c], samples[~c], equal_var=False, alternative='two-sided')
            for c in classification.T
        ]
        test_statistics = np.array([r.statistic for r in results])
        p_values = np.array([r.pvalue for r in results])
        return WelchTTestNd.Result(test_statistics, p_values, self.alpha)

    class Result(TestResult):

        def __init__(self, test_statistics, p_values, alpha):
            self.test_statistics = test_statistics
            self.p_values = p_values
            self.alpha = alpha

        def reject(self):
            return np.any(self.p_values < self.alpha)

        def best_split(self):
            return np.argmin(self.p_values)

        def split_ranks(self):
            return np.argsort(np.argsort(self.p_values))


class KolmogorovSmirnovTestNd(object):

    def __init__(self, alpha=0.05):
        self.alpha = alpha

    def compute(self, samples, classification):
        assert len(samples.shape) == 1
        num_samples = len(samples)
        assert len(classification) == num_samples
        results = [
            stats.ks_2samp(samples[c], samples[~c], alternative='two-sided', mode='asymp')
            for c in classification.T
        ]
        test_statistics = np.array([r.statistic for r in results])
        p_values = np.array([r.pvalue for r in results])
        return KolmogorovSmirnovTestNd.Result(test_statistics, p_values, self.alpha)

    class Result(TestResult):

        def __init__(self, test_statistics, p_values, alpha):
            self.test_statistics = test_statistics
            self.p_values = p_values
            self.alpha = alpha

        def reject(self):
            return np.any(self.p_values < self.alpha)

        def best_split(self):
            return np.argmin(self.p_values)

        def split_ranks(self):
            return np.argsort(np.argsort(self.p_values))


# class CombinedTestAnd(object):
#
#     def __init__(self, *tests):
#         self.tests = tests
#
#     def compute(self, samples, classification):
#         return [t.compute(samples, classification) for t in self.tests]
#
#     def outcome(self):
#         return np.all([t.outcome() for t in self.tests])
#
#     def best_split(self):
#         mean_ranks = np.mean(np.array([t.ranks() for t in self.tests]))
#         return np.argmin(mean_ranks)
#
#     def split_ranks(self):
#         mean_ranks = np.mean(np.array([t.ranks() for t in self.tests]))
#         return np.argsort(np.argsort(mean_ranks))
#
#
# class CombinedTestOr(object):
#
#     def __init__(self, *tests):
#         self.tests = tests
#
#     def compute(self, samples, classification):
#         return [t.compute(samples, classification) for t in self.tests]
#
#     def outcome(self):
#         return np.any([t.outcome() for t in self.tests])
#
#     def best_split(self):
#         mean_ranks = np.mean(np.array([t.ranks() for t in self.tests]))
#         return np.argmin(mean_ranks)
#
#     def split_ranks(self):
#         mean_ranks = np.mean(np.array([t.ranks() for t in self.tests]))
#         return np.argsort(np.argsort(mean_ranks))
#
#
# class MeanAboveThreshold(object):
#
#     def __init__(self, threshold, alpha=0.05):
#         self.threshold = threshold
#         self.alpha = alpha
#         self.p_value = None
#         self.test_statistic = None
#
#     def compute(self, sample):
#         result = stats.ttest_1samp(sample, self.threshold, alternative='greater')
#         self.p_value = result.pvalue
#         self.test_statistic = result.statistic
#         return self
#
#     def outcome(self):
#         return self.p_value < self.alpha


class WhiteHomoscedasticityTest(object):

    def __init__(self, alpha=0.05, simplify_predictors=True):
        self.alpha = alpha
        self.simplify_predictors = simplify_predictors

    def compute(self, sample, coordinates):
        X = sm.add_constant(coordinates)
        result = sm.OLS(sample, X).fit()
        if self.simplify_predictors:
            X = sm.add_constant(result.predict()[:, None])
        i0, i1 = np.triu_indices(X.shape[-1])
        X = X[:, i0] * X[:, i1]
        result = sm.OLS(result.resid ** 2, X).fit()
        lm = len(sample) * result.rsquared
        # assert result.df_model == np.linalg.matrix_rank(X) - 1
        p_value = stats.chi2.sf(lm, result.df_model)
        test_statistic = lm
        return WhiteHomoscedasticityTest.Result(test_statistic, p_value, self.alpha)

    class Result(object):

        def __init__(self, test_tatistic, p_value, alpha):
            self.test_tatistic = test_tatistic
            self.p_value = p_value
            self.alpha = alpha

        def reject(self):
            return self.p_value < self.alpha


class MyWhiteHomoscedasticityTest(object):

    def __init__(self, alpha=0.05, simplify_predictors=True):
        self.alpha = alpha
        self.simplify_predictors = simplify_predictors

    def compute(self, samples: np.ndarray, coordinates: np.ndarray):
        coordinates = np.transpose(coordinates, (1, 0, 2))
        # coordinates shape: (nodes, num_samples, dim)
        samples = samples.T
        # samples shape (nodes, num_samples)
        X = self._add_constant(coordinates)
        _, predictions, residuals = self._ols(X, samples)
        if self.simplify_predictors:
            X = self._add_constant(predictions[..., None])
        i0, i1 = np.triu_indices(X.shape[-1])
        X = X[..., i0] * X[..., i1]
        residuals_squared = residuals ** 2
        _, predictions, residuals = self._ols(X, residuals_squared)
        r_squared = self._compute_r_squared(residuals_squared, residuals)
        lm = samples.shape[-1] * r_squared
        df = X.shape[-1] - 1 # degrees of freedom
        p_value = stats.chi2.sf(lm, df)
        test_statistic = lm
        return MyWhiteHomoscedasticityTest.Result(test_statistic, p_value, self.alpha)

    @staticmethod
    def _add_constant(data):
        X = np.concatenate([np.ones(data.shape[:-1])[..., None], data], axis=-1)
        return X

    @staticmethod
    def _ols(X: np.ndarray, targets: np.array):
        pinv = np.linalg.pinv(X)  # shape (*nodes, dim + 1, num_samples)
        weights = np.matmul(pinv, targets[..., None])  # shape (*nodes, dim+ 1, 1)
        predictions = np.matmul(X, weights)[..., 0]  # shape (*nodes, num_samples)
        residuals = targets - predictions
        return weights, predictions, residuals

    @staticmethod
    def _compute_r_squared(samples, residuals):
        ss_res = np.mean(residuals ** 2, axis=-1)
        ss_tot = np.var(samples, axis=-1, ddof=0)
        return 1. - (ss_res / ss_tot)

    class Result(object):

        def __init__(self, test_tatistic, p_value, alpha):
            self.test_tatistic = test_tatistic
            self.p_value = p_value
            self.alpha = alpha

        def reject(self):
            return self.p_value < self.alpha


def _test():
    num_samples = 8
    dim = 3
    alpha = 0.05
    simplify = True
    samples = np.random.randn(num_samples)
    coordinates = np.random.rand(num_samples, dim)
    classification = coordinates < 0.5
    test1 = KolmogorovSmirnovTestNd(alpha=alpha)
    test2 = MyKolmogorovSmirnovTestNd(alpha=alpha)
    result1 = test1.compute(samples, classification)
    # result2 = test2.compute(samples, classification)
    samples = np.stack([
        samples,
        np.random.randn(num_samples)
    ], axis=0)
    coordinates = np.stack([
        coordinates,
        np.random.rand(num_samples, dim)
    ], axis=0)
    classification = coordinates < 0.5
    result3 = test2.compute(samples, classification)
    print('Finished')

if __name__ == '__main__':
    _test()


