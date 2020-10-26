import numpy as np
from scipy.special import softmax

from dataclasses import dataclass

from seir.data import CovidData
from seir.parameters import FittingParams


def log_lognormal_likelihood(model, truth):
    if truth is None:
        return (np.array(0), np.array(0))
    sigma = np.sqrt(np.mean((np.log(model + 1e-20) - np.log(truth + 1e-20)) ** 2, axis=(0, 1), keepdims=True))
    log_weights = -1 / 2 * np.log(2 * np.pi * sigma ** 2 + 1e-20) - (np.log(model + 1e-20) - np.log(truth + 1e-20)) ** 2 / (
            2 * sigma ** 2)
    return np.sum(log_weights, axis=(0, 1))


@dataclass
class BayesSIRFitter:
    model: CovidData
    truth: CovidData
    params: FittingParams

    @property
    def nb_resamples(self):
        return max(int(self.params.ratio_resample * self.model.nb_samples), 1)

    def __post_init__(self):
        assert self.truth.nb_samples == 1 or self.truth.nb_samples == self.model.nb_samples

    def get_posterior_samples(self, **kwargs):
        log_weights = np.zeros(max(self.model.nb_samples, 1))
        if self.params.fit_deaths:
            log_weights += self._fit_attr('deaths', self.params.fit_daily)
        if self.params.fit_recovered:
            log_weights += self._fit_attr('recovered', self.params.fit_daily)
        if self.params.fit_infected:
            log_weights += self._fit_attr('infected', self.params.fit_daily)
        if self.params.fit_hospitalised:
            log_weights += self._fit_attr('hospitalised')
        if self.params.fit_critical:
            log_weights += self._fit_attr('critical')

        weights = softmax(log_weights)
        resample_indices = np.random.choice(self.model.nb_samples, self.nb_resamples, p=weights)

        posteriors = {}
        for k, v in kwargs.items():
            if isinstance(v, list):
                posteriors[k] = []
                for vi in v:
                    posteriors[k].append(self.resample_value(vi, resample_indices))
            else:
                posteriors[k] = self.resample_value(v, resample_indices)
            if k == 'nb_samples':
                posteriors[k] = self.nb_resamples

        return posteriors

    def resample_value(self, value, resample_indices):
        if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[-1] == self.model.nb_samples:
            return value[..., resample_indices]
        else:
            return value

    def _fit_attr(self, attr, fit_daily: bool = False):
        if (
                getattr(self.model, attr) is not None
                and getattr(self.truth, attr) is not None
        ):
            model_td = getattr(self.model, attr)
            truth_td = getattr(self.truth, attr)

            model_timestamp = model_td.timestamp
            truth_timestamp = truth_td.timestamp

            intersect = np.intersect1d(model_timestamp, truth_timestamp)
            model_idx = []
            truth_idx = []
            for i in range(len(model_timestamp)):
                if model_timestamp[i] in intersect:
                    model_idx.append(i)
                    truth_idx.append(np.where(truth_timestamp == model_timestamp[i])[0][0])

            model_data = model_td.data
            model_data = model_data[model_idx]
            model_timestamp = model_timestamp[model_idx]
            truth_data = truth_td.data
            truth_data = truth_data[truth_idx]
            truth_timestamp = truth_timestamp[truth_idx]

            model_data = np.sum(model_data, axis=1, keepdims=True) if self.params.fit_totals else model_data
            truth_data = np.sum(truth_data, axis=1, keepdims=True) if self.params.fit_totals else truth_data
            assert model_data.shape[1] == truth_data.shape[1], \
                "Number of groups between model and true data does not match. Can be overcome by fitting to totals"

            if self.params.fit_interval > 0:
                model_data = model_data[::self.params.fit_interval]
                truth_data = truth_data[::self.params.fit_interval]
                model_timestamp = model_timestamp[::self.params.fit_interval]
                truth_timestamp = truth_timestamp[::self.params.fit_interval]

            if fit_daily:
                model_data = np.diff(model_data, axis=0) \
                             / np.expand_dims(np.diff(model_timestamp), axis=(1, 2)) * self.params.fit_interval
                truth_data = np.diff(truth_data, axis=0) \
                             / np.expand_dims(np.diff(truth_timestamp), axis=(1, 2)) * self.params.fit_interval

            log_weights = log_lognormal_likelihood(model_data, truth_data)
            return log_weights
        log_weights = np.zeros(max(self.model.nb_samples, 1))
        return log_weights
