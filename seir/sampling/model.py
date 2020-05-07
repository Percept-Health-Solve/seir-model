import numpy as np
import pandas as pd

from scipy.integrate import odeint
from scipy.special import softmax, gammaln

import logging


class SamplingNInfectiousModel:

    def __init__(self,
                 nb_groups: int,
                 baseline_beta=None,
                 rel_lockdown_beta=None,
                 rel_postlockdown_beta=None,
                 rel_beta_as=None,
                 time_inc=None,
                 inf_as_prop=None,
                 inf_m_prop=None,
                 time_infectious=None,
                 time_s_to_h=None,
                 time_h_to_icu=None,
                 time_h_to_r=None,
                 time_icu_to_r=None,
                 time_icu_to_d=None,
                 hosp_icu_prop=None,
                 icu_d_prop=None,
                 y0=None,
                 imported_func=None):
        logging.info('Initizializing model')
        # cast potential sampling variables to correct data types
        baseline_beta = np.asarray(baseline_beta)
        rel_lockdown_beta = np.asarray(rel_lockdown_beta)
        rel_postlockdown_beta = np.asarray(rel_postlockdown_beta)
        rel_beta_as = np.asarray(rel_beta_as)
        time_inc = np.asarray(time_inc)
        inf_as_prop = np.asarray(inf_as_prop)
        inf_m_prop = np.asarray(inf_m_prop)
        time_infectious = np.asarray(time_infectious)
        time_s_to_h = np.asarray(time_s_to_h)
        time_h_to_icu = np.asarray(time_h_to_icu)
        time_h_to_r = np.asarray(time_h_to_r)
        time_icu_to_r = np.asarray(time_icu_to_r)
        time_icu_to_d = np.asarray(time_icu_to_d)
        hosp_icu_prop = np.asarray(hosp_icu_prop)
        icu_d_prop = np.asarray(icu_d_prop)

        nb_samples, (scalar_vars, group_vars, sample_vars) = _determine_sample_vars({
            'baseline_beta': baseline_beta,
            'rel_lockdown_beta': rel_lockdown_beta,
            'rel_postlockdown_beta': rel_postlockdown_beta,
            'rel_beta_as': rel_beta_as,
            'time_inc': time_inc,
            'inf_as_prop': inf_as_prop,
            'inf_m_prop': inf_m_prop,
            'time_infectious': time_infectious,
            'time_s_to_h': time_s_to_h,
            'time_h_to_icu': time_h_to_icu,
            'time_h_to_r': time_h_to_r,
            'time_icu_to_r': time_icu_to_r,
            'time_icu_to_d': time_icu_to_d,
            'hosp_icu_prop': hosp_icu_prop,
            'icu_d_prop': icu_d_prop
        }, nb_groups)

        logging.info(f'Scalar variables: {list(scalar_vars.keys())}')
        logging.info(f'Group variables: {list(group_vars.keys())}')
        logging.info(f'Sampled variables: {list(sample_vars.keys())}')

        # recalculate hospital and ICU proportions given competing time rates
        f_hosp_icu_prop = hosp_icu_prop * 1/time_h_to_r / \
                          ((1-hosp_icu_prop) * 1/time_h_to_icu + hosp_icu_prop * 1/time_h_to_r)

        f_icu_d_prop = icu_d_prop * time_icu_to_r / (icu_d_prop * 1/time_icu_to_r + (1 - icu_d_prop) * 1/time_icu_to_d)

        # check if y0 is correct
        # 1 s and 1 e state, 6 infectious states, and 4 recovered states (since severe cases must first go through h
        # in order to recover)
        y0 = np.asarray(y0)
        assert y0.size == 13 * nb_groups * nb_samples, \
            f"y0 should have size {13 * nb_groups * nb_samples}, got {y0.size} instead"
        n = np.sum(y0.reshape(nb_samples, nb_groups * 13), axis=1, keepdims=True)

        # check infectious func
        def infectious_func(t):
            if t < -11:
                return 1
            elif -11 <= t < 0:
                return 1 - (1 - rel_lockdown_beta) / 11 * (t - 11)
            elif 0 <= t < 6 * 7:
                return rel_lockdown_beta
            # else
            return rel_postlockdown_beta

        # check imported func
        if imported_func is not None:
            assert callable(imported_func), "imported_func is not callable"
        else:
            imported_func = lambda t: 0

        # set self variables
        self.nb_groups = nb_groups
        self.nb_samples = nb_samples
        self.nb_infectious = 6
        self.beta = baseline_beta
        self.rel_beta_as = rel_beta_as
        self.time_inc = time_inc
        self.inf_as_prop = inf_as_prop
        self.inf_m_prop = inf_m_prop
        self.time_infectious = time_infectious
        self.time_s_to_h = time_s_to_h
        self.time_h_to_icu = time_h_to_icu
        self.time_h_to_r = time_h_to_r
        self.time_icu_to_r = time_icu_to_r
        self.time_icu_to_d = time_icu_to_d
        self.hosp_icu_prop = hosp_icu_prop
        self.icu_d_prop = icu_d_prop
        self.y0 = y0
        self.n = n

        self.f_hosp_icu_prop = f_hosp_icu_prop
        self.f_icu_d_prop = f_icu_d_prop

        self.scalar_vars = scalar_vars
        self.group_vars = group_vars
        self.sample_vars = sample_vars

        self.infectious_func = infectious_func
        self.imported_func = imported_func

        self._solved = False
        self._t = None
        self.solution = None

    def _ode(self, y, t):
        # get seird
        s, e, i_as, i_m, i_s, i_i, i_h, i_icu, _, _, _, _, _ = self._get_seird_from_flat_y(y)

        # get meta vars
        inf_s_prop = 1 - self.inf_as_prop - self.inf_m_prop
        time_i_to_h = self.time_s_to_h - self.time_infectious

        # solve seird equations
        ds = - 1 / self.n * self.infectious_func(t) * self.beta * np.sum(self.rel_beta_as * i_as + i_m + i_s, axis=1, keepdims=True) * s
        de = 1 / self.n * self.infectious_func(t) * self.beta * np.sum(self.rel_beta_as * i_as + i_m + i_s, axis=1, keepdims=True) * s - e / self.time_inc
        di_as = self.inf_as_prop * e / self.time_inc - i_as / self.time_infectious
        di_m = self.inf_m_prop * e / self.time_inc - i_m / self.time_infectious
        di_s = inf_s_prop * e / self.time_inc - i_s / self.time_infectious
        di_i = i_s / self.time_infectious - i_i / time_i_to_h
        di_h = i_i / time_i_to_h - self.f_hosp_icu_prop * i_h / self.time_h_to_icu - (1 - self.f_hosp_icu_prop) * i_h / self.time_h_to_r
        di_icu = self.f_hosp_icu_prop * i_h / self.time_h_to_icu - self.f_icu_d_prop * i_icu / self.time_icu_to_d \
                 - (1 - self.f_icu_d_prop) * i_icu / self.time_icu_to_r
        dr_as = i_as / self.time_infectious
        dr_m = i_m / self.time_infectious
        # dr_s = np.zeros((self.nb_samples, self.nb_groups))
        # dr_i = np.zeros((self.nb_samples, self.nb_groups))
        dr_h = (1 - self.f_hosp_icu_prop) * i_h / self.time_h_to_r
        dr_icu = (1 - self.f_icu_d_prop) * i_icu / self.time_icu_to_r
        dd_icu = self.f_icu_d_prop * i_icu / self.time_icu_to_d

        dydt = np.concatenate([
            ds.reshape(self.nb_samples, self.nb_groups, 1),
            de.reshape(self.nb_samples, self.nb_groups, 1),
            di_as.reshape(self.nb_samples, self.nb_groups, 1),
            di_m.reshape(self.nb_samples, self.nb_groups, 1),
            di_s.reshape(self.nb_samples, self.nb_groups, 1),
            di_i.reshape(self.nb_samples, self.nb_groups, 1),
            di_h.reshape(self.nb_samples, self.nb_groups, 1),
            di_icu.reshape(self.nb_samples, self.nb_groups, 1),
            dr_as.reshape(self.nb_samples, self.nb_groups, 1),
            dr_m.reshape(self.nb_samples, self.nb_groups, 1),
            dr_h.reshape(self.nb_samples, self.nb_groups, 1),
            dr_icu.reshape(self.nb_samples, self.nb_groups, 1),
            dd_icu.reshape(self.nb_samples, self.nb_groups, 1)
        ], axis=-1).reshape(-1)

        return dydt

    def solve(self, t, y0=None):
        y0 = self.y0 if y0 is None else y0
        if not self._solved:
            sol = odeint(self._ode, y0, t).reshape(-1, self.nb_samples, self.nb_groups, 13).clip(min=0)
            self.solution = sol
            self._t = t
            self._solved = True
            return sol
        else:
            if np.all(t != self._t) or np.all(y0 != self.y0):
                sol = odeint(self._ode, y0, t).reshape(-1, self.nb_samples, self.nb_groups, 13).clip(min=0)
                self._t = t
                self.solution = sol
                return sol
            else:
                return self.solution

    def sir_posterior(self,
                      t,
                      i_d_obs=None,
                      i_h_obs=None,
                      i_icu_obs=None,
                      d_icu_obs=None,
                      ratio_as_detected=0.,
                      ratio_m_detected=0.3,
                      ratio_s_detected=1.0,
                      ratio_resample: int = 0.1,
                      y0=None) -> dict:
        # cast variables
        t = np.asarray(t)
        i_d_obs = None if i_d_obs is None else np.asarray(i_d_obs).reshape(-1, 1, 1).astype(int)
        i_h_obs = None if i_h_obs is None else np.asarray(i_h_obs).reshape(-1, 1, 1).astype(int)
        i_icu_obs = None if i_icu_obs is None else np.asarray(i_icu_obs).reshape(-1, 1, 1).astype(int)
        d_icu_obs = None if d_icu_obs is None else np.asarray(d_icu_obs).reshape(-1, 1, 1).astype(int)

        # assert shapes
        # TODO: Implement linear interpolation for cases where t does not directly match the data
        # TODO: Implement checks for when data is group specific

        # assert i_d_obs.ndim == 1 and i_d_obs.size == t.size, "Observed detected cases does not match time size"
        # assert i_h_obs.ndim == 1 and i_h_obs.size == t.size, "Observed hospital cases does not match time size"
        # assert i_icu_obs.ndim == 1 and i_icu_obs.size == t.size, "Observed ICU cases does not match time size"
        # assert d_icu_obs.ndim == 1 and d_icu_obs.size == t.size, "Observed deaths does not match time size"

        logging.info('Solving system')
        y = self.solve(t, y0)

        logging.info('Collecting necessary variables')
        i_as = y[:, :, :, 2]
        i_m = y[:, :, :, 3]
        i_s = y[:, :, :, 4]
        i_i = y[:, :, :, 5]
        i_h = y[:, :, :, 6]
        i_icu = y[:, :, :, 7]
        r_as = y[:, :, :, 8]
        r_m = y[:, :, :, 9]
        r_h = y[:, :, :, 10]
        r_icu = y[:, :, :, 11]
        d_icu = y[:, :, :, 12]

        cum_detected_samples = ratio_as_detected * (i_as + r_as) + ratio_m_detected * (i_m + r_m) \
                               + ratio_s_detected * (i_s + i_i + i_h + i_icu + r_h + r_icu + d_icu)


        # model detected cases as poisson distribution y~P(lambda=detected_cases) with stirling's approximation for log y!
        log_weights_detected = 0 if i_d_obs is None else _log_poisson(i_d_obs, cum_detected_samples)
        log_weights_hospital = 0 if i_h_obs is None else _log_poisson(i_h_obs, i_h)
        log_weights_icu = 0 if i_icu_obs is None else _log_poisson(i_icu_obs, i_icu)
        log_weights_dead = 0 if d_icu_obs is None else _log_poisson(d_icu_obs, d_icu)

        # calculate the log weights
        log_weights = log_weights_detected + log_weights_hospital + log_weights_icu + log_weights_dead
        print('log_weights:', log_weights)
        print('log_weights_min:', log_weights.min())
        weights = softmax(log_weights)

        # resample the sampled variables
        m = int(np.round(self.nb_samples * ratio_resample))
        logging.info(f'Resampling {list(self.sample_vars.keys())} {m} times from {self.nb_samples} original samples')
        resample_vars = {}
        for key, value in self.sample_vars.items():
            logging.info(f'Resampling {key}')
            resample_vars[key] = value[np.random.choice(value.shape[0], m, p=weights)]
        logging.info(f'Succesfully resampled {list(resample_vars.keys())} {m} times from {self.nb_samples} original samples')

        return resample_vars

    def _get_seird_from_flat_y(self, y):
        y = y.reshape(self.nb_samples, self.nb_groups, 13)
        s = y[:, :, 0]
        e = y[:, :, 1]
        i_as = y[:, :, 2]
        i_m = y[:, :, 3]
        i_s = y[:, :, 4]
        i_i = y[:, :, 5]
        i_h = y[:, :, 6]
        i_icu = y[:, :, 7]
        r_as = y[:, :, 8]
        r_m = y[:, :, 9]
        r_h = y[:, :, 10]
        r_icu = y[:, :, 11]
        d_icu = y[:, :, 12]

        return s, e, i_as, i_m, i_s, i_i, i_h, i_icu, r_as, r_m, r_h, r_icu, d_icu


def _determine_sample_vars(vars: dict, nb_groups):
    dim_dict = {}
    for key, value in vars.items():
        dim_dict[key] = value.ndim

    # determine scalars, group specific vars, and variables with samples
    scalar_vars = {}
    group_vars = {}
    sample_vars = {}
    nb_samples = None
    for key, value in vars.items():
        if value.ndim == 0:
            # scalar
            scalar_vars[key] = value
        elif value.ndim == 1:
            # shouldn't exist, this is either an ill-defined sampler or ill-defined group var
            raise ValueError(f'Variable {key} should either be zero or two dimensional. This is either an\n'
                             'ill-defined sampler or population group specific variable. If the former, it\n'
                             'take the shape [nb_samples x 1] or [nb_samples x nb_groups], if the latter, it\n'
                             'should take the value [1 x nb_groups].')
        elif value.ndim == 2:
            # sample variable
            val_shape = value.shape
            if val_shape[0] > 1:
                nb_samples = val_shape[0]
            elif val_shape == (1, nb_groups):
                group_vars[key] = value
            else:
                raise ValueError(f'Variable {key} seems to be an ill-defined group specific variable. It should take\n'
                                 f'a shape of (1, {nb_groups}), got {val_shape} instead.')
            if nb_samples:
                if val_shape[0] != nb_samples:
                    raise ValueError(f'Inconsistencies in number of samples made for variable {key}.\n'
                                     f'A previous variable had {nb_samples} samples, this variables\n'
                                     f'as {val_shape[0]} samples.')
                elif val_shape != (nb_samples, 1) and val_shape != (nb_samples, nb_groups):
                    raise ValueError(f'Variable {key} is either an\n'
                                     f'ill-defined sampler or population group specific variable. If the former, it\n'
                                     f'take the shape ({nb_samples}, 1) or ({nb_samples}, {nb_groups}), if the latter,\n'
                                     f'it should take the value (1, {nb_groups}). Got {val_shape} instead.')
                else:
                    sample_vars[key] = value
        else:
            raise ValueError(f'Variable {key} has too many dimension. Should be 0 or 2, got {value.ndims}')

    if not nb_samples:
        nb_samples = 1

    return nb_samples, (scalar_vars, group_vars, sample_vars)


def _log_k_factorial(k):
    if k == 0:
        return 0
    else:
        return 1 / 2 * np.log(2 * np.pi * k) + k * (np.log(k) - 1)


def _log_l(l):
    if l <= 1:
        return 0
    else:
        return np.log(l)


_log_k_factorial = np.vectorize(_log_k_factorial)
_log_l = np.vectorize(_log_l)


def _log_poisson(k, l):
    out = k * _log_l(l) - l - gammaln(k+1)
    out = np.sum(out, axis=(0, 2))
    return out