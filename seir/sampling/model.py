import numpy as np
import pandas as pd

from scipy.integrate import odeint
from scipy.special import softmax, gammaln

from seir.utils import calculate_detected_cases

import logging


class SamplingNInfectiousModel:

    nb_states = 18

    def __init__(self,
                 nb_groups: int,
                 beta=None,
                 rel_lockdown_beta=None,
                 rel_postlockdown_beta=None,
                 rel_beta_as=None,
                 prop_a=None,
                 prop_m=None,
                 prop_s_to_h=None,
                 prop_h_to_c=None,
                 prop_h_to_d=None,
                 prop_c_to_d=None,
                 time_incubate=None,
                 time_infectious=None,
                 time_s_to_h=None,
                 time_s_to_c=None,
                 time_h_to_c=None,
                 time_h_to_r=None,
                 time_h_to_d=None,
                 time_c_to_r=None,
                 time_c_to_d=None,
                 contact_heterogeneous: bool = False,
                 contact_k: float = None,
                 y0=None,
                 imported_func=None):
        logging.info('Initizializing model')

        # infectious and relative to infectious rates
        beta = np.asarray(beta)
        rel_lockdown_beta = np.asarray(rel_lockdown_beta)
        rel_postlockdown_beta = np.asarray(rel_postlockdown_beta)
        rel_beta_as = np.asarray(rel_beta_as)

        # proportions
        prop_a = np.asarray(prop_a)
        prop_m = np.asarray(prop_m)
        prop_s_to_h = np.asarray(prop_s_to_h)
        prop_h_to_c = np.asarray(prop_h_to_c)
        prop_h_to_d = np.asarray(prop_h_to_d)
        prop_c_to_d = np.asarray(prop_c_to_d)

        # times
        time_incubate = np.asarray(time_incubate)
        time_infectious = np.asarray(time_infectious)
        time_s_to_h = np.asarray(time_s_to_h)
        time_s_to_c = np.asarray(time_s_to_c)
        time_h_to_c = np.asarray(time_h_to_c)
        time_h_to_r = np.asarray(time_h_to_r)
        time_h_to_d = np.asarray(time_h_to_d)
        time_c_to_r = np.asarray(time_c_to_r)
        time_c_to_d = np.asarray(time_c_to_d)

        # contact heterogeneity variables
        if contact_heterogeneous and contact_k is None:
            raise ValueError("Setting a heterogenous contact model model requires setting the 'contact_k' variable!")
        contact_k = np.asarray(contact_k)

        # calculated vars
        r0 = beta * time_infectious  # TODO: Calculate r0 as leading eigenvalue of NGM

        prop_s = 1 - prop_a - prop_m
        prop_s_to_c = 1 - prop_s_to_h
        # TODO: Remove prop_h_to_r clip in favor of appropriate sampling criteria
        prop_h_to_r = np.clip(1 - prop_h_to_c - prop_h_to_d, 0, 1)
        prop_c_to_r = 1 - prop_c_to_d

        time_i_to_h = time_s_to_h - time_infectious
        time_i_to_c = time_s_to_c - time_infectious


        # collect variables into specific dictionaries

        beta_vars = {
            'beta': beta,
            'rel_lockdown_beta': rel_lockdown_beta,
            'rel_postlockdown_beta': rel_postlockdown_beta,
            'rel_beta_as': rel_beta_as
        }

        prop_vars = {
            'prop_a': prop_a,
            'prop_m': prop_m,
            'prop_s_to_h': prop_s_to_h,
            'prop_h_to_c': prop_h_to_c,
            'prop_h_to_d': prop_h_to_d,
            'prop_c_to_d': prop_c_to_d
        }

        time_vars = {
            'time_incubate': time_incubate,
            'time_infectious': time_infectious,
            'time_s_to_h': time_s_to_h,
            'time_s_to_c': time_s_to_c,
            'time_h_to_c': time_h_to_c,
            'time_h_to_r': time_h_to_r,
            'time_h_to_d': time_h_to_d,
            'time_c_to_r': time_c_to_r,
            'time_c_to_d': time_c_to_d,
        }

        calculated_vars = {
            'r0': r0,
            'prop_s': prop_s,
            'prop_s_to_c': prop_s_to_c,
            'prop_h_to_r': prop_h_to_r,
            'prop_c_to_r': prop_c_to_r,
            'time_i_to_h': time_i_to_h,
            'time_i_to_c': time_i_to_c
        }

        contact_vars = {
            'contact_k': contact_k,
        }

        # assert specific properties of variables
        for key, value in beta_vars.items():
            assert np.all(beta >= 0), f"Value in '{key}' is smaller than 0"
        for key, value in prop_vars.items():
            assert np.all(value <= 1), f"Value in proportion '{key}' is greater than 1"
            assert np.all(value >= 0), f"Value in proportion '{key}' is smaller than 0"
        for key, value in time_vars.items():
            assert np.all(value > 0), f"Value in time '{key}' is smaller than or equal to 0"
        for key, value in contact_vars.items():
            assert np.all(value > 0), f"Value in '{key} is smaller than or equal to 0."

        # check if calculated vars obey constraints
        # only need to check the few that aren't caught by the above checks
        assert np.all(prop_s <= 1), "Value in proportion 'prop_s = 1 - prop_a - prop_m' is greater than 1"
        assert np.all(prop_s >= 0), "Value in proportion 'prop_s = 1 - prop_a - prop_m' is smaller than 0"

        assert np.all(prop_h_to_r <= 1), \
            "Value in proportion 'prop_h_to_r = 1 - prop_h_to_c - prop_h_to_d' is greater than 1"
        assert np.all(prop_h_to_r >= 0), \
            "Value in proportion 'prop_h_to_r = 1 - prop_h_to_c - prop_h_to_d' is smaller than 0"

        assert np.all(time_i_to_h > 0), "Value in time 'time_i_to_h' is smaller than or equal to 0"
        assert np.all(time_i_to_c > 0), "Value in time 'time_i_to_c' is smaller than or equal to 0"

        # intrinsic parameter measuring the number of internal states of which to keep track
        nb_states = SamplingNInfectiousModel.nb_states

        # detect the number of given samples, check for consistency, and assert the shapes of the parameters
        nb_samples, (scalar_vars, group_vars, sample_vars) = _determine_sample_vars({
            **beta_vars,
            **prop_vars,
            **time_vars,
            **contact_vars
        }, nb_groups)

        # do the same for the calculated variables
        _, (calculated_scalar_vars, calculated_group_vars, calculated_sample_vars) = _determine_sample_vars({
            **calculated_vars
        }, nb_groups)

        logging.info(f'Scalar variables: {list(scalar_vars.keys())}')
        logging.info(f'Group variables: {list(group_vars.keys())}')
        logging.info(f'Sampled variables: {list(sample_vars.keys())}')

        logging.info(f'Calculated scalar variables: {list(calculated_scalar_vars.keys())}')
        logging.info(f'Calculated group variables: {list(calculated_group_vars.keys())}')
        logging.info(f'Calculated sampled variables: {list(calculated_sample_vars.keys())}')

        # check if y0 shape is correct
        y0 = np.asarray(y0)
        assert y0.size == nb_states * nb_groups * nb_samples, \
            f"y0 should have size {nb_states * nb_groups * nb_samples}, got {y0.size} instead"

        # find the total population from y0, assumed to be constant or change very little over time
        n = np.sum(y0.reshape(nb_samples, nb_groups * nb_states), axis=1, keepdims=True)

        # build infectious function from given parameters
        def infectious_func(t):
            if t < -11:
                return 1
            elif -11 <= t < 0:
                return 1 - (1 - rel_lockdown_beta) / 11 * (t - 11)
            elif 0 <= t < 5 * 7:
                return rel_lockdown_beta
            # else
            return rel_postlockdown_beta

        # check imported func
        if imported_func is not None:
            assert callable(imported_func), "imported_func is not callable"
        else:
            imported_func = lambda t: 0

        # set properties
        self.nb_groups = nb_groups
        self.nb_states = nb_states
        self.nb_samples = nb_samples
        self.nb_infectious = 10  # for consistency with previous versions of the ASSA model

        # beta proporties
        self.beta = beta
        self.rel_beta_as = rel_beta_as
        self.rel_lockdown_beta = rel_lockdown_beta
        self.rel_postlockdown_beta = rel_postlockdown_beta

        # proportion proporties
        self.prop_a = prop_a
        self.prop_m = prop_m
        self.prop_s = prop_s
        self.prop_s_to_h = prop_s_to_h
        self.prop_s_to_c = prop_s_to_c
        self.prop_h_to_c = prop_h_to_c
        self.prop_h_to_d = prop_h_to_d
        self.prop_h_to_r = prop_h_to_r
        self.prop_c_to_d = prop_c_to_d
        self.prop_c_to_r = prop_c_to_r

        # time properties
        self.time_incubate = time_incubate
        self.time_infectious = time_infectious
        self.time_s_to_h = time_s_to_h
        self.time_s_to_c = time_s_to_c
        self.time_i_to_h = time_i_to_h
        self.time_i_to_c = time_i_to_c
        self.time_h_to_c = time_h_to_c
        self.time_h_to_r = time_h_to_r
        self.time_h_to_d = time_h_to_d
        self.time_c_to_r = time_c_to_r
        self.time_c_to_d = time_c_to_d

        # contact proprties
        self.contact_heterogeneous = contact_heterogeneous
        self.contact_k = contact_k

        # y0 properties
        self.y0 = y0
        self.n = n

        # variable disctionaries
        self.beta_vars = beta_vars
        self.prop_vars = prop_vars
        self.time_vars = time_vars
        self.calculated_vars = calculated_vars
        self.contact_vars = contact_vars

        # scalar, group, and sample properties
        self.scalar_vars = scalar_vars
        self.group_vars = group_vars
        self.sample_vars = sample_vars

        self.calculated_scalar_vars = calculated_scalar_vars
        self.calculated_group_vars = calculated_group_vars
        self.calculated_sample_vars = calculated_sample_vars

        # function properties
        self.infectious_func = infectious_func
        self.imported_func = imported_func

        # private proporties relating to whether the model has been internally solved at least once
        self._solved = False
        self._t = None
        self._solution = None

        # initialising proporties for use in the calculate_sir_posterior function
        self.resample_vars = None
        self.calculated_resample_vars = None
        self.log_weights = None
        self.weights = None
        self.nb_resamples = None
        self.resample_indices = None

    def _ode(self, y, t):
        # get seird
        s, e, i_a, i_m, i_s, i_h, i_c, h_r, h_c, h_d, c_r, c_d = self._get_seird_from_flat_y(y, return_removed=False)

        infectious_strength = np.sum(self.rel_beta_as * i_a + i_m + i_s, axis=1, keepdims=True)

        # solve seird equations
        if not self.contact_heterogeneous:
            ds = - 1 / self.n * self.infectious_func(t) * self.beta * infectious_strength * s
            de = 1 / self.n * self.infectious_func(t) * self.beta * infectious_strength * s - e / self.time_incubate
        else:
            ds = - self.contact_k * np.log1p(self.infectious_func(t) * self.beta * infectious_strength /
                                             (self.contact_k * self.n)) * s
            de = self.contact_k * np.log1p(self.infectious_func(t) * self.beta * infectious_strength /
                                           (self.contact_k * self.n)) * s - e / self.time_incubate

        di_a = self.prop_a * e / self.time_incubate - i_a / self.time_infectious
        di_m = self.prop_m * e / self.time_incubate - i_m / self.time_infectious
        di_s = self.prop_s * e / self.time_incubate - i_s / self.time_infectious

        di_h = self.prop_s_to_h * i_s / self.time_infectious - i_h / self.time_i_to_h
        di_c = self.prop_s_to_c * i_s / self.time_infectious - i_c / self.time_i_to_c

        dh_r = self.prop_h_to_r * i_h / self.time_i_to_h - h_r / self.time_h_to_r
        dh_c = self.prop_h_to_c * i_h / self.time_i_to_h - h_c / self.time_h_to_c
        dh_d = self.prop_h_to_d * i_h / self.time_i_to_h - h_d / self.time_h_to_d

        dc_r = self.prop_c_to_r * (h_c / self.time_h_to_c + i_c / self.time_i_to_c) - c_r / self.time_c_to_r
        dc_d = self.prop_c_to_d * (h_c / self.time_h_to_c + i_c / self.time_i_to_c) - c_d / self.time_c_to_d

        dr_a = i_a / self.time_infectious
        dr_m = i_m / self.time_infectious
        dr_h = h_r / self.time_h_to_r
        dr_c = c_r / self.time_c_to_r

        dd_h = h_d / self.time_h_to_d
        dd_c = c_d / self.time_c_to_d

        # concatenate
        dydt = np.concatenate([
            ds.reshape(self.nb_samples, self.nb_groups, 1),
            de.reshape(self.nb_samples, self.nb_groups, 1),
            di_a.reshape(self.nb_samples, self.nb_groups, 1),
            di_m.reshape(self.nb_samples, self.nb_groups, 1),
            di_s.reshape(self.nb_samples, self.nb_groups, 1),
            di_h.reshape(self.nb_samples, self.nb_groups, 1),
            di_c.reshape(self.nb_samples, self.nb_groups, 1),
            dh_r.reshape(self.nb_samples, self.nb_groups, 1),
            dh_c.reshape(self.nb_samples, self.nb_groups, 1),
            dh_d.reshape(self.nb_samples, self.nb_groups, 1),
            dc_r.reshape(self.nb_samples, self.nb_groups, 1),
            dc_d.reshape(self.nb_samples, self.nb_groups, 1),
            dr_a.reshape(self.nb_samples, self.nb_groups, 1),
            dr_m.reshape(self.nb_samples, self.nb_groups, 1),
            dr_h.reshape(self.nb_samples, self.nb_groups, 1),
            dr_c.reshape(self.nb_samples, self.nb_groups, 1),
            dd_h.reshape(self.nb_samples, self.nb_groups, 1),
            dd_c.reshape(self.nb_samples, self.nb_groups, 1)
        ], axis=-1).reshape(-1)

        return dydt

    def solve(self, t, y0=None, return_as_seird: bool = True, exclude_t0: bool = False):
        y0 = self.y0 if y0 is None else y0
        if not self._solved:
            y = odeint(self._ode, y0, t).reshape(-1, self.nb_samples, self.nb_groups, self.nb_states).clip(min=0)
            self._solution = y
            self._t = t
            self._solved = True
        else:
            if np.all(t != self._t) or np.all(y0 != self.y0):
                y = odeint(self._ode, y0, t).reshape(-1, self.nb_samples, self.nb_groups, self.nb_states).clip(min=0)
                self._t = t
                self._solution = y
            else:
                y = self._solution

        if exclude_t0:
            y = y[1:]

        if return_as_seird:
            s = y[:, :, :, 0]
            e = y[:, :, :, 1]
            i_a = y[:, :, :, 2]
            i_m = y[:, :, :, 3]
            i_s = y[:, :, :, 4]
            i_h = y[:, :, :, 5]
            i_c = y[:, :, :, 6]
            h_r = y[:, :, :, 7]
            h_c = y[:, :, :, 8]
            h_d = y[:, :, :, 9]
            c_r = y[:, :, :, 10]
            c_d = y[:, :, :, 11]
            r_a = y[:, :, :, 12]
            r_m = y[:, :, :, 13]
            r_h = y[:, :, :, 14]
            r_c = y[:, :, :, 15]
            d_h = y[:, :, :, 16]
            d_c = y[:, :, :, 17]
            return s, e, i_a, i_m, i_s, i_h, i_c, h_r, h_c, h_d, c_r, c_d, r_a, r_m, r_h, r_c, d_h, d_c
        return y

    def calculate_sir_posterior(self,
                                t0,
                                t_obs,
                                det_obs=None,
                                h_obs=None,
                                c_obs=None,
                                deaths_obs=None,
                                ratio_as_detected=0.,
                                ratio_m_detected=0.3,
                                ratio_s_detected=1.0,
                                ratio_resample: float = 0.1,
                                y0=None,
                                smoothing=1,
                                group_total: bool = False):
        # number of resamples
        m = int(np.round(self.nb_samples * ratio_resample))

        # cast variables
        t0 = np.asarray(t0)
        t_obs = np.asarray(t_obs)
        det_obs = None if det_obs is None else np.asarray(det_obs).reshape(-1, 1, 1).astype(int)
        h_obs = None if h_obs is None else np.asarray(h_obs).reshape(-1, 1, 1).astype(int)
        c_obs = None if c_obs is None else np.asarray(c_obs).reshape(-1, 1, 1).astype(int)
        deaths_obs = None if deaths_obs is None else np.asarray(deaths_obs).reshape(-1, 1, 1).astype(int)

        # assert shapes
        assert t0.ndim == 0, "t0 should be a scalar, not a vector"

        t = np.append(t0, t_obs)

        logging.info('Solving system')
        s, e, i_a, i_m, i_s, i_h, i_c, h_r, h_c, h_d, c_r, c_d, r_a, r_m, r_h, r_c, d_h, d_c = self.solve(t, y0, exclude_t0=True)

        detected = calculate_detected_cases(infected_asymptomatic=i_a,
                                            infected_mild=i_m,
                                            infected_severe=i_s + i_h + i_c + h_r + h_c + h_c + c_r + c_d,
                                            removed_asymptomatic=r_a,
                                            removed_mild=r_m,
                                            removed_severe=r_h + r_c + d_h + d_c,
                                            ratio_asymptomatic_detected=ratio_as_detected,
                                            ratio_mild_detected=ratio_m_detected,
                                            ratio_severe_detected=ratio_s_detected)

        h_tot = h_r + h_c + h_d
        c_tot = c_r + c_d
        d_tot = d_h + d_c

        # compare totals if needed
        if group_total:
            detected = np.sum(detected, axis=2, keepdims=True)
            h_tot = np.sum(h_tot, axis=2, keepdims=True)
            c_tot = np.sum(c_tot, axis=2, keepdims=True)
            d_tot = np.sum(d_tot, axis=2, keepdims=True)

        # model detected cases as poisson distribution y~P(lambda=detected_cases)
        logging.info('Calculating log weights')
        log_weights_detected = 0 if det_obs is None else _log_poisson(det_obs, detected)
        log_weights_hospital = 0 if h_obs is None else _log_poisson(h_obs, h_tot)
        log_weights_icu = 0 if c_obs is None else _log_poisson(c_obs, c_tot)
        log_weights_dead = 0 if deaths_obs is None else _log_poisson(deaths_obs, d_tot)

        # Free up memory at this point
        del s, e, i_a, i_m, i_s, i_h, i_c, h_r, h_c, h_d, c_r, c_d, r_a, r_m, r_h, r_c, d_h, d_c

        log_weights = log_weights_detected + log_weights_hospital + log_weights_icu + log_weights_dead
        weights = softmax(log_weights/smoothing)

        logging.info(f'log_weights_min: {log_weights.min()}')
        logging.info(f'log_weights_max: {log_weights.max()}')
        logging.info(f'Proportion weights above 0: {np.mean(weights > 0):.6}')
        logging.info(f'Proportion weights above 1E-20: {np.mean(weights > 1E-20):.6}')
        logging.info(f'Proportion weights above 1E-10: {np.mean(weights > 1E-10):.8}')
        logging.info(f'Proportion weights above 1E-3: {np.mean(weights > 1E-3):.10}')
        logging.info(f'Proportion weights above 1E-2: {np.mean(weights > 1E-2):.10}')
        logging.info(f'Proportion weights above 1E-1: {np.mean(weights > 1E-1):.10}')
        logging.info(f'Proportion weights above 0.5: {np.mean(weights > 0.5):.10}')

        # resample the sampled variables
        logging.info(f'Resampling {list(self.sample_vars.keys())} {m} times from {self.nb_samples} original samples')
        resample_indices = np.random.choice(self.nb_samples, m, p=weights)
        resample_vars = {}
        for key, value in self.sample_vars.items():
            resample_vars[key] = value[resample_indices]
        logging.info(f'Succesfully resampled {list(resample_vars.keys())}')

        # resample calculated variables
        logging.info(f'Resampling calculated variables {list(self.calculated_sample_vars.keys())}')
        calculated_resample_vars = {}
        for key, value in self.calculated_sample_vars.items():
            calculated_resample_vars[key] = value[resample_indices]
        logging.info(f'Succesfully resampled {list(resample_vars.keys())}')

        self.resample_vars = resample_vars
        self.calculated_resample_vars = calculated_resample_vars
        self.log_weights = log_weights
        self.weights = weights
        self.nb_resamples = m
        self.resample_indices = resample_indices

    def _get_seird_from_flat_y(self, y, return_removed=True):
        y = y.reshape(self.nb_samples, self.nb_groups, self.nb_states)
        # susceptible
        s = y[:, :, 0]

        # exposed
        e = y[:, :, 1]

        # infectious
        i_a = y[:, :, 2]
        i_m = y[:, :, 3]
        i_s = y[:, :, 4]

        # isolated
        i_h = y[:, :, 5]
        i_c = y[:, :, 6]

        # hospitalised
        h_r = y[:, :, 7]
        h_c = y[:, :, 8]
        h_d = y[:, :, 9]

        # critical care
        c_r = y[:, :, 10]
        c_d = y[:, :, 11]

        # removed
        r_a = y[:, :, 12]
        r_m = y[:, :, 13]
        r_h = y[:, :, 14]
        r_c = y[:, :, 15]

        # deceased
        d_h = y[:, :, 16]
        d_c = y[:, :, 17]

        if return_removed:
            return s, e, i_a, i_m, i_s, i_h, i_c, h_r, h_c, h_d, c_r, c_d, r_a, r_m, r_h, r_c, d_h, d_c
        else:
            return s, e, i_a, i_m, i_s, i_h, i_c, h_r, h_c, h_d, c_r, c_d


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
            continue
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
                continue
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
            raise ValueError(f'Variable {key} has too many dimension. Should be 0 or 2, got {value.ndim}')

    if not nb_samples:
        nb_samples = 1

    return nb_samples, (scalar_vars, group_vars, sample_vars)


def _log_poisson(k, l):
    out = k * np.log(l+1E-20) - l - gammaln(k+1)
    out = np.sum(out, axis=(0, 2))
    return out