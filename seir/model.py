import numpy as np
import pandas as pd

from scipy.integrate import odeint


def _assert_vectors(nb_groups, nb_infectious, vectors: list):
    for vector in vectors:
        assert vector.ndim == 2
        assert vector.shape[0] == nb_groups
        assert vector.shape[1] == nb_infectious


class NInfectiousModel:

    def __init__(self,
                 nb_groups: int,
                 nb_infectious: int,
                 t_inc: float,
                 alpha,
                 q_se,
                 q_ii,
                 q_ir,
                 q_id,
                 rho_delta,
                 rho_beta,
                 infectious_func=None,
                 imported_func=None,
                 extend_vars=True):
        """Creates a generalised multi-population-group SEIR model with multiple infectious states I and two removed
        populations R (recovering and isolated) and D (dying and isolated). The model ignores the social dynamics
        between population groups and assumes that any member of an infectious sub-population can infect all members
        of the susceptible population.

        Params
        ------

        nb_groups: int
            The number of population groups to keep track of. The individual population groups can have their own
            transition dynamics within the model.

        nb_infectious: int
            The number of infected states in the model.

        t_inc: float
            The incubation time of the disease, transitioning between the exposed and infected states.

        alpha: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The proportion of the population leaving the E state and entering each of the infected states.
            The rows of this array must sum to 1. Can only have shape (nb_infectious,) if extend_vars is True.

        q_se: [nb_infectious X 1] array
            The transition rates from the S state to the E state. This is interpreted as the number of secondary
            infections on the susceptible population caused by a member of the corresponding infectious state.

        q_ii: [nb_group X nb_infectious X nb_infectious] or [nb_infectious X nb_infectious] array
            The transition rates between infectious states. The columns of the each q_ii[j] matrix must sum
            to 0 for the population to be preserved. Can only have shape (nb_infectious, nb_infectious) if extend_vars
            is True.

        q_ir: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The transition rates from the I states to the R states for each population group. Can only have shape
            (nb_infectious) if extend_vars is True.

        q_id: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The transition from the I states to the R states for each population group. Can only have shape
            (nb_infectious,) if extend_vars is True.

        rho_delta: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The proportion of I states undergoing transitions from one I state to another I state for each population
            group. Conversely, 1 - rho_delta represents the proportion of the population that are transitioning to the
            removed states R and D. Can only have shape (nb_infectious,) if extend_vars is True.

        rho_beta: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The mortality rate of the I states undergoing a transition from I to D. Can only have shape (nb_infectious,)
            if extend_vars is True.

        infectious_func: callable, default=None
            A function that takes one input (time) and returns a multiplicative factor on the infectious rate. This can
            be used to model quarantines or periods of high infectivity.

        imported_func: callable, default=None
            A function that takes one input argument (time) and returns the rate of change in infections for that time.
            Used to seed the model with imported cases.

        extend_vars: bool, default=True
            Whether to assume that the given transition rates apply to all population groups. Useful when data on
            certain transition rates between population groups are unknown but data on average transitions rates are
            available.
        """
        alpha = np.asarray(alpha)
        q_se = np.asarray(q_se)
        q_ii = np.asarray(q_ii)
        q_ir = np.asarray(q_ir)
        q_id = np.asarray(q_id)
        rho_beta = np.asarray(rho_beta)
        rho_delta = np.asarray(rho_delta)

        if nb_groups == 1:
            q_ii = np.reshape(1, *q_ii.shape) if q_ii.ndim == 2 else q_ii
            q_ir = np.reshape(1, *q_ir.shape) if q_ir.ndim == 1 else q_ir
            q_id = np.reshape(1, *q_id.shape) if q_id.ndim == 1 else q_id
            rho_beta = np.reshape(1, *rho_beta.shape) if rho_beta.ndim == 1 else rho_beta
            rho_delta = np.reshape(1, *rho_delta.shape) if rho_delta.ndim == 1 else rho_delta
            alpha = np.reshape(1, *alpha.shape) if alpha.ndim == 1 else alpha

        if extend_vars is True:
            alpha = np.broadcast_to(alpha, (nb_groups, nb_infectious)) if alpha.ndim == 1 else alpha
            q_ii = np.broadcast_to(q_ii, (nb_groups, nb_infectious, nb_infectious)) if q_ii.ndim == 2 else q_ii
            q_ir = np.broadcast_to(q_ir, (nb_groups, nb_infectious)) if q_ir.ndim == 1 else q_ir
            q_id = np.broadcast_to(q_id, (nb_groups, nb_infectious)) if q_id.ndim == 1 else q_id
            rho_delta = np.broadcast_to(rho_delta, (nb_groups, nb_infectious)) if rho_delta.ndim == 1 else rho_delta
            rho_beta = np.broadcast_to(rho_beta, (nb_groups, nb_infectious)) if rho_beta.ndim == 1 else rho_beta

        # assert variables
        assert nb_groups > 0
        assert nb_infectious > 0
        assert t_inc > 0
        _assert_vectors(nb_groups, nb_infectious, [alpha, q_ir, q_id, rho_beta, rho_delta])
        assert q_se.ndim == 1
        assert q_se.shape[0] == nb_infectious
        assert q_ii.ndim == 3
        assert q_ii.shape[0] == nb_groups
        assert q_ii.shape[1] == q_ii.shape[2] == nb_infectious

        # ensure variables maintain constraints
        assert np.all(np.abs(np.sum(alpha, axis=1) - 1) <= 1E-15)
        assert np.all(np.abs(np.sum(q_ii, axis=1)) <= 1E-15)

        # check infectious func
        if infectious_func is not None:
            assert callable(infectious_func), "infectious_func is not callable"
        else:
            infectious_func = lambda x: 1

        # check imported func
        if imported_func is not None:
            assert callable(imported_func), "imported_func is not callable"
        else:
            imported_func = lambda x: 0

        # rho_beta goes through a function before being used in the ode
        beta = np.array([
            rho_beta[g] * q_ir[g]
            / (rho_beta[g] * q_ir[g] + (1 - rho_beta[g]) * q_id[g] + 1E-20)
            for g in range(nb_groups)
        ])
        # in the case where element i of q_ir[g]_i is zero and rho_beta[g]_i is one, the result should be 1
        fix_args = np.argwhere((rho_beta == 1) & (q_ir == 0))
        for arg in fix_args:
            beta[arg[0], arg[1]] = 1

        # rho_delta goes through a function before being used in the ode:
        delta = np.array([
            rho_delta[g] * (q_ir[g] * (1 - beta[g]) + q_id[g] * beta[g])
            / (np.diag(q_ii[g]) * (1 - rho_delta[g]) + (q_ir[g] * (1 - beta[g]) + q_id[g] * beta[g]) * rho_delta[g] + 1E-20)
            for g in range(nb_groups)
        ])
        # in the case where element i of q_ir[g]_i and q_id[g] is zero and rho_delta[g]_i is one, the result should be 1
        fix_args = np.argwhere((rho_delta == 1) & ((q_ir * (1 - beta) + q_id * beta) == 0))
        for arg in fix_args:
            delta[arg[0], arg[1]] = 1

        # set public properties
        self.nb_groups = nb_groups
        self.nb_infectious = nb_infectious
        self.t_inc = t_inc
        self.alpha = alpha
        self.q_se = q_se
        self.q_ii = q_ii
        self.q_ir = q_ir
        self.q_id = q_id
        self.rho_beta = rho_beta
        self.beta = beta
        self.rho_delta = rho_delta
        self.delta = delta
        self.infectious_func = infectious_func
        self.imported_func = imported_func

        # set private properties
        self._solved = False
        self._solution = None
        self._N = 0
        self._N_g = 0
        self.y_idx_dict = {
            's': self.nb_groups,
            'e': self.nb_groups * 2,
            'i': self.nb_groups * 2 + self.nb_groups * self.nb_infectious,
            'r': self.nb_groups * 2 + self.nb_groups * self.nb_infectious * 2
        }

    @property
    def N(self):
        # TODO: Add ability to solve N given some initial vectors
        if self._solved:
            return self._N
        else:
            raise ValueError('Attempted to return N when model has not been solved!')

    @property
    def N_g(self):
        # TODO: Add ability to solve N_g given some initial vectors
        if self._solved:
            return self._N_g
        else:
            raise ValueError('Attempted to return N_g when model has not been solved!')

    @property
    def solution(self):
        if self._solved:
            return self._solution
        else:
            raise ValueError('Attempted to return solution when model has not been solved!')

    @property
    def r_0_eff(self):
        out = [
            self.q_se / (
                    np.diag(self.q_ii[g]) * self.delta[g]
                    + self.q_ir[g] * (1 - self.beta[g]) * (1 - self.delta[g])
                    + self.q_id[g] * self.beta[g] * (1 - self.delta[g])
                    + 1E-20
            ) * (self.N_g[g] > 0) for g in range(self.nb_groups)
        ]
        return np.array(out)

    @property
    def r_0(self):
        r_0_eff = self.r_0_eff
        out = 1 / self.N * np.sum([self.N_g[g] * self.alpha[g].dot(r_0_eff[g]) for g in range(self.nb_groups)])
        return out

    def ode(self, y, t, N):
        # find indices for various vectors in flat vector
        idx_s = self.y_idx_dict['s']
        idx_e = self.y_idx_dict['e']
        idx_i = self.y_idx_dict['i']
        # idx_r = self.y_idx_dict['r']

        # assign quantities from flat vector y based on indices
        s = y[:idx_s].reshape(self.nb_groups, 1)
        e = y[idx_s:idx_e].reshape(self.nb_groups, 1)
        i = y[idx_e:idx_i].reshape(self.nb_groups, self.nb_infectious)
        # r = y[idx_i:idx_r].reshape(self.nb_groups, self.nb_infectious)
        # d = y[idx_r:].reshape(self.nb_groups, self.nb_infectious)

        # define ode
        dsdt = - 1 / N * (self.infectious_func(t) * self.q_se).dot(np.sum(i, axis=0)) * s
        dedt = 1 / N * (self.infectious_func(t) * self.q_se).dot(np.sum(i, axis=0)) * s - e / self.t_inc
        didt = self.alpha * e / self.t_inc \
            - np.array([self.q_ii[idx].dot(self.delta[idx] * i[idx]) for idx in range(self.nb_groups)]) \
            - self.q_ir * (1 - self.delta) * (1 - self.beta) * i \
            - self.q_id * (1 - self.delta) * self.beta * i \
            + self.imported_func(t)
        drdt = self.q_ir * (1 - self.delta) * (1 - self.beta) * i
        dddt = self.q_id * (1 - self.delta) * self.beta * i

        # reshape ode back to flat vector
        dydt = np.concatenate([
            dsdt.reshape(-1),
            dedt.reshape(-1),
            didt.reshape(-1),
            drdt.reshape(-1),
            dddt.reshape(-1)
        ])

        return dydt

    def solve(self, init_vectors: dict, t, to_csv: bool = False, fp: str = None) -> tuple:
        """Solve the SEIR equations for this model.

        Params
        ------

        init_vectors: dict
            A dictionary representing the initial states of the S, E, I, R, and D states of our model, corresponding
            to keys 's_0', 'e_0', 'i_0', 'r_0', and 'd_0', respectively. If a key is not found, the model assumes
            a zero vector for the corresponding intiial state.

        t: array
            The time values to solve the ODE over.

        to_csv: bool (default = False)
            Sets where to save the solution to a csv.

        fp: str (default = None)
            The filepath of the desired csv.

        Returns
        -------

        The function returns a tuple comprising the following elements.

        s_t: numpy.ndarray
            The solution of the S state with shape (len(t), nb_groups)

        e_t: numpy.ndarray
            The solution of the E state with shape (len(t), nb_groups)

        i_t: numpy.ndarray
            The solution of the I state with shape (len(t), nb_groups, nb_infectious)

        r_t: numpy.ndarray
            The solution of the R state with shape(len(t), nb_groups, nb_infectious)

        d_t: numpy.ndarray
            The solution of the D state with shape(len(t), nb_groups, nb_infectious)
        """
        if to_csv and fp is None:
            raise ValueError("Attempting to save solution but no file path is specified.")
        if not to_csv and fp is not None:
            raise Warning('File path given but to_csv = False')

        s_0, e_0, i_0, r_0, d_0 = self._parse_init_vectors(init_vectors)

        assert s_0.shape == (self.nb_groups,) or s_0.shape == (self.nb_groups, 1)
        assert e_0.shape == (self.nb_groups,) or e_0.shape == (self.nb_groups, 1)
        if self.nb_groups > 1:
            assert i_0.shape == (self.nb_groups, self.nb_infectious)
            assert r_0.shape == (self.nb_groups, self.nb_infectious)
            assert d_0.shape == (self.nb_groups, self.nb_infectious)
        elif self.nb_groups == 1:
            assert i_0.shape == (self.nb_infectious,)
            assert r_0.shape == (self.nb_infectious,)
            assert d_0.shape == (self.nb_infectious,)

        y_0 = np.concatenate([
            s_0.reshape(-1),
            e_0.reshape(-1),
            i_0.reshape(-1),
            r_0.reshape(-1),
            d_0.reshape(-1)
        ])

        N = np.sum(y_0)
        N_g = s_0.reshape(self.nb_groups) + e_0.reshape(self.nb_groups) + np.sum(i_0 + r_0 + d_0, axis=1)

        solution = odeint(self.ode, y_0, t, args=(N,))

        if to_csv:
            self._to_csv(solution, t, fp)

        s_t = solution[:, :self.y_idx_dict['s']]
        e_t = solution[:, self.y_idx_dict['s']: self.y_idx_dict['e']]
        i_t = solution[:, self.y_idx_dict['e']: self.y_idx_dict['i']]
        r_t = solution[:, self.y_idx_dict['i']: self.y_idx_dict['r']]
        d_t = solution[:, self.y_idx_dict['r']:]

        s_t = s_t.reshape(-1, self.nb_groups)
        e_t = e_t.reshape(-1, self.nb_groups)
        i_t = i_t.reshape(-1, self.nb_groups, self.nb_infectious)
        r_t = r_t.reshape(-1, self.nb_groups, self.nb_infectious)
        d_t = d_t.reshape(-1, self.nb_groups, self.nb_infectious)

        self._N = N
        self._N_g = N_g
        self._solved = True
        self._solution = (s_t, e_t, i_t, r_t, d_t)

        return s_t, e_t, i_t, r_t, d_t

    def _to_csv(self, solution, t, fp):
        s_cols = [f'S_{i}' for i in range(self.nb_groups)]
        e_cols = [f'E_{i}' for i in range(self.nb_groups)]
        i_cols = [f'I_{i}_{j}' for i in range(self.nb_groups) for j in range(self.nb_infectious)]
        r_cols = [f'R_{i}_{j}' for i in range(self.nb_groups) for j in range(self.nb_infectious)]
        d_cols = [f'D_{i}_{j}' for i in range(self.nb_groups) for j in range(self.nb_infectious)]
        cols = s_cols + e_cols + i_cols + r_cols + d_cols
        df = pd.DataFrame(solution, columns=cols)
        df.insert(0, 'Day', t)
        df.to_csv(fp, index=False)

    def _parse_init_vectors(self, init_vectors: dict) -> tuple:
        s_0 = init_vectors.get('s_0')
        e_0 = init_vectors.get('e_0')
        i_0 = init_vectors.get('i_0')
        r_0 = init_vectors.get('r_0')
        d_0 = init_vectors.get('d_0')

        s_0 = np.zeros(self.nb_groups) if s_0 is None else np.asarray(s_0)
        e_0 = np.zeros(self.nb_groups) if e_0 is None else np.asarray(e_0)
        i_0 = np.zeros((self.nb_groups, self.nb_infectious)) if i_0 is None else np.asarray(i_0)
        r_0 = np.zeros((self.nb_groups, self.nb_infectious)) if r_0 is None else np.asarray(r_0)
        d_0 = np.zeros((self.nb_groups, self.nb_infectious)) if d_0 is None else np.asarray(d_0)

        return s_0, e_0, i_0, r_0, d_0
