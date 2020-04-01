import numpy as np
from scipy.integrate import odeint


def _assert_vectors(nb_groups, nb_infectious, vectors: list):
    for vector in vectors:
        assert vector.ndim == 2
        assert vector.shape[0] == nb_groups
        assert vector.shape[1] == nb_infectious


class NInfectiousModel:

    def __init__(self, nb_groups: int, nb_infectious: int, t_inc: float, alpha, q_se, q_ii, q_ir, q_id, beta, delta):
        alpha = np.asarray(alpha)
        q_se = np.asarray(q_se)
        q_ii = np.asarray(q_ii)
        q_ir = np.asarray(q_ir)
        q_id = np.asarray(q_id)
        beta = np.asarray(beta)
        delta = np.asarray(delta)

        if nb_groups == 1:
            q_ii = np.reshape(1, *q_ii.shape) if q_ii.ndim == 2 else q_ii
            q_ir = np.reshape(1, *q_ir.shape) if q_ir.ndim == 1 else q_ir
            q_id = np.reshape(1, *q_id.shape) if q_id.ndim == 1 else q_id
            beta = np.reshape(1, *beta.shape) if beta.ndim == 1 else beta
            delta = np.reshape(1, *delta.shape) if delta.ndim == 1 else delta
            alpha = np.reshape(1, *alpha.shape) if alpha.ndim == 1 else alpha

        # assert variables
        assert nb_groups > 0
        assert nb_infectious > 0
        assert t_inc > 0
        _assert_vectors(nb_groups, nb_infectious, [alpha, q_ir, q_id, beta, delta])
        assert q_se.ndim == 1
        assert q_se.shape[0] == nb_infectious
        assert q_ii.ndim == 3
        assert q_ii.shape[0] == nb_groups
        assert q_ii.shape[1] == q_ii.shape[2] == nb_infectious

        # ensure variables maintain constraints
        assert np.all(np.sum(alpha, axis=1) == 1)
        assert np.all(np.sum(q_ii, axis=1) == 0)

        self.nb_groups = nb_groups
        self.nb_infectious = nb_infectious
        self.t_inc = t_inc
        self.alpha = alpha
        self.q_se = q_se
        self.q_ii = q_ii
        self.q_ir = q_ir
        self.q_id = q_id
        self.beta = beta
        self.delta = delta
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

    def ode(self, y, t, N):
        idx_s = self.y_idx_dict['s']
        idx_e = self.y_idx_dict['e']
        idx_i = self.y_idx_dict['i']
        # idx_r = self.y_idx_dict['r']

        s = y[:idx_s].reshape(self.nb_groups, 1)
        e = y[idx_s:idx_e].reshape(self.nb_groups, 1)
        i = y[idx_e:idx_i].reshape(self.nb_groups, self.nb_infectious)
        # r = y[idx_i:idx_r].reshape(self.nb_groups, self.nb_infectious)
        # d = y[idx_r:].reshape(self.nb_groups, self.nb_infectious)

        dsdt = - 1 / N * self.q_se.dot(np.sum(i, axis=0)) * s
        dedt = 1 / N * self.q_se.dot(np.sum(i, axis=0)) * s - e / self.t_inc
        didt = self.alpha * e / self.t_inc \
               - np.array([self.q_ii[idx].dot(self.delta[idx] * i[idx]) for idx in range(self.nb_groups)]) \
               - self.q_ir * (1 - self.delta) * (1 - self.beta) * i \
               - self.q_id * (1 - self.delta) * self.beta * i
        drdt = self.q_ir * (1 - self.delta) * (1 - self.beta) * i
        dddt = self.q_id * (1 - self.delta) * self.beta * i

        dydt = np.concatenate([
            dsdt.reshape(-1),
            dedt.reshape(-1),
            didt.reshape(-1),
            drdt.reshape(-1),
            dddt.reshape(-1)
        ])

        return dydt

    def solve(self, init_vectors: dict, t):
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

        s_t = solution[:, :self.y_idx_dict['s']]
        e_t = solution[:, self.y_idx_dict['s']: self.y_idx_dict['e']]
        i_t = solution[:, self.y_idx_dict['e']: self.y_idx_dict['i']]
        r_t = solution[:, self.y_idx_dict['i']: self.y_idx_dict['r']]
        d_t = solution[:, self.y_idx_dict['r']:]

        s_t = s_t.reshape(-1, self.nb_groups)
        e_t = e_t.reshape(-1, self.nb_groups)
        if self.nb_groups > 1:
            i_t = i_t.reshape(-1, self.nb_groups, self.nb_infectious)
            r_t = r_t.reshape(-1, self.nb_groups, self.nb_infectious)
            d_t = d_t.reshape(-1, self.nb_groups, self.nb_infectious)
        else:
            i_t = i_t.reshape(-1, self.nb_infectious)
            r_t = r_t.reshape(-1, self.nb_infectious)
            d_t = d_t.reshape(-1, self.nb_infectious)

        out = {
            's_t': s_t,
            'e_t': e_t,
            'i_t': i_t,
            'r_t': r_t,
            'd_t': d_t
        }

        self._N = N
        self._N_g = N_g
        self._solved = True
        self._solution = out

        return out

    @property
    def N(self):
        # TODO: Add ability to solve N given some initial vectors
        if self._solved:
            return self._N
        else:
            raise ValueError('Attempted to return N when model has not been solved!')

    @ property
    def N_g(self):
        # TODO: Add ability to solve N_g given some initial vectors
        if self._solved:
            return self._N_g
        else:
            raise ValueError('Attempted to return N_g when model has not been solved!')

    @ property
    def solution(self):
        if self._solved:
            return self._solution
        else:
            raise ValueError('Attempted to return solution when model has not been solved!')