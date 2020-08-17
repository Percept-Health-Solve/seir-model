import numpy as np
from scipy.integrate import odeint

from dataclasses import dataclass, field
from typing import Union, List, Tuple

from seir.ode import BaseODE
from seir.data import TimestampData, CovidData


@dataclass
class ScipyOdeIntSolver:

    ode: BaseODE

    def solve(self, y0, t, return_full: bool = False, exclude_t0=False) -> Union[CovidData, Tuple[CovidData, np.ndarray]]:
        y0 = np.asarray(y0)
        t = np.asarray(t)

        def _ode_wrap(y, t):
            return self.ode(y, t).reshape(-1)

        y = odeint(_ode_wrap, y0.reshape(-1), t)
        y = y.reshape(-1, self.ode.nb_states, self.ode.nb_groups, self.ode.nb_samples)

        infected = np.sum(y[:, self.ode.infected_idx], axis=1)
        hospitalised = np.sum(y[:, self.ode.hospital_idx], axis=1)
        critical = np.sum(y[:, self.ode.critical_idx], axis=1)
        deaths = np.sum(y[:, self.ode.deaths_idx], axis=1)

        if exclude_t0:
            infected = infected[1:]
            hospitalised = hospitalised[1:]
            critical = critical[1:]
            deaths = deaths[1:]
            t = t[1:]

        solution = CovidData(
            nb_samples=self.ode.nb_samples,
            nb_groups=self.ode.nb_groups,
            infected=TimestampData(t, infected),
            hospitalised=TimestampData(t, hospitalised),
            critical=TimestampData(t, critical),
            deaths=TimestampData(t, deaths)
        )

        if return_full:
            return solution, y
        return solution




