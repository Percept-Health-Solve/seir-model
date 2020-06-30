import numpy as np

from dataclasses import dataclass, field

from seir.ode_parameters import LockdownParams, OdeParams


class BaseODE:

    def __call__(self, y, t):
        raise NotImplementedError


@dataclass
class CovidODE(BaseODE):

    nb_samples: int = field(init=False)
    nb_groups: int = field(init=False)
    age_heterogeneity: bool
    lockdown_params: LockdownParams
    ode_params: OdeParams

    def __post_init__(self):
        assert self.lockdown_params.nb_samples == self.ode_params.nb_samples
        self.nb_samples = self.lockdown_params.nb_samples
        self.nb_groups = 9 if self.age_heterogeneity else 1
        self._assert_param_age_groups()

    def _assert_param_age_group(self):
        self.lockdown_params._assert_shapes(self.nb_groups)
        self.ode_params._assert_shapes(self.nb_groups)

    def __call__(self, y, t):
        pass


