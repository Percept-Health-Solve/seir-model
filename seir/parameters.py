import numpy as np

from dataclasses import dataclass, field
from typing import Union, Iterable, List

from seir.cli import BaseDistributionCLI, OdeParamCLI, LockdownCLI, MetaCLI, FittingCLI
from seir.defaults import NB_SAMPLES_DEFAULT


@dataclass
class BaseParams:

    @classmethod
    def from_default(cls):
        raise NotImplementedError

    @classmethod
    def sample_from_cli(cls, cli):
        raise NotImplementedError


@dataclass
class BaseSampleParams(BaseParams):
    nb_samples: int

    def _assert_param_shape(self, name, param, nb_groups: int = None):
        param = np.asarray(param)
        assert param.ndim == 0 or param.ndim == 2, \
            f"The number of dimensions of the values in `rel_beta_lockdown` should be 0 (a float) or 2 (a " \
            f"sampled/group vector). Found dimension {param.ndim} in `{name}'."
        if param.ndim == 2:
            assert param.shape[-1] == self.nb_samples or param.shape[-1] == 1, \
                f"Vector in '{name}' should take shape (?, {self.nb_samples}) or (?, 1). Found shape {param.shape} " \
                f"instead."
        if param.ndim == 2 and nb_groups is not None:
            assert param.shape[0] == 1 or param.shape[0] == nb_groups, \
                f"Vector in '{name}' should take shape ({nb_groups}, ?) or (1, ?). Got {param.shape} instead."

    @classmethod
    def sample_from_cli(cls,
                        cli: Union[BaseDistributionCLI, Iterable[BaseDistributionCLI]],
                        nb_samples: int = NB_SAMPLES_DEFAULT):
        if isinstance(cli, BaseDistributionCLI):
            cli = [cli]
        kwargs = {}
        for cli in cli:
            cli_attrs = vars(cli).copy()
            for attr in cli_attrs:
                kwargs[attr] = cli.sample_attr(attr, size=(1, nb_samples))
        kwargs['nb_samples'] = nb_samples
        return cls(**kwargs)


@dataclass
class LockdownParams(BaseSampleParams):

    rel_beta_lockdown: List[Union[float, np.ndarray]]
    rel_beta_period: List[Union[float, np.ndarray]]
    cum_periods: Union[float, np.ndarray] = field(init=False)

    def __post_init__(self):
        assert len(self.rel_beta_period) == len(self.rel_beta_lockdown), \
            "Mismatched number of relative beta periods to relative lockdown values."
        self._assert_shapes()
        self.cum_periods = np.cumsum(self.rel_beta_period, axis=0)

    def _assert_shapes(self, nb_groups: int = None):
        for k, v in self.__dict__.items():
            if k == 'nb_samples':
                continue
            for i, rel_beta in enumerate(v):
                self._assert_param_shape(f'{k}[{i}]', rel_beta, nb_groups)
        return True

    @classmethod
    def from_default(cls):
        default_cli = LockdownCLI()
        return cls.sample_from_cli(default_cli)


@dataclass
class OdeParams(BaseSampleParams):
    r0: Union[float, np.ndarray]
    beta: Union[float, np.ndarray] = field(init=False)
    rel_beta_asymptomatic: Union[float, np.ndarray]
    prop_a: Union[float, np.ndarray]
    prop_m: Union[float, np.ndarray] = field(init=False)
    prop_s: Union[float, np.ndarray]
    prop_s_to_h: Union[float, np.ndarray]
    prop_s_to_c: Union[float, np.ndarray] = field(init=False)
    prop_h_to_c: Union[float, np.ndarray]
    prop_h_to_d: Union[float, np.ndarray]
    prop_h_to_r: Union[float, np.ndarray] = field(init=False)
    prop_c_to_d: Union[float, np.ndarray]
    prop_c_to_r: Union[float, np.ndarray] = field(init=False)
    time_incubate: Union[float, np.ndarray]
    time_infectious: Union[float, np.ndarray]
    time_s_to_h: Union[float, np.ndarray]
    time_s_to_c: Union[float, np.ndarray]
    time_rsh_to_h: Union[float, np.ndarray] = field(init=False)
    time_rsc_to_c: Union[float, np.ndarray] = field(init=False)
    time_h_to_c: Union[float, np.ndarray]
    time_h_to_d: Union[float, np.ndarray]
    time_h_to_r: Union[float, np.ndarray]
    time_c_to_d: Union[float, np.ndarray]
    time_c_to_r: Union[float, np.ndarray]
    contact_k: Union[float, np.ndarray]
    prop_e0: Union[float, np.ndarray]

    def __post_init__(self):
        self.beta = self.r0 / self.time_infectious
        self.prop_m = 1 - self.prop_a - self.prop_s
        self.prop_s_to_c = 1 - self.prop_s_to_h
        self.prop_h_to_r = 1 - self.prop_h_to_c - self.prop_h_to_d
        self.prop_c_to_r = 1 - self.prop_c_to_d
        self.time_rsh_to_h = self.time_s_to_h - self.time_infectious
        self.time_rsc_to_c = self.time_s_to_c - self.time_infectious
        self._assert_shapes()

    def _assert_shapes(self, nb_groups: int = None):
        for k, v in self.__dict__.items():
            self._assert_param_shape(k, v, nb_groups)

    @classmethod
    def from_default(cls):
        default_cli = OdeParamCLI()
        return cls.sample_from_cli(default_cli)


@dataclass
class MetaParams(BaseParams):

    nb_samples: int
    nb_groups: int

    @classmethod
    def from_default(cls):
        default_cli = MetaCLI()
        return cls.sample_from_cli(default_cli)

    @classmethod
    def sample_from_cli(cls, cli: MetaCLI):
        nb_groups = 9 if cli.age_heterogeneity else 1
        return cls(nb_samples=cli.nb_samples,
                   nb_groups=nb_groups)


@dataclass
class FittingParams(BaseParams):

    nb_runs: int = 1
    ratio_resample: float = 0.05
    fit_totals: bool = True
    fit_deaths: bool = False
    fit_recovered: bool = False
    fit_infected: bool = False
    fit_hospitalised: bool = False
    fit_critical: bool = False

    @classmethod
    def from_default(cls):
        default_cli = FittingCLI()
        return cls.sample_from_cli(default_cli)

    @classmethod
    def sample_from_cli(cls, cli: FittingCLI):
        return cls(
            nb_runs=cli.nb_runs,
            ratio_resample=cli.ratio_resample,
            fit_totals=cli.fit_totals,
            fit_deaths=cli.fit_deaths,
            fit_recovered=cli.fit_recovered,
            fit_infected=cli.fit_infected,
            fit_hospitalised=cli.fit_hospitalised,
            fit_critical=cli.fit_critical
        )
