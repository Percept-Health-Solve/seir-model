import numpy as np

from dataclasses import dataclass, field
from typing import Union, Iterable

from seir.cli import DistributionCLI, OdeCLI, LockdownCLI, SampleCLI, AgeCLI


@dataclass
class BaseSampleParams:
    nb_samples: int

    def _assert_param_shape(self, name, param, nb_groups: int = None):
        param = np.asarray(param)
        print(name, param.ndim)
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
    def from_default(cls, nb_samples: int):
        raise NotImplementedError

    @classmethod
    def samples_from_cli(cls, nb_samples: int, clis: Union[DistributionCLI, Iterable[DistributionCLI]]):
        if isinstance(clis, DistributionCLI):
            clis = [clis]
        kwargs = {}
        for cli in clis:
            cli_attrs = vars(cli).copy()
            for attr in cli_attrs:
                kwargs[attr] = cli.parse_attr(attr, size=(1, nb_samples))
        kwargs['nb_samples'] = nb_samples
        return cls(**kwargs)


@dataclass
class LockdownParams(BaseSampleParams):

    rel_beta_lockdown: Iterable[Union[float, np.ndarray]]
    rel_beta_period: Iterable[Union[float, np.ndarray]]

    @classmethod
    def from_default(cls, nb_samples: int):
        default_cli = LockdownCLI()
        return cls.samples_from_cli(nb_samples, default_cli)

    def __post_init__(self):
        self._assert_shapes()

    def _assert_shapes(self, nb_groups: int = None):
        for k, v in self.__dict__.items():
            for i, rel_beta in enumerate(v):
                self._assert_param_shape(f'{k}[{i}]', rel_beta, nb_groups)

        return True


@dataclass
class OdeParams(BaseSampleParams):
    r0: Union[float, np.ndarray]
    beta: Union[float, np.ndarray] = field(init=False)
    rel_beta_postlockdown: Union[float, np.ndarray]
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
        self._assert_shapes()

    def _assert_shapes(self, nb_groups: int = None):
        for k, v in self.__dict__.items():
            self._assert_param_shape(k, v, nb_groups)


    @classmethod
    def from_default(cls, nb_samples: int):
        default_cli = OdeCLI()
        return cls.samples_from_cli(nb_samples, default_cli)
