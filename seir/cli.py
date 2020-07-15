import numpy as np
import json

from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Union

from seir.defaults import (
    NB_SAMPLES_DEFAULT,
    NB_RUNS_DEFAULT,
    AGE_GROUPS_DEFAULT,
    R0_DEFAULT,
    REL_BETA_LOCKDOWN_DEFAULT,
    REL_BETA_PERIOD_DEFAULT,
    REL_BETA_ASYMPTOMATIC_DEFAULT,
    PROP_A_DEFAULT,
    PROP_S_DEFAULT,
    PROP_S_TO_H_DEFAULT,
    PROP_H_TO_C_DEFAULT,
    PROP_H_TO_D_DEFAULT,
    PROP_C_TO_D_DEFAULT,
    TIME_INCUBATE_DEFAULT,
    TIME_INFECTIOUS_DEFAULT,
    TIME_S_TO_H_DEFAULT,
    TIME_S_TO_C_DEFAULT,
    TIME_H_TO_C_DEFAULT,
    TIME_H_TO_R_DEFAULT,
    TIME_H_TO_D_DEFAULT,
    TIME_C_TO_R_DEFAULT,
    TIME_C_TO_D_DEFAULT,
    CONTACT_K_DEFAULT,
    PROP_E0_DEFAULT
)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


def _sample_cli_attr(attr, nb_groups, nb_samples) -> np.ndarray:
    if len(attr) == 1:
        return np.asarray(attr[0])
    elif len(attr) == 2:
        return np.random.uniform(attr[0], attr[1], size=(nb_groups, nb_samples))
    else:
        raise ValueError(f"Uniform distribution should have two values, a lower and upper bound. Got {len(attr)} "
                         f"number of parameters instead.")


class BaseCLI:

    def save_to_json(self, path: Union[str, Path]):
        if not isinstance(path, Path):
            path = Path(path)
        # TODO: flesh this out


class BaseDistributionCLI(BaseCLI):

    _defaults_dict = {}

    def __post_init__(self):
        if not self._defaults_dict.keys() == self.__dict__.keys():
            raise NotImplementedError("CLI objects _defaults_dict should contain the default values for all "
                                      "attributes.")
        self._set_defaults()

    def _set_defaults(self):
        self_vars = self.__dict__
        for k in self_vars:
            self_vars[k] = self._defaults_dict.get(k, None) if self_vars[k] is None else self_vars[k]
        return True

    def sample_attr(self, attr: str, nb_groups: int = 1, nb_samples: int = 1) -> np.ndarray:
        """
        Samples an attribute of the cli to the required scalar or uniform distribution.
        :value attr: Attribute of the CLI to parse.
        :return parsed_attr: Returns the appropriate sampled argument.
        """
        attr_val = getattr(self, attr)
        try:
            if (
                    hasattr(self.__dataclass_fields__[attr].type, '__origin__')
                    and issubclass(self.__dataclass_fields__[attr].type.__origin__, List)
            ):
                if self.__dataclass_fields__[attr].metadata.get('action', None) == 'append':
                    return np.asarray([[_sample_cli_attr(x, nb_groups, nb_samples)] for x in attr_val])
                return np.asarray(_sample_cli_attr(attr_val, nb_groups, nb_samples))
            return attr_val
        except Exception as e:
            raise ValueError(f"Attribute '{attr}' failed to be parsed. Raised exception '{e}'.")


@dataclass
class LockdownCLI(BaseDistributionCLI):

    _defaults_dict = {
        'rel_beta_lockdown': REL_BETA_LOCKDOWN_DEFAULT,
        'rel_beta_period': REL_BETA_PERIOD_DEFAULT
    }

    rel_beta_lockdown: List[float] = list_field(
        default=None,
        metadata={
            "help": "Bounds of relative beta uniform prior distributions corresponding to relative beta "
                    "periods. Used to inform the behavior of Rt during periods of lockdown. Negative lower bound "
                    "implies using the previous sample as a minimum (minus the negative value). Can be called multiple "
                    "times to create multiple successive lockdown levels.",
            "action": "append"
        }
    )

    rel_beta_period: List[float] = list_field(
        default=None,
        metadata={
            "help": "Length of each period for which the relative beta's apply, in days, after the start of lockdown."
                    " Can be called multiple times to define multiple successive lockdown periods occuring one after "
                    "the other.",
            "action": "append"
        }
    )

    def sample_attr(self, attr: str, nb_groups: int = 1, nb_samples: int = 1) -> List[np.ndarray]:
        attr_val = getattr(self, attr)
        outputs = [_sample_cli_attr(attr_val[0], nb_groups, nb_samples)]
        for i in range(1, len(attr_val)):
            if len(attr_val[i]) == 1:
                outputs.append(np.asarray(attr_val[i][0]))
            elif len(attr_val[i]) == 2:
                if attr_val[i][0] < 0:
                    outputs.append(np.random.uniform(outputs[i-1] - abs(attr_val[i][0]), attr_val[i][1],
                                                     size=(nb_groups, nb_samples)))
                else:
                    outputs.append(np.random.uniform(attr_val[i][0], attr_val[i][1], size=(nb_groups, nb_samples)))
        return outputs


@dataclass
class OdeParamCLI(BaseDistributionCLI):

    _defaults_dict = {
        'r0': R0_DEFAULT,
        'rel_beta_asymptomatic': REL_BETA_ASYMPTOMATIC_DEFAULT,
        'prop_a': PROP_A_DEFAULT,
        'prop_s': PROP_S_DEFAULT,
        'prop_s_to_h': PROP_S_TO_H_DEFAULT,
        'prop_h_to_c': PROP_H_TO_C_DEFAULT,
        'prop_h_to_d': PROP_H_TO_D_DEFAULT,
        'prop_c_to_d': PROP_C_TO_D_DEFAULT,
        'time_incubate': TIME_INCUBATE_DEFAULT,
        'time_infectious': TIME_INFECTIOUS_DEFAULT,
        'time_s_to_h': TIME_S_TO_H_DEFAULT,
        'time_s_to_c': TIME_S_TO_C_DEFAULT,
        'time_h_to_c': TIME_H_TO_C_DEFAULT,
        'time_h_to_r': TIME_H_TO_R_DEFAULT,
        'time_h_to_d': TIME_H_TO_D_DEFAULT,
        'time_c_to_r': TIME_C_TO_R_DEFAULT,
        'time_c_to_d': TIME_C_TO_D_DEFAULT,
        'contact_k': CONTACT_K_DEFAULT,
    }

    r0: List[float] = list_field(
        default=None,
        metadata={
            "help": "Basic reproductive number r0. Single input defines a scalar value, two inputs define a "
                    "Uniform prior."
        }
    )

    rel_beta_asymptomatic: List[float] = list_field(
        default=None,
        metadata={
            "help": "The relative infectivity strength of asymptomatic cases. Single input defines a scalar value, two"
                    " inputs define a Uniform prior."
        }
    )

    prop_a: List[float] = list_field(
        default=None,
        metadata={
            "help": "Proportion of asymptomatic infected individuals. Single input defines a scalar, two inputs define "
                    "a Uniform prior."
        }
    )

    prop_s: List[float] = list_field(
        default=None,
        metadata={
            "help": "Proportion of symptomatic individuals (1 - prop_a) that experience severe symptoms. Defaults to "
                    "attack rates defined by Ferguson et. al. (see References documentation). ",
            "action": "append"
        }
    )

    prop_s_to_h: List[float] = list_field(
        default=None,
        metadata={
            "help": "Propotion of severe cases that will be admitted to general hospital. The rest will present "
                    "directly to ICU. Defaults to calculations based off WC data (see Data documentation).",
            "action": "append"
        }
    )

    prop_h_to_c: List[float] = list_field(
        default=None,
        metadata={
            "help": "Proportion of general hospital cases expected to be transferred to critical care. Defaults to "
                    "calculations based off WC data (see Data documentation).",
            "action": "append"
        }
    )

    prop_h_to_d: List[float] = list_field(
        default=None,
        metadata={
            "help": "Proportion of general hospital cases that are expected to die. Defaults to calculations based off "
                    "WC data (see Data documentation).",
            "action": "append"
        }
    )

    prop_c_to_d: List[float] = list_field(
        default=None,
        metadata={
            "help": "Proportions of critical care cases that are expected to die. Defaults to calculations based off "
                    "WC data (see Data documentation).",
            "action": "append"
        }
    )

    time_incubate: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days of disease incubation. Defaults to 5.1. Single attr defines a scalar, two inputs define a "
                    "Uniform prior."
        }
    )

    time_infectious: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days that infectious individuals can spread the virus."
        }
    )

    time_s_to_h: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days from onset of symptoms to hospital admission for severe cases. Defaults to 6. Single input "
                    "defines a scalar, two inputs define a Uniform prior."
        }
    )

    time_s_to_c: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days from onset of symptoms to critcal care admission for severe cases. Defaults to 6. Single "
                    "input defines a scalar, two inputs define a Uniform prior."
        }
    )

    time_h_to_c: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days from admission of general hospital to critical care, for those that will require it. "
                    "Defaults to calculations based off WC data (see Data documentation)."
        }
    )

    time_h_to_r: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days from general hospital admission to recovery, for those that will recover. Defaults to "
                    "calculations from WC data (see Data documentation)."
        }
    )

    time_h_to_d: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days from general hospital admission to death, for those that will die. Defaults to calculations "
                    "from WC data (see Data documentation)."
        }
    )

    time_c_to_r: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days from critical care admission to recovery, for those that will recover. Defaults to "
                    "calcualtions from WC data (see Data documentation)."
        }
    )

    time_c_to_d: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days from critical care admission to death, for those that will die. Defaults to "
                    "calcualtions from WC data (see Data documentation)."
        }
    )

    contact_k: List[float] = list_field(
        default=None,
        metadata={
            "help": "Contact heterogeneity factor from Kong et. al. (see References documentation). Defaults to None, "
                    "implying contact is homogeneous."
        }
    )


@dataclass
class Y0CLI(BaseDistributionCLI):

    _defaults_dict = {
        'prop_e0': PROP_E0_DEFAULT
    }

    prop_e0: List[float] = list_field(
        default=None,
        metadata={
            "help": "Proportion of starting population in the exposed category. Used to seed the SEIR model. Defaults "
                    "to a Uniform prior U(0, 1e-6). Single attr defines a scalar, two inputs define a Uniform prior."
        }
    )


@dataclass
class MetaCLI(BaseCLI):

    nb_samples: Optional[int] = field(
        default=NB_SAMPLES_DEFAULT,
        metadata={
            "help": "Number of samples to take for the prior distributions in the ASSA model SIR algorithm."
        }
    )

    age_heterogeneity: bool = field(
        default=AGE_GROUPS_DEFAULT,
        metadata={
            "help": "Flag to set the use of population age bands. Bands are in ten years, from 0-9, 10-19, ..., to "
                    "80+. The age defined attack rates are informed by Ferguson et al. (see References documentation)."
        }
    )


@dataclass
class FittingCLI(BaseCLI):

    nb_runs: Optional[int] = field(
        default=NB_RUNS_DEFAULT,
        metadata={
            "help": "Number of runs to perform. Used when running into memory errors with a large number of samples. "
                    "Final result will have had nb_samples * nb_runs number of samples for the prior, and "
                    "ratio_resample * nb_samples * nb_runs number of resamples."
        }
    )

    ratio_resample: Optional[int] = field(
        default=0.05,
        metadata={
            "help": "The percentage of resamples to take in the SIR algorithm."
        }
    )

    fit_totals: bool = field(
        default=True,
        metadata={
            "help": "Fit data to the sub totals of all population groups. Useful when the data does not contain "
                    "population group differences (as in, for example, the DSFSI data."
        }
    )

    fit_deaths: bool = field(
        default=False,
        metadata={
            "help": "Fits model to death data, if available."
        }
    )

    fit_recovered: bool = field(
        default=False,
        metadata={
            "help": "Fits model to recovered data, if available."
        }
    )

    fit_infected: bool = field(
        default=False,
        metadata={
            "help": "Fits model to infected cases, if available."
        }
    )

    fit_hospitalised: bool = field(
        default=False,
        metadata={
            "help": "Fits model to hospital data, if available."
        }
    )

    fit_critical: bool = field(
        default=False,
        metadata={
            "help": "Fits model to ICU data, if available."
        }
    )
