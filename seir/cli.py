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
    HOSPITAL_LOADING_DEFAULT,
    MORTALITY_LOADING_DEFAULT
)


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


def _sample_cli_attr(attr, nb_groups, nb_samples) -> np.ndarray:
    if len(attr) == 1:
        return np.expand_dims(np.asarray(attr[0]), axis=(0, 1))
    elif len(attr) == 2:
        return np.random.uniform(attr[0], attr[1], size=(nb_groups, nb_samples))
    else:
        raise ValueError(f"Uniform distribution should have two values, a lower and upper bound. Got {len(attr)} "
                         f"number of parameters instead.")



class BaseCLI:
    """
    Base class for all command line interface dataclass objects. These are designed to be used with the
    DataClassArgumentParser object from the seir.argparser module. Contains basic methods that allow the the CLI
    arguments to be saved as a json object or parsed as a json string.

    Objects inheriting from this should be a dataclass. Parameters of the inherited objects should be defined as a
    dataclass field, with a dict "metadata" being parsed as the kwargs of an argument in pythons built-in argparser
    module.
    """

    def to_json(self, fp: Union[str, Path]):
        """
        Saves the command line interface object to a json file. Only saves the file, will not create any parent
        directories.

        Parameters
        ----------
        fp: str, Path
            The file path at which to save the objects json string.

        Returns
        -------
        None
        """
        if isinstance(fp, str):
            fp = Path(fp)
        if not fp.parent.is_dir():
            raise ValueError(f"The directory {fp.parent} is not a directory.")

        with fp.open('wb') as f:
            json.dump(asdict(self), f, indent=4)

    def to_json_string(self) -> str:
        """
        Returns the json string of the command line object.

        Returns
        -------
        json_string: str
            The json string of the object.
        """
        return json.dumps(asdict(self), indent=4)


class BaseDistributionCLI(BaseCLI):
    """
    Base class for command line arguments that define distributions. Assumes that the defined distribution is uniform.
    Parses its own arguments by treating a single input in a list as a float, and two inputs as defining the bounds
    of a uniform distribution. Possesses a sample_attr method that will return a number of random samples from the
    uniform distribution, shaped in such a way as to be digested for the sample parameter objects in the seir.parameters
    module.

    Objects inheriting from this should be a dataclass. Parameters of inherited the objects should be defined as a
    dataclass field, with a dict "metadata" being parsed as the kwargs of an argument in pythons built-in argparser
    module.
    """

    _defaults_dict = {}

    def __post_init__(self):
        if not self._defaults_dict.keys() == self.__dict__.keys():
            raise NotImplementedError("CLI objects _defaults_dict should contain the default values for all "
                                      "attributes.")
        self._set_defaults()

    def _set_defaults(self):
        """
        Method for setting of defaults in the base distribution object. Used to overcome problems with setting the
        defaults of parameters that contain the allow appending via the key value pair ("action", "append") in a
        parameters metedata.

        Returns
        -------
        out: bool
            Returns true if performed successfully.
        """
        self_vars = self.__dict__
        for k in self_vars:
            self_vars[k] = self._defaults_dict.get(k, None) if self_vars[k] is None else self_vars[k]
        return True

    def sample_attr(self, attr: str, nb_groups: int = 1, nb_samples: int = 1) -> np.ndarray:
        """
        Samples an attribute of the cli to the required scalar or uniform distribution.

        An example of the sampling method applied to an attribute is as follows. Let the attribute of interest be
        named x. If the value of x is a single value in a list, say x=[0.8], then this method will return the
        zero-dimensional numpy array: array(0.8). If instead two values of present, say x=[0, 1], then x is sampled
        from the uniform distribution x ~ U(0, 1). If the parameters metadata allows for appending, then x will be a
        list of lists, say x=[[0.8, 0.9], [0, 1]]. In this case the sampling algorithm is applied to each value within
        the list, leading to a concatenated array x ~ [U(0.8, 0.9), U(0, 1)].

        Parameters
        ----------
        attr: str
            Attribute of the CLI to parse.

        nb_groups: int, default=1
            Number of population groups from which to define the size of the uniform distribution.

        nb_samples: int, default=1
            Number of samples to take from the uniform distribution.

        Returns
        -------
        out: np.ndarray
            Returns either a zero dimensional float as an numpy array, or a numpy array of size (nb_groups, nb_samples).
            If the attributes metadata allows appending, then instead the method is applied to each element in the list
            of appended values, returning an array of shape (n, nb_samples), where n is the number of appended elements.
        """
        attr_val = getattr(self, attr)
        try:
            if (
                    hasattr(self.__dataclass_fields__[attr].type, '__origin__')
                    and issubclass(self.__dataclass_fields__[attr].type.__origin__, List)
            ):
                if self.__dataclass_fields__[attr].metadata.get('action', None) == 'append':
                    return np.concatenate([_sample_cli_attr(x, 1, nb_samples) for x in attr_val], axis=0)
                return np.asarray(_sample_cli_attr(attr_val, nb_groups, nb_samples))
            return attr_val
        except Exception as e:
            raise ValueError(f"Attribute '{attr}' failed to be parsed. Raised exception '{e}'.")


@dataclass
class LockdownCLI(BaseDistributionCLI):
    """
    Lockdown CLI. Used to define the periods of various phases of a lockdown, as well as the relative strength of the
    effects. Any number of lockdown periods and strengths can be defined.
    """

    _defaults_dict = {
        'rel_beta_lockdown': REL_BETA_LOCKDOWN_DEFAULT,
        'rel_beta_period': REL_BETA_PERIOD_DEFAULT,
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
                    " Can be called multiple times to define multiple successive lockdown periods occurring one after "
                    "the other.",
            "action": "append"
        }
    )

    def __post_init__(self):
        super().__post_init__()
        assert len(self.rel_beta_period) == len(self.rel_beta_lockdown), \
            f"There should be a one-to-one correspondence between the number of lockdown periods and the strengths of " \
            f"each of the lockdown periods. Instead found {len(self.rel_beta_period)} number of lockdown periods " \
            f"and {len(self.rel_beta_lockdown)} number of lockdown strengths."

    def sample_attr(self, attr: str, nb_groups: int = 1, nb_samples: int = 1) -> List[np.ndarray]:
        """
        Overrides the base sample_attr method to suit the needs of the lockdown parameters. Since lockdowns can have
        various strengths, and may have more certain periods than others, the output of this sampling method is to
        produce a variable list of numpy arrays, rather than a single numpy array.

        Parameters
        ----------
        attr: str
            The attribute to sample.

        nb_groups: int, default=1
            The number of population groups to sample for.

        nb_samples: int, default=1
            The number of samples to take.

        Returns
        -------
        out: list
            List of numpy arrays. If the parsed value contained a single element, then a zero dimensional array is
            returned in its place, otherwise a set of samples is given from a uniform distribution defined by the bounds
            of the parsed value.
        """
        attr_val = getattr(self, attr)
        outputs = [_sample_cli_attr(attr_val[0], nb_groups, nb_samples)]
        for i in range(1, len(attr_val)):
            if len(attr_val[i]) == 1:
                outputs.append(_sample_cli_attr(attr_val[i], nb_groups, nb_samples))
            elif len(attr_val[i]) == 2:
                if attr_val[i][0] < 0:
                    outputs.append(np.random.uniform(outputs[i-1] - abs(attr_val[i][0]), attr_val[i][1],
                                                     size=(nb_groups, nb_samples)))
                else:
                    outputs.append(np.random.uniform(attr_val[i][0], attr_val[i][1], size=(nb_groups, nb_samples)))
        return outputs


@dataclass
class OdeParamCLI(BaseDistributionCLI):
    """
    Command line interface for all parameters relating to the Assa Covid SEIR ordinary differential equation.
    """

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
        'mortality_loading': MORTALITY_LOADING_DEFAULT,
        'hospital_loading': HOSPITAL_LOADING_DEFAULT,
        'smoothing_time': 11,
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
                    "calculations from WC data (see Data documentation)."
        }
    )

    time_c_to_d: List[float] = list_field(
        default=None,
        metadata={
            "help": "Days from critical care admission to death, for those that will die. Defaults to "
                    "calculations from WC data (see Data documentation)."
        }
    )

    contact_k: List[float] = list_field(
        default=None,
        metadata={
            "help": "Contact heterogeneity factor from Kong et. al. (see References documentation). Defaults to 0, "
                    "implying contact is homogeneous."
        }
    )

    mortality_loading: List[float] = list_field(
        default=None,
        metadata={
            "help": "Mortality loading parameter applied to deaths. Used to pseudo inform the uncertainty in these"
                    "parameters while keeping the shape of mortality over age groups constant."
        }
    )

    hospital_loading: List[float] = list_field(
        default=None,
        metadata={
            "help": "Hospital loading parameter applied to in bound patients. Used to pseudo inform the uncertainty "
                    "in these parameters while keeping the shape of those going to hospital over age groups constant."
        }
    )

    smoothing_time: float = field(
        default=11,
        metadata={
            "help": "Period over which the lockdown is smoothed before the lockdown begins. Interpolates the relative "
                    "beta strength from 1 to the first lockdown beta strength value linearly over the smoothing period."
        }
    )


@dataclass
class MetaCLI(BaseCLI):
    """
    Command line interface that stores any meta data about the system we are solving, such as the number of samples
    the ode is taking, as well as whether or not the model should incorporate an age heterogenous structure. The age
    heterogeneity is introduced by means of 10 year age bands.
    """

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

    nb_groups: int = field(init=False)

    def __post_init__(self):
        self.nb_groups = 9 if self.age_heterogeneity else 1


@dataclass
class FittingCLI(BaseCLI):
    """
    Command line interface for all fitting parameters used by the model.
    """

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

    fit_daily: bool = field(
        default=False,
        metadata={
            "help": "Will fit to daily cases/deaths/etc cases instead of cumulative cases. Used to remove the serial "
                    "dependence found in such cumulative cases."
        }
    )

    fit_interval: int = field(
        default=1,
        metadata={
            "help": "For concurrent data (hospital/icu cases), fitter will fit to every X data point, where X is "
                    "defined here. If fitting to daily cases (see --fit_daily), will take a sum of X data points to "
                    "use in fitting (in order to smooth out the fitting data and account for the noise in daily "
                    "reporting statistics)."
        }
    )
