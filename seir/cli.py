from dataclasses import dataclass, field
from typing import List, Optional


def list_field(default=None, metadata=None):
    return field(default_factory=lambda: default, metadata=metadata)


@dataclass
class AssaCovidModelCLI:

    r0: List[float] = list_field(
        default=[1.5, 3.5],
        metadata={
            "help": "Basic reproductive number r0. Single input defines a scalar value, two inputs define a "
                    "Uniform prior."
        }
    )

    rel_beta_lockdown_upper: List[float] = list_field(
        default=[0.9, 0.9, 0.9, 0.9],
        metadata={
            "help": "Upper bounds of relative beta uniform prior distributions corresponding to relative beta "
                    "periods. Used to inform the behavior of Rt during periods of lockdown."
        }
    )

    rel_beta_lockdown_lower: List[float] = list_field(
        default=[-0.05, -0.05, -0.05, -0.05],
        metadata={
            "help": "Lower bounds of the relative beta uniform prior distributions corresponding to relative beta "
                    "periods. Negative values indicate setting the lower bound according to the previous lockdown "
                    "period's relative beta. Used to inform the behavior of Rt during periods of lockdown."
        }
    )

    rel_beta_lockdown_periods: List[float] = list_field(
        default=[35, 31, 30, 31],
        metadata={
            "help": "Length of each period for which the relative beta's apply, in days, after the start of lockdown."
        }
    )

    rel_beta_postlockdown: List[float] = list_field(
        default=[0.8],
        metadata={
            "help": "Relative beta post all lockdown measures. Used to inform Rt due to behaviour changes introduced "
                    "by lockdown. Single input defines a scalar value, two inputs define a Uniform prior."
        }
    )

    rel_beta_asymptomatic: List[float] = list_field(
        default=[0, 1],
        metadata={
            "help": "The relative infectivity strength of asymptomatic cases. Single input defines a scalar value, two"
                    " inputs define a Uniform prior."
        }
    )

    age_groups: bool = field(
        default=False,
        metadata={
            "help": "Flag to set the use of population age bands. Bands are in ten years, from 0-9, 10-19, ..., to "
                    "80+. The age defined attack rates are informed by Ferguson et al. (see References documentation)."
        }
    )

    prop_a: List[float] = list_field(
        default=[0.5],
        metadata={
            "help": "Proportion of asymptomatic infected individuals. Single input defines a scalar, two inputs define "
                    "a Uniform prior."
        }
    )

    prop_s: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Proportion of symptomatic individuals (1 - prop_a) that experience severe symptoms. Defaults to "
                    "attack rates defined by Ferguson et. al. (see References documentation)."
        }
    )

    prop_s_to_h: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Propotion of severe cases that will be admitted to general hospital. The rest will present "
                    "directly to ICU. Defaults to calculations based off WC data (see Data documentation)."
        }
    )

    prop_h_to_c: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Proportion of general hospital cases expected to be transferred to critical care. Defaults to "
                    "calculations based off WC data (see Data documentation)."
        }
    )

    prop_h_to_d: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Proportion of general hospital cases that are expected to die. Defaults to calculations based off "
                    "WC data (see Data documentation)."
        }
    )

    prop_c_to_d: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Proportions of critical care cases that are expected to die. Defaults to calculations based off "
                    "WC data (see Data documentation)."
        }
    )

    time_incubate: Optional[List[float]] = list_field(
        default=[5.1],
        metadata={
            "help": "Days of disease incubation. Defaults to 5.1. Single input defines a scalar, two inputs define a "
                    "Uniform prior."
        }
    )

    time_s_to_h: Optional[List[float]] = list_field(
        default=[6],
        metadata={
            "help": "Days from onset of symptoms to hospital admission for severe cases. Defaults to 6. Single input "
                    "defines a scalar, two inputs define a Uniform prior."
        }
    )

    time_s_to_c: Optional[List[float]] = list_field(
        default=[6],
        metadata={
            "help": "Days from onset of symptoms to critcal care admission for severe cases. Defaults to 6. Single "
                    "input defines a scalar, two inputs define a Uniform prior."
        }
    )

    time_h_to_c: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Days from admission of general hospital to critical care, for those that will require it. "
                    "Defaults to calculations based off WC data (see Data documentation)."
        }
    )

    time_h_to_r: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Days from general hospital admission to recovery, for those that will recover. Defaults to "
                    "calculations from WC data (see Data documentation)."
        }
    )

    time_h_to_d: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Days from general hospital admission to death, for those that will die. Defaults to calculations "
                    "from WC data (see Data documentation)."
        }
    )

    time_c_to_r: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Days from critical care admission to recovery, for those that will recover. Defaults to "
                    "calcualtions from WC data (see Data documentation)."
        }
    )

    contact_k: Optional[List[float]] = list_field(
        default=None,
        metadata={
            "help": "Contact heterogeneity factor from Kong et. al. (see References documentation). Defaults to None, "
                    "implying contact is homogeneous."
        }
    )

    prop_e0: Optional[List[float]] = list_field(
        default=[0, 1e-6],
        metadata={
            "help": "Proportion of starting population in the exposed category. Used to seed the SEIR model. Defaults "
                    "to a Uniform prior U(0, 1e-6). Single input defines a scalar, two inputs define a Uniform prior."
        }
    )
