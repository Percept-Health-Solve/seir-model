# Parameter values for calibrations

Last updated: 10 June 2020. Parameter values/distributions shown only for age-banded model.

Calibration is done through the use of sampling-importance-resampling (Rubin 1987).

## Parameter values

There are four different types of parameter in the model:

1. Controlling arguments, typically categorical variables which govern model choices;

2. Deterministic parameters, whose value is fixed deterministically across all samples and all scenarios;

3. Scenario parameters, whose value is also fixed deterministically for all samples within a scenario, but varied between scenarios; and

4. Variable parameters, whose prior distributions are set: sampled values are then drawn (independently) from these distributions in order to test the fit of parameter combinations within a very wide parameter space.



### Controlling parameters

* ```--nb_runs``` is the number of runs (split up this way to avoid overburdening memory)
* ```--nb_samples``` is the number of samples; the product of ```nb_runs``` and ```nb_samples``` is 2,000,000 by default
* ```--ratio_resample``` is the ratio of resamples to samples = 0.05
* ```--fit_detected``` (Boolean) indicates whether to fit to confirmed cases, using assumed ratios of detected cases to true infections (set False)
* ```---fit_hospitalised``` (Boolean) indicates whether to fit to hospitalised cases (set True for WC calibration, False for national calibration)
* ```---fit_icu``` (Boolean) indicates whether to fit to ICU cases (set True for WC calibration, False for national calibration)
* ```---fit_deaths``` (Boolean) indicates whether to fit to hospitalised cases (set True)
* ```--contact_heterogeneous``` (Boolean) indicates whether to allow for the Kong et al. approach to heterogeneity, with False indicating a homogeneous run (set True)
* ```--likelihood``` = `lognormal` (default) or `poisson`
* ```--fit_new_deaths``` (Boolean) determines whether to fit to new periodic observations (if True, default) or cumulative observations (if False)
* ```--fit_interval``` specifies the periodicity of data for fitting, e.g. if set to 3, we will fit to new deaths reported over non-overlapping intervals of 3 days (this strips out some of the noise from apparent clumped reporting)



### Deterministic parameters

*Notes: update Western Cape survival analysis (need updated data); consider whether the model should allow for reduced future LOS*

* `time_incubate`: mean incubation period from exposed to infected = 5.1
* `time_s_to_h`, `time_s_to_c`: mean period from infection to onset before hospitalisation or ICU admission for serious cases = 6 for both
* `time_h_to_c`: LOS in hospital before ICU admission, for those who go through both states = 10
* `time_h_to_r`: LOS in hospital before recovery, for those who recover = 10.1
* `time_h_to_d`: LOS in hospital before death, for those who die without going to ICU = 9.9
* `time_c_to_r`: LOS in ICU before recovery, for those who recover = 18.3
* `time_c_to_d`: LOS in ICU before death, for those who die = 18.8
* `t0`: seeding date for infections, relative to start of lockdown = -50
* `ratio_as_detected`, `ratio_m_detected`, `ratio_s_detected`: proportion of asymptomatic, mild and severe cases assumed to be detected and hence reported in confirmed cases = (0,0.3,1)
* `prop_s_to_h`: proportion of severe cases who will go to hospital general ward (the remainder go straight to ICU) = 0.8875


### Scenario parameters

*Notes: finalise scenarios; model structure changes to be considered for post-lockdown period with variable relative betas*

* ```contact_k``` is the value of `k` in the Kong et al. method, 0.25 by default; scenario values to be determined
* `rel_postlockdown_beta`: infectious spread post Level 5 relative to baseline, 0.8 by default; scenario values to be determined
* ```prop_as_range``` is the range of values which can be taken by the assumed proportion asymptomatic; at present we are making this deterministic per scenario with a base assumption of 0.5 (achieved by setting the upper and lower bounds to 0.5), pending data from the WCDoH, with other scenario values to be determined



### Variable parameters

*Notes: check source of time infectious assumption; confirm sources of prop parameters; consider whether range of R0 distribution should be shifted upward; review prop parameters as use of empirical WC experience leads to counter-intuitive shapes by age in some parts of the curves*

* `time_infectious`: days for which an infected individual is infectious to others before being isolated $\sim U(1.5,2.6)$
* `prop_m`: proportion of infections that are mild = age-banded proportions of non-asymptomatic infections in table below, taken from Fergus et al. (2020)
* `prop_h_to_c`: proportion of initial hospital (general ward) admissions in table below, from observed Western Cape data
* `prop_h_to_d`: proportion of hospital (general ward) admissions that will die in table below, from observed Western Cape data
* `prop_c_to_d`: proportion of ICU admissions that will die in table below, from observed Western Cape data
* `r0`: baseline reproductive number with fully susceptible population $\sim U(1.5,3.5)$
* `rel_lockdown_beta`: rate of inectious spread in Level 5 lockdown compared to baseline $\sim U(0.4,1)$
* `rel_beta_as`: relative infectiousness of asymptomatic cases $\sim U(0.3,1)$
* `e0`: proportion of initial population at `t0` that is exposed $\sim U(0,10^{-6})$


Age-banded table:

|   Age  | `prop_m` | `prop_h_to_c` | `prop_h_to_d` | `prop_c_to_d` |
|:------:|---------:|--------------:|--------------:|--------------:|
|   0-9  |    0.999 |          1/81 |             0 |           0.1 |
| 10-19  |    0.997 |          1/81 |             0 |           0.1 |
| 20-29  |    0.988 |          1/81 |             0 |           0.1 |
| 30-39  |    0.968 |         7/184 |         7/177 |           2/7 |
| 40-49  |    0.951 |        32/200 |         8/168 |         14/32 |
| 50-59  |    0.898 |        38/193 |        23/155 |         18/38 |
| 60-69  |    0.834 |        24/129 |        28/105 |         12/24 |
| 70-79  |    0.757 |         10/88 |         26/78 |          6/10 |
|   80+  |    0.727 |          5/31 |         11/26 |           2/5 |


### References

(Rubin, D.B., 1987). The calculation of posterior distributions by data augmentation: Comment: A noniterative sampling/importance resampling alternative to the data augmentation algorithm for creating a few imputations when fractions of missing information are modest: The SIR algorithm. Journal of the American Statistical Association, 82(398), pp.543-546.

