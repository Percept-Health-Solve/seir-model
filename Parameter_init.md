# Parameter values for calibrations

Last updated: 13 June 2020. Parameter values/distributions shown only for age-banded model.

Calibration is done through the use of sampling-importance-resampling (Rubin 1987).

## Parameter values

There are four different types of parameter in the model:

1. Controlling arguments, typically categorical variables which govern model choices;

2. Deterministic parameters, whose value is fixed deterministically across all samples and all scenarios;

3. Scenario parameters, whose value is also fixed deterministically for all samples within a scenario, but varied between scenarios; and

4. Variable parameters, whose prior distributions are set: sampled values are then drawn (independently) from these distributions in order to test the fit of parameter combinations within a very wide parameter space.

Parameters beginning with `--` are parser arguments in the script, i.e. can be set in the command line call.

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
* `time_h_to_c`: LOS in hospital before ICU admission, for those who go through both states = 2.6 (previously 10)
* `time_h_to_r`: LOS in hospital before recovery, for those who recover = 8 (previously 10.1)
* `time_h_to_d`: LOS in hospital before death, for those who die without going to ICU = 8 (previously 9.9)
* `time_c_to_r`: LOS in ICU before recovery, for those who recover = 16 (previously 18.3)
* `time_c_to_d`: LOS in ICU before death, for those who die = 13 (previously 18.8)
* `t0` (from `--t0`): seeding date for infections, relative to start of lockdown = -50
* `ratio_as_detected`, `ratio_m_detected`, `ratio_s_detected`: proportion of asymptomatic, mild and severe cases assumed to be detected and hence reported in confirmed cases = (0,0.3,1)
* `prop_s_to_h` (can be set as a calibration parameter with upper and lower bounds for a uniform distribution in `--prop_s_to_h_range`): proportion of severe cases who will go to hospital general ward (the remainder go straight to ICU) = 0.8875
* `period_lockdown5` = 35 (Level 5 lockdown 27 March to 30 April)
* `period_lockdown4` = 66 (Level 5 plus May)- note that these periods are defined in terms of the number of days after the start of Level 5 that they end
* `period_lockdown3` = 96 (plus June)
* `period_lockdown2` = 127  (plus July)

### Scenario parameters

*Notes: finalise scenarios; model structure changes to be considered for post-lockdown period with variable relative betas*

* ```--contact_k``` is the value of `k` in the Kong et al. method, 0.25 by default; scenario values to be determined
* `rel_lockdown4_beta`: infectious spread post Level 5 relative to baseline, 0.8 by default; scenario values to be determined
* `rel_lockdown3_beta`, `rel_lockdown3_beta`, `rel_lockdown2_beta`, `rel_postlockdown_beta` = 0.8 by default; scenarios to be determined
* ```prop_as_range``` is the range of values which can be taken by the assumed proportion asymptomatic; at present we are making this deterministic per scenario with a base assumption of 0.5 (achieved by setting the upper and lower bounds to 0.5), pending data from the WCDoH, with other scenario values to be determined.
* `prop_m`: proportion of infections that are mild = age-banded proportions of non-asymptomatic infections in table below, taken from Ferguson et al. (2020)
* `prop_h_to_c`: proportion of initial hospital (general ward) admissions in table below, from observed Western Cape data - not updated 14 June due to time constraints but needs review
* `prop_h_to_d`: proportion of hospital (general ward) admissions that will die in table below, from observed Western Cape data - updated 14 June
* `prop_c_to_d`: proportion of ICU admissions that will die in table below, from observed Western Cape data - updated 14 June


Age-banded table:

|   Age  | `prop_m` | `prop_h_to_c` | `prop_h_to_d` | `prop_c_to_d` |
|:------:|---------:|--------------:|--------------:|--------------:|
|   0-9  |    0.999 |          1/81 |         0.011 |         0.011 |
| 10-19  |    0.997 |          1/81 |         0.042 |         0.042 |
| 20-29  |    0.988 |          1/81 |         0.045 |         0.410 |
| 30-39  |    0.968 |         7/184 |         0.063 |         0.540 |
| 40-49  |    0.951 |        32/200 |         0.096 |         0.590 |
| 50-59  |    0.898 |        38/193 |         0.245 |         0.650 |
| 60-69  |    0.834 |        24/129 |         0.408 |         0.660 |
| 70-79  |    0.757 |         10/88 |         0.448 |         0.670 |
|   80+  |    0.727 |          5/31 |         0.526 |         0.710 |


### Variable parameters

*Notes: check source of time infectious assumption; confirm sources of prop parameters; consider whether range of R0 distribution should be shifted upward; review prop parameters as use of empirical WC experience leads to counter-intuitive shapes by age in some parts of the curves*

* `time_infectious` (upper and lower bounds from `--time_infectious_range`: days for which an infected individual is infectious to others before being isolated $\sim U(1.5,2.6)$
* `r0` (from `--r0_range`): baseline reproductive number with fully susceptible population $\sim U(1.5,3.5)$
* `rel_lockdown5_beta` (upper and lower bounds from `--rel_lockdown5_beta_range`): rate of infectious spread in Level 5 lockdown compared to baseline $\sim U(0.4,1)$
* `rel_beta_as` (upper and lower bounds from `--rel_beta_as_range`): relative infectiousness of asymptomatic cases $\sim U(0.3,1)$
* `e0` (upper and lower bounds from `--e0_range`): proportion of initial population at `t0` that is exposed $\sim U(0,10^{-5})$
* `mort_loading`: applies a loading to the age-banded mortality rates (`prop_*_to_d` in the table above) $\sim U(0.8,1.2)$




### References

(Rubin, D.B., 1987). The calculation of posterior distributions by data augmentation: Comment: A noniterative sampling/importance resampling alternative to the data augmentation algorithm for creating a few imputations when fractions of missing information are modest: The SIR algorithm. Journal of the American Statistical Association, 82(398), pp.543-546.

