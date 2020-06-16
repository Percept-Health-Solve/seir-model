# Notes on review 13-16 June

## Changes to code and questions pre calibration run

### Changes to code

* minor spelling changes in some of the parser arg help strings

* added Western Cape population  adjustment to 7 million for WC calibration (we can remove this once the population is finalised; for now it needs to be consistent with Andrew's population estimate)

* updated `Parameter_init.md` to accommodate changes (such as introduction of five lockdown periods with different relative betas)

* updated default lockdown periods in `sampling/model.py`; noted that these are defined (non-intuitively, given the parameter names) as the end-points of the respective periods in days since start of Level 5, as interpreted by `infectious_func` so updated parameters accordingly

* removed mort_loading from prop_h_to_c; this is not a mortality parameter

* added filename to logging.basicConfig to preserve log

* made changes to `hospitalisation_transform.py` and `process_raw_WC_data.py` to work with (questionable) June data provided by Andrew, not in the same format as previous datasets

* added a 0.3678 line to the KM plots; used rather than means for our parameters, since the means are being pulled far to the right by long-stayers- they are also more consistent with NCEM assumptions

* changed `max_date` adjustment to 5 days in `main_sampling_model.py`

* `WC_mortality.py`: introduced death_adjt to gross up for deaths outside of hospital, and fitted Gompertz curves to this (but didn't use these in the calibration run)

* changed relative betas to 0.6 (level 4 and 3), 0.7 (2) and 0.8 (post) - needs discussion
 

### Questions

* need to check with Jason why the relative beta structure was not more granular as per Barry's instructions

* why is there a parser arg for `rel_post_lockdown_beta` but not for the intermediate lockdown periods?

* should there be some variability in the split between mild and severe, give that we're locking in the proportion asymptomatic?

* `mort_loading` is not included in prior/posterior plot; it would be more parsimonious to show this than to show 18 plots for the ageband mortality rates from hospital and ICU

* the x-axis scale of `sigma_*` in the prior/posterior plots needs adjusting

* have not updated assumptions for `prop_h_to_c` given time constraints

## Calibration run commands

### Test run
python main_sampling_model.py --nb_samples 50000 --ratio_resample 0.1 --model_name 'WC_calib_test' --nb_runs 2 --age_groups --fit_hospitalised --fit_icu --fit_deaths --fit_data 'WC' --fit_interval 3 --fit_new_deaths --contact_heterogeneous --contact_k 0.25 --likelihood 'lognormal'

### Main calibration
python main_sampling_model.py --nb_samples 200000 --ratio_resample 0.05 --model_name 'WC_calib_20200614' --nb_runs 10 --age_groups --fit_hospitalised --fit_icu --fit_deaths --fit_data 'WC' --fit_interval 3 --fit_new_deaths --contact_heterogeneous --contact_k 0.25 --likelihood 'lognormal'

Note: 19 hour run-time with memory errors at the end.

## Changes after calibration run:

* changed sign in a line in `infectious_func`: return 1 - (1 - rel_lockdown5_beta) / 11 * (t **+** 11) 



