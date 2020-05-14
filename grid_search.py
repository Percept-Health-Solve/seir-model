import numpy as np
import itertools

r0 = [1.5,2,2.25,2.5,2.75,3,3.25,3.5,4] 
time_infectious = [1.5,2.5,3,4]
e0 = [0.5,1,2,3,4,5]
inf_as_prop = [0.2,0.4,0.5,0.6,0.7,0.8]
rel_lockdown_beta = [0.25,0.4,0.5,0.6,0.8]
rel_postlockdown_beta = [0.5,0.7,0.9]
rel_beta_as = [0.4,0.6,0.8]
hosp_icu_prop = [0.15,0.2,0.25,0.3]
icu_d_prop = [0.5,0.6,0.7,0.8]

params = list(itertools.product(r0,time_infectious,e0,inf_as_prop,rel_lockdown_beta,rel_postlockdown_beta,rel_beta_as,hosp_icu_prop,icu_d_prop))
print(f'Implied nb_samples = {len(params):,.0f}.')

r0 = [x[0] for x in params]
time_infectious = [x[1] for x in params]
e0 = [x[2] for x in params]
inf_as_prop = [x[3] for x in params]
rel_lockdown_beta = [x[4] for x in params]
rel_postlockdown_beta = [x[5] for x in params]
rel_beta_as = [x[6] for x in params]
hosp_icu_prop = [x[7] for x in params]
icu_d_prop = [x[8] for x in params]

# next steps:
# tweak parameter value lists above if desired
# fit model with above parameters (model.__init__ converts to numpy arrays)
# identify ranges of parameters associated with scenarios with non-negligible likelihoods (> say 1e-20)
# if necessary (e.g. only one feasible value for some parameters), repeat grid search with tighter bands
# use insights to inform distribution of priors for sampling


