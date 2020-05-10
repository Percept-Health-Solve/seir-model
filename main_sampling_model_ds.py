import numpy as np
import scipy.stats as st
import pandas as pd

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from seir.sampling.model import SamplingNInfectiousModel

import logging
logging.basicConfig(level=logging.INFO)


def check_priors_plot(model):
    tt = np.arange(t0, 50).astype(int)
    y = model.solve(tt)

    i_as = y[:, :, :, 2]
    i_m = y[:, :, :, 3]
    i_s = y[:, :, :, 4]
    i_i = y[:, :, :, 5]
    i_h = y[:, :, :, 6]
    i_icu = y[:, :, :, 7]
    r_as = y[:, :, :, 8]
    r_m = y[:, :, :, 9]
    r_h = y[:, :, :, 10]
    r_icu = y[:, :, :, 11]
    d_icu = y[:, :, :, 12]

    ratio_as_detected = 0
    ratio_m_detected = 0.3
    ratio_s_detected = 1

    cum_detected_samples = ratio_as_detected * (i_as + r_as) + ratio_m_detected * (i_m + r_m) \
                           + ratio_s_detected * (i_s + i_i + i_h + i_icu + r_h + r_icu + d_icu)

    det_mean = np.mean(cum_detected_samples, axis=1)
    det_lc, det_hc = st.t.interval(0.95, len(cum_detected_samples) - 1, loc=det_mean, scale=st.sem(cum_detected_samples, axis=1))

    d_icu_mean = np.mean(d_icu, axis=1)
    d_lc, d_hc = st.t.interval(0.95, len(d_icu_mean) - 1, loc=d_icu_mean, scale=st.sem(d_icu, axis=1))

    fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    ax = ax.flat
    ax[0].plot(tt, det_mean[:, 0])
    ax[0].fill_between(tt, det_lc[:, 0], det_hc[:, 0], color='C0', alpha=0.1)
    ax[0].plot(tt, cum_detected_samples[:, :, 0], c='k', alpha=0.005)
    ax[0].plot(t, i_d_obs)
    ax[0].set_ylim((-1, 4000))

    ax[1].plot(tt, i_h[:, :, 0], c='k', alpha=0.005)
    ax[1].plot(t, i_h_obs)
    ax[1].set_ylim((-1, 200))

    ax[2].plot(tt, i_icu[:, :, 0], c='k', alpha=0.005)
    ax[2].plot(t, i_icu_obs)
    ax[2].set_ylim((-1, 200))

    ax[3].plot(tt, d_icu_mean[:, 0])
    ax[3].fill_between(tt, d_lc[:, 0], d_hc[:, 0], color='C0', alpha=0.1)
    ax[3].plot(tt, d_icu[:, :, 0], c='k', alpha=0.005)
    ax[3].plot(t, d_icu_obs)
    ax[3].set_ylim((-1, 1000))

    plt.tight_layout()
    plt.show()


def save_vars_to_csv(resample_vars: dict, scalar_vars: dict, group_vars: dict, nb_groups, nb_samples, base='data/samples'):
    logging.info(f'Saving variables to {base}_*.csv for {nb_groups} groups and {nb_samples} samples')
    # need to reshape resample vars
    reshaped_resample_vars = {}
    for key, value in resample_vars.items():
        reshaped_resample_vars[key] = value.reshape(-1)
    reshaped_resample_vars['group'] = np.asarray([[i] * nb_samples for i in range(nb_groups)]).reshape(-1)
    for key, value in reshaped_resample_vars.items():
        print(key, value.shape)

    # define df
    df_resample = pd.DataFrame(reshaped_resample_vars)

    # save files
    df_resample.to_csv(f"{base}_resample.csv", index=False)
    with open(f'{base}_scalar.pkl', 'wb') as f:
        pickle.dump(scalar_vars, f)
    with open(f'{base}_group.pkl', 'wb') as f:
        pickle.dump(group_vars, f)

if __name__ == '__main__':

    nb_groups = 1
    nb_samples = 100 #100000

    r0 = np.random.uniform(2, 3.5, size=(nb_samples, 1))
    time_infectious = np.random.uniform(1.5, 4, size=(nb_samples, 1))
    e0 = np.random.uniform(0.5, 5, size=(nb_samples, 1))
    y0 = np.zeros((nb_samples, nb_groups, 13))
    y0[:, :, 0] = 7000000 - e0
    y0[:, :, 1] = e0
    y0 = y0.reshape(-1)
    # y0 = [7000000-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    t0 = -50

    inf_as_prop = np.random.uniform(0.1, 0.9, size=(nb_samples, 1))

    model = SamplingNInfectiousModel(
        nb_groups=nb_groups,
        baseline_beta=r0/time_infectious,
        rel_lockdown_beta=np.random.uniform(0, 1, size=(nb_samples, 1)),
        rel_postlockdown_beta=np.random.uniform(0.7, 0.8, size=(nb_samples, 1)),
        rel_beta_as=np.random.uniform(0.3, 0.7, size=(nb_samples, 1)),
        time_inc=5.1,
        inf_as_prop=inf_as_prop,
        inf_m_prop=(1 - inf_as_prop) * np.random.beta(a=10, b=1, size=(nb_samples, 1)),
        time_infectious=time_infectious,
        time_s_to_h=6,
        time_h_to_icu=8,
        time_h_to_r=10,
        time_icu_to_r=10,
        time_icu_to_d=6,
        hosp_icu_prop=0.2133,
        icu_d_prop=0.6,
        y0=y0
    )

    # get data
    df_deaths = pd.read_csv(
        'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_deaths.csv',
        parse_dates=['date'],
        date_parser=lambda t: pd.to_datetime(t, format='%d-%m-%Y')
    )
    df_confirmed = pd.read_csv(
        'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_confirmed.csv',
        parse_dates=['date'],
        date_parser=lambda t: pd.to_datetime(t, format='%d-%m-%Y')
    )
    df_hosp_icu = pd.read_csv('data/WC_hosp_icu.csv',
                              parse_dates=['Date'],
                              date_parser=lambda t: pd.to_datetime(t, format='%d/%m/%Y'))

    df_deaths = df_deaths.sort_values('date')
    df_confirmed = df_confirmed.sort_values('date')
    df_hosp_icu = df_hosp_icu.sort_values('Date')

    df_deaths = df_deaths[['date', 'WC']]
    df_confirmed = df_confirmed[['date', 'WC']]

    df_confirmed = df_confirmed.interpolate(method='linear')

    df_deaths['Day'] = (df_deaths['date'] - pd.to_datetime('2020-03-27')).dt.days
    df_confirmed['Day'] = (df_confirmed['date'] - pd.to_datetime('2020-03-27')).dt.days
    df_hosp_icu['Day'] = (df_hosp_icu['Date'] - pd.to_datetime('2020-03-27')).dt.days

    df_merge = df_confirmed.merge(df_deaths, on='Day', how='left', suffixes=('_confirmed', '_deaths'))
    df_merge = df_merge.merge(df_hosp_icu, on='Day', how='left')
    df_merge = df_merge.interpolate(method='linear')
    df_merge = df_merge[['date_confirmed', 'WC_confirmed', 'WC_deaths', 'Current hospitalisations', 'Current ICU', 'Day']]
    df_merge = df_merge.fillna(0)

    df_merge['WC_confirmed'] = df_merge['WC_confirmed'].astype(int)
    df_merge['WC_deaths'] = df_merge['WC_deaths'].astype(int)
    df_merge['Day'] = df_merge['Day'].astype(int)

    # remove small observations
    df_merge = df_merge[df_merge['WC_confirmed'] > 500]
    df_merge = df_merge[df_merge['Current hospitalisations'] > 20]
    df_merge = df_merge[df_merge['Current ICU'] > 20]
    df_merge = df_merge[df_merge['WC_deaths'] > 20]

    t = df_merge['Day'].to_numpy()
    i_h_obs = df_merge['Current hospitalisations']
    i_icu_obs = df_merge['Current ICU']
    i_d_obs = df_merge['WC_confirmed']
    d_icu_obs = df_merge['WC_deaths']

    # get y_t.min() from y0
    logging.info('Solving for y at minimum data time')
    tt = np.arange(t0, t.min()+1).astype(int)
    y_tmin = model.solve(tt)[-1]
    y_tmin = y_tmin.reshape(-1)

    # check if priors match the data
    # check_priors_plot(model)

    # fit to data

    ratio_as_detected = 0
    ratio_m_detected = 0.3
    ratio_s_detected = 1
    ratio_resample = 0.05

    model.calculate_sir_posterior(t, i_d_obs, i_h_obs + i_icu_obs, None, d_icu_obs, y0=y_tmin,
                                  ratio_as_detected=ratio_as_detected,
                                  ratio_m_detected=ratio_m_detected,
                                  ratio_s_detected=ratio_s_detected,
                                  ratio_resample=ratio_resample)

    sample_vars = model.sample_vars
    resample_vars = model.resample_vars
    scalar_vars = model.scalar_vars
    group_vars = model.group_vars

    e0_resample = e0[np.random.choice(nb_samples, int(ratio_resample*nb_samples), p=model.weights)]

    # add e0 and t0 manually
    # TODO: Let the model accept initial parameters as potential random variables
    scalar_vars['t0'] = t0
    sample_vars['e0'] = e0
    sample_vars['r0'] = r0
    resample_vars['e0'] = e0_resample
    resample_vars['r0'] = resample_vars['time_infectious'] * resample_vars['baseline_beta']

    # save variables
    save_vars_to_csv(resample_vars, scalar_vars, group_vars, nb_groups, int(ratio_resample * nb_samples))

    logging.info('Plotting prior and posterior distributions')
    fig, axes = plt.subplots(2, 5, figsize=(16, 8))
    i = 0
    axes = axes.flat
    for key, value in resample_vars.items():
        # TODO: plot variables for multiple groups
        print(f'{key}: mean = {value.mean():.3f} - std = {value.std():.3f}')
        sns.kdeplot(value[:, 0], ax=axes[i], color='C0')
        sns.kdeplot(sample_vars[key][:, 0], ax=axes[i], color='C1')
        axes[i].set_title(key)
        i += 1

    plt.tight_layout()
    fig.savefig('data/priors_posterior.png')
    plt.show()




