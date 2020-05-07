import numpy as np
import scipy.stats as st
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from seir.sampling.model import SamplingNInfectiousModel

import logging
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':

    nb_samples = 500000

    infectious_func = lambda t: 1 if t < 0 else 0.6 if 0 <= t <= 0 + 6*7 else 0.75
    r0 = np.random.normal(loc=2.75, scale=0.375, size=(nb_samples, 1)).clip(min=1.5)
    time_infectious = np.random.normal(loc=2.3, scale=0.375, size=(nb_samples, 1)).clip(min=1.5)
    y0 = [7000000-1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    t0 = -50

    inf_as_prop = np.random.normal(loc=0.75, scale=0.1, size=(nb_samples, 1)).clip(min=0, max=1)

    model = SamplingNInfectiousModel(
        nb_groups=1,
        baseline_beta=r0/time_infectious,
        rel_lockdown_beta=np.random.uniform(0, 1, size=(nb_samples, 1)),
        rel_postlockdown_beta=np.random.normal(loc=0.75, scale=0.05, size=(nb_samples, 1)).clip(max=1, min=0.3),
        rel_beta_as=np.random.uniform(0.3, 0.7, size=(nb_samples, 1)),
        time_inc=5.1,
        inf_as_prop=inf_as_prop,
        inf_m_prop=(1 - inf_as_prop) * np.random.beta(a=10, b=1, size=(nb_samples, 1)).clip(min=0, max=1),
        time_infectious=time_infectious,
        time_s_to_h=6,
        time_h_to_icu=8,
        time_h_to_r=10,
        time_icu_to_r=10,
        time_icu_to_d=6,
        hosp_icu_prop=0.2133,
        icu_d_prop=0.6,
        y0=y0 * 1 * nb_samples
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

    t = df_merge['Day'].to_numpy()
    # pre_days = np.arange(-30+t.min(), t.min()+1, 1).astype(int)
    # t = np.concatenate([pre_days, t])
    # print(t.min())
    # t = t + t.min()
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
    # tt = np.arange(t0, 50).astype(int)
    # y = model.solve(tt)
    #
    # i_as = y[:, :, :, 2]
    # i_m = y[:, :, :, 3]
    # i_s = y[:, :, :, 4]
    # i_i = y[:, :, :, 5]
    # i_h = y[:, :, :, 6]
    # i_icu = y[:, :, :, 7]
    # r_as = y[:, :, :, 8]
    # r_m = y[:, :, :, 9]
    # r_h = y[:, :, :, 10]
    # r_icu = y[:, :, :, 11]
    # d_icu = y[:, :, :, 12]
    #
    # ratio_as_detected = 0
    # ratio_m_detected = 0.3
    # ratio_s_detected = 1
    #
    # cum_detected_samples = ratio_as_detected * (i_as + r_as) + ratio_m_detected * (i_m + r_m) \
    #                        + ratio_s_detected * (i_s + i_i + i_h + i_icu + r_h + r_icu + d_icu)
    #
    # det_mean = np.mean(cum_detected_samples, axis=1)
    # det_lc, det_hc = st.t.interval(0.95, len(cum_detected_samples) - 1, loc=det_mean, scale=st.sem(cum_detected_samples, axis=1))
    #
    # d_icu_mean = np.mean(d_icu, axis=1)
    # d_lc, d_hc = st.t.interval(0.95, len(d_icu_mean) - 1, loc=d_icu_mean, scale=st.sem(d_icu, axis=1))
    #
    # fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    # ax = ax.flat
    # ax[0].plot(tt, det_mean[:, 0])
    # ax[0].fill_between(tt, det_lc[:, 0], det_hc[:, 0], color='C0', alpha=0.1)
    # ax[0].plot(tt, cum_detected_samples[:, :, 0], c='k', alpha=0.005)
    # ax[0].plot(t, i_d_obs)
    # ax[0].set_ylim((-1, 4000))
    #
    # ax[1].plot(tt, i_h[:, :, 0], c='k', alpha=0.005)
    # ax[1].plot(t, i_h_obs)
    # ax[1].set_ylim((-1, 200))
    #
    # ax[2].plot(tt, i_icu[:, :, 0], c='k', alpha=0.005)
    # ax[2].plot(t, i_icu_obs)
    # ax[2].set_ylim((-1, 200))
    #
    # ax[3].plot(tt, d_icu_mean[:, 0])
    # ax[3].fill_between(tt, d_lc[:, 0], d_hc[:, 0], color='C0', alpha=0.1)
    # ax[3].plot(tt, d_icu[:, :, 0], c='k', alpha=0.005)
    # ax[3].plot(t, d_icu_obs)
    # ax[3].set_ylim((-1, 1000))
    #
    # plt.tight_layout()
    # plt.show()

    # fit to data

    ratio_as_detected = 0
    ratio_m_detected = 0.3
    ratio_s_detected = 1

    resampled_vars = model.sir_posterior(t, i_d_obs, i_h_obs + i_icu_obs, None, None, y0=y_tmin,
                                         ratio_as_detected=ratio_as_detected,
                                         ratio_m_detected=ratio_m_detected,
                                         ratio_s_detected=ratio_s_detected)

    scalar_vars = model.scalar_vars
    group_vars = model.group_vars

    fig, axes = plt.subplots(2, 5, figsize=(8, 2))
    i = 0
    axes = axes.flat
    for key, value in resampled_vars.items():
        # TODO: plot variables for multiple groups
        print(f'{key}: mean = {value.mean():.3f} - std = {value.std():.3f}')
        sns.kdeplot(value[:, 0], ax=axes[i])
        axes[i].set_title(key)
        i += 1

    plt.show()

    # free up memory for the new model
    del model

    # logging.info('Creating resampled model')
    # model = SamplingNInfectiousModel(
    #     nb_groups=1,
    #     **group_vars,
    #     **scalar_vars,
    #     **resampled_vars,
    #     y0=y0 * 1 * int(0.1 * nb_samples)
    # )
    #
    # logging.info('Solving resampled model')
    # tt = np.arange(t0, 300).astype(int)
    # y = model.solve(tt)
    #
    # logging.info("Plotting resampled model")
    # i_as = y[:, :, :, 2]
    # i_m = y[:, :, :, 3]
    # i_s = y[:, :, :, 4]
    # i_i = y[:, :, :, 5]
    # i_h = y[:, :, :, 6]
    # i_icu = y[:, :, :, 7]
    # r_as = y[:, :, :, 8]
    # r_m = y[:, :, :, 9]
    # r_h = y[:, :, :, 10]
    # r_icu = y[:, :, :, 11]
    # d_icu = y[:, :, :, 12]
    #
    # ratio_as_detected = 0
    # ratio_m_detected = 0.3
    # ratio_s_detected = 1
    #
    # cum_detected_samples = ratio_as_detected * (i_as + r_as) + ratio_m_detected * (i_m + r_m) \
    #                        + ratio_s_detected * (i_s + i_i + i_h + i_icu + r_h + r_icu + d_icu)
    #
    # det_mean = np.mean(cum_detected_samples, axis=1)
    # det_lc, det_hc = st.t.interval(0.95, len(cum_detected_samples) - 1, loc=det_mean, scale=st.sem(cum_detected_samples, axis=1))
    #
    # d_icu_mean = np.mean(d_icu, axis=1)
    # d_lc, d_hc = st.t.interval(0.95, len(d_icu_mean) - 1, loc=d_icu_mean, scale=st.sem(d_icu, axis=1))
    #
    # fig, ax = plt.subplots(1, 4, figsize=(12, 3))
    # ax = ax.flat
    # ax[0].plot(tt, det_mean[:, 0])
    # ax[0].fill_between(tt, det_lc[:, 0], det_hc[:, 0], color='C0', alpha=0.1)
    # ax[0].plot(tt, cum_detected_samples[:, :, 0], c='k', alpha=0.005)
    # ax[0].plot(t, i_d_obs)
    #
    # ax[1].plot(tt, i_h[:, :, 0], c='k', alpha=0.005)
    # ax[1].plot(t, i_h_obs)
    #
    # ax[2].plot(tt, i_icu[:, :, 0], c='k', alpha=0.005)
    # ax[2].plot(t, i_icu_obs)
    #
    # ax[3].plot(tt, d_icu_mean[:, 0])
    # ax[3].fill_between(tt, d_lc[:, 0], d_hc[:, 0], color='C0', alpha=0.1)
    # ax[3].plot(tt, d_icu[:, :, 0], c='k', alpha=0.005)
    # ax[3].plot(t, d_icu_obs)
    #
    # plt.tight_layout()
    # plt.show()






