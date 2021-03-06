# NB: You have to run main_sampling.py in order for this script to function

import numpy as np
import pandas as pd
import pickle

import datetime

from seir.sampling.model import SamplingNInfectiousModel

import matplotlib.pyplot as plt
import seaborn as sns

import scipy.stats as st

import logging
logging.basicConfig(level=logging.INFO)

if __name__ == '__main__':
    sns.set(style='darkgrid')

    # get data
    logging.info('Loading data')
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

    logging.info('Taking intersection of dates in all dataframes')
    max_date = np.min([df_deaths['date'].max(), df_confirmed['date'].max(), df_hosp_icu['Date'].max()])
    logging.info(f'Maximum date at which all data sources had data: {max_date}')
    df_confirmed = df_confirmed[df_confirmed['date'] < max_date]

    df_deaths = df_deaths[['date', 'WC']]
    df_confirmed = df_confirmed[['date', 'WC']]

    logging.info('Linearly interpolating missing data')
    df_confirmed = df_confirmed.interpolate(method='linear')

    logging.info('Setting date of lockdown 2020-03-27 to day 0')
    df_deaths['Day'] = (df_deaths['date'] - pd.to_datetime('2020-03-27')).dt.days
    df_confirmed['Day'] = (df_confirmed['date'] - pd.to_datetime('2020-03-27')).dt.days
    df_hosp_icu['Day'] = (df_hosp_icu['Date'] - pd.to_datetime('2020-03-27')).dt.days

    logging.info('Merging data sources')
    df_merge = df_confirmed.merge(df_deaths, on='Day', how='left', suffixes=('_confirmed', '_deaths'))
    df_merge = df_merge.merge(df_hosp_icu, on='Day', how='left')
    df_merge = df_merge.interpolate(method='linear')
    df_merge = df_merge[
        ['date_confirmed', 'WC_confirmed', 'WC_deaths', 'Current hospitalisations', 'Current ICU', 'Day']]
    df_merge = df_merge.fillna(0)

    logging.info('Casting data')
    df_merge['WC_confirmed'] = df_merge['WC_confirmed'].astype(int)
    df_merge['WC_deaths'] = df_merge['WC_deaths'].astype(int)
    df_merge['Day'] = df_merge['Day'].astype(int)

    t = df_merge['Day'].to_numpy()
    i_h_obs = df_merge['Current hospitalisations']
    i_icu_obs = df_merge['Current ICU']
    i_d_obs = df_merge['WC_confirmed']
    d_icu_obs = df_merge['WC_deaths']


    # load solution data
    nb_runs = 5

    for run in range(nb_runs):
        with open(f'data/sampling-runs/run{run:02}_scalar.pkl', 'rb') as f:
            scalar_vars = pickle.load(f)
        with open(f'data/sampling-runs/run{run:02}_group.pkl', 'rb') as f:
            group_vars = pickle.load(f)
        df_resample = pd.read_csv(f'data/sampling-runs/run{run:02}_resample.csv')
        # df_resample = df_resample.sample(n=10)
        nb_groups = df_resample['group'].max() + 1
        nb_samples = int(len(df_resample) / nb_groups)

        logging.info(f"Samples: {nb_samples}")
        logging.info(f"Groups: {nb_groups}")

        resample_vars = {}
        for col in df_resample:
            resample_vars[col] = df_resample[col].to_numpy()
        for key, value in resample_vars.items():
            resample_vars[key] = value.reshape(nb_samples, nb_groups)

        t0 = scalar_vars.pop('t0')
        e0_resample = resample_vars.pop('e0')
        groups = resample_vars.pop('group')

        y0 = np.zeros((nb_samples, nb_groups, SamplingNInfectiousModel.nb_states))
        y0[:, :, 0] = 7000000 - e0_resample
        y0[:, :, 1] = e0_resample
        y0 = y0.reshape(-1)

        logging.info('Creating resampled model')
        model = SamplingNInfectiousModel(
            nb_groups=nb_groups,
            **resample_vars,
            **group_vars,
            **scalar_vars,
            y0=y0
        )

        logging.info('Solving model')
        # have to do this nonsense fiesta to prevent segmentation faults
        t_skip = 10
        ys = []
        ts = []
        y = None
        for t_start in range(t0, 300-t_skip, t_skip):
            tt = np.arange(t_start, t_start + t_skip+1)
            logging.info(f'Solving in range {tt}')
            if y is None:
                y = model.solve(tt, y0=y0, return_as_seird=False)
            else:
                y = model.solve(tt, y0=y[-1].reshape(-1), return_as_seird=False)
            ts.append(tt[:-1])
            ys.append(y[:-1])

        tt = np.concatenate(ts)
        y = np.concatenate(ys)

        # turn time values into dates
        tt_date = [datetime.date(2020, 3, 27) + datetime.timedelta(days=int(day)) for day in tt]
        t_date = [datetime.date(2020, 3, 27) + datetime.timedelta(days=int(day)) for day in t]

        i_a = y[:, :, :, 2]
        i_m = y[:, :, :, 3]
        i_s = y[:, :, :, 4]
        i_h = y[:, :, :, 5]
        i_c = y[:, :, :, 6]
        h_r = y[:, :, :, 7]
        h_c = y[:, :, :, 8]
        h_d = y[:, :, :, 9]
        c_r = y[:, :, :, 10]
        c_d = y[:, :, :, 11]
        r_a = y[:, :, :, 12]
        r_m = y[:, :, :, 13]
        r_h = y[:, :, :, 14]
        r_c = y[:, :, :, 15]
        d_h = y[:, :, :, 16]
        d_c = y[:, :, :, 17]

        # total deaths on last predicted day
        d = d_h + d_c
        d = d[-1]
        d = np.sum(d, axis=1)

        # want to find the samples the best approximates median and 95% CIs
        d_med = np.median(d)
        d_25 = np.percentile(d, 2.5)
        d_975 = np.percentile(d, 97.5)

        arg_med = np.argmin((d-d_med)**2)
        arg_25 = np.argmin((d-d_25)**2)
        arg_975 = np.argmin((d-d_975)**2)

        # find underlying samples that correspond to these cases
        percentile_vars = {}
        for key, value in resample_vars.items():
            percentile_vars[key] = value[[arg_25, arg_med, arg_975]]
        for key, value in scalar_vars.items():
            percentile_vars[key] = np.array([[value], [value], [value]])
        for key, value in group_vars.items():
            percentile_vars[key] = np.concatenate([value, value, value], axis=0)
        for key, value in model.calculated_sample_vars.items():
            percentile_vars[key] = value[[arg_25, arg_med, arg_975]]

        # reshape this for converting to a df
        reshaped_percentile_vars = {}
        for key, value in percentile_vars.items():
            reshaped_percentile_vars[key] = value.reshape(-1)
        reshaped_percentile_vars['group'] = np.asarray([[i] * 3 for i in range(nb_groups)]).reshape(-1)
        df_percentiles = pd.DataFrame(reshaped_percentile_vars)
        df_percentiles.insert(0, column='Percentile', value=[2.5, 50, 97.5])
        df_percentiles.to_csv(f'data/sampling-runs/run{run:02}_death_percentile_params.csv', index=False)

        # plot final deaths as a kdeplot
        sns.kdeplot(d)
        ax = plt.gca()
        ax.axvline(d_25, c='C1', ls='--', label='2.5 Percentile')
        ax.axvline(d_med, c='C2', ls='--', label='Median')
        ax.axvline(d_975, c='C3', ls='--', label='97.5 Percentile')
        ax.set_xlabel('Deaths')
        ax.set_ylabel('Probability density')
        ax.set_title(f'Deaths distribution on {tt_date[-1]}')
        ax.legend()
        plt.tight_layout()
        plt.gcf().savefig(f'data/sampling-runs/run{run:02}_death_distribution.png')
        plt.clf()

        ratio_as_detected = 0
        ratio_m_detected = 0.3
        ratio_s_detected = 1
        ratio_resample = 0.1

        cum_detected_samples = ratio_as_detected * (i_a + r_a) + ratio_m_detected * (i_m + r_m) \
                               + ratio_s_detected * (i_s + i_h + i_c + h_c + h_r + h_d + c_r + c_d + r_h + r_c + d_h + d_c)

        logging.info('Plotting solutions')

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))

        pred_vars = [cum_detected_samples, h_r + h_d + h_c, c_r + c_d, d_c + d_h]
        obs_vars = [i_d_obs, i_h_obs, i_icu_obs, d_icu_obs]
        titles = ['Detected', 'Hospitalised', 'ICU', 'Deaths']

        summary_stats = {}
        for j, row in enumerate(axes):
            for i in range(len(row)):
                mu = np.median(pred_vars[i], axis=1)
                low = np.percentile(pred_vars[i], 2.5, axis=1)
                high = np.percentile(pred_vars[i], 97.5, axis=1)

                axes[j, i].plot(tt_date, mu, c='C0')
                axes[j, i].fill_between(tt_date, low[:, 0], high[:, 0], alpha=.2, facecolor='C0')
                for tick in axes[j, i].get_xticklabels():
                    tick.set_rotation(45)
                # axes[i].plot(tt, pred_vars[i][:, :, 0], c='k', alpha=0.1)
                if j == 0:
                    axes[j, i].plot(t_date, obs_vars[i], 'x', c='C1')
                    axes[j, i].set_xlim((np.min(t_date) - datetime.timedelta(days=1), np.max(t_date) + datetime.timedelta(days=1)))
                    axes[j, i].set_ylim((-1, np.max(obs_vars[i])*1.05))
                    axes[j, i].set_title(titles[i])
                if j == 1:
                    axes[j, i].plot(tt_date, pred_vars[i][:, arg_25], c='C1', ls='--')
                    axes[j, i].plot(tt_date, pred_vars[i][:, arg_med], c='C2', ls='--')
                    axes[j, i].plot(tt_date, pred_vars[i][:, arg_975], c='C3', ls='--')
                    axes[j, i].set_xlabel('Date')

                summary_stats[f'{titles[i]}_mean'] = mu.reshape(-1)
                summary_stats[f'{titles[i]}_2.5CI'] = low.reshape(-1)
                summary_stats[f'{titles[i]}_97.5CI'] = high.reshape(-1)

        plt.tight_layout()
        plt.savefig(f'data/sampling-runs/run{run:02}_prediction.png')
        plt.clf()

        logging.info('Saving summary stats')
        groups = np.asarray([[i] * len(tt) for i in range(nb_groups)]).reshape(-1)
        summary_stats['group'] = groups
        df_stats = pd.DataFrame(summary_stats)
        df_stats.insert(0, 'Date', tt_date)
        print(df_stats.head())
        df_stats.to_csv(f'data/sampling-runs/run{run:02}_prediction.csv', index=False)

