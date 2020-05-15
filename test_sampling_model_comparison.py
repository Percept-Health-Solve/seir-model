import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from seir.sampling.model import SamplingNInfectiousModel

import logging
logging.basicConfig(level=logging.INFO)


if __name__ == '__main__':
    logging.info('Loading data')

    # read calibration data
    actual_hospitalisations = pd.read_excel('data/calibration.xlsx', sheet_name='Hospitalisations')
    actual_hospitalisations['Date'] = [pd.to_datetime(x, ).date() for x in actual_hospitalisations['Date']]

    # TODO: should check if file is downloaded: if not, download, then use the downloaded file
    actual_infections = pd.read_csv(
        'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_provincial_cumulative_timeline_confirmed.csv')
    actual_infections.rename(columns={'date': 'Date', 'total': 'Cum. Confirmed'}, inplace=True)
    actual_infections.index = pd.to_datetime(actual_infections['Date'], dayfirst=True)
    actual_infections = actual_infections.resample('D').mean().ffill().reset_index()
    actual_infections['Date'] = [pd.to_datetime(x, dayfirst=True).date() for x in actual_infections['Date']]

    # TODO: should check if file is downloaded: if not, download, then use the downloaded file
    reported_deaths = pd.read_csv(
        'https://raw.githubusercontent.com/dsfsi/covid19za/master/data/covid19za_timeline_deaths.csv')
    reported_deaths.rename(columns={'date': 'Date'}, inplace=True)
    reported_deaths['Date'] = [pd.to_datetime(x, dayfirst=True).date() for x in reported_deaths['Date']]
    actual_deaths = reported_deaths.groupby('Date').report_id.count().reset_index()
    actual_deaths.rename(columns={'report_id': 'Daily deaths'}, inplace=True)
    actual_deaths.index = pd.to_datetime(actual_deaths['Date'])
    actual_deaths = actual_deaths.resample('D').mean().fillna(0).reset_index()
    actual_deaths['Cum. Deaths'] = np.cumsum(actual_deaths['Daily deaths'])

    df_assa = pd.read_csv('data/scenarios/scenario1_daily_output_asymp_0.75_R0_3.0_imported_scale_2.5_lockdown_0.6'
                          '_postlockdown_0.75_ICU_0.2133_mort_1.0_asympinf_0.5.csv',
                          parse_dates=['Day'],
                          date_parser=lambda t: pd.to_datetime(t, format='%Y-%m-%d'))
    assa_detected = df_assa['Cumulative Detected'].to_numpy()
    assa_hospitalised = df_assa['Hospitalised'].to_numpy()
    assa_icu = df_assa['ICU'].to_numpy()
    assa_dead = df_assa['Dead'].to_numpy()
    assa_time = (df_assa['Day'] - pd.to_datetime('2020-03-27')).dt.days.to_numpy()

    df_start = pd.read_csv('data/Startpop_2density_0comorbidity.csv')
    N = df_start['Population'].sum()

    nb_samples = 1000
    nb_groups = 1

    r0 = np.random.uniform(2, 3.5, size=(nb_samples, 1))
    time_infectious = 2.3
    e0 = np.random.uniform(0.5, 1.5, size=(nb_samples, 1))
    y0 = np.zeros((nb_samples, nb_groups, 13))
    y0[:, :, 0] = N - e0
    y0[:, :, 1] = e0
    y0 = y0.reshape(-1)
    t0 = -100

    inf_as_prop = np.random.uniform(0.7, 0.9, size=(nb_samples, 1))

    model = SamplingNInfectiousModel(
        nb_groups=nb_groups,
        beta=r0 / time_infectious,
        rel_lockdown_beta=np.random.uniform(0.55, 0.65, size=(nb_samples, 1)),
        rel_postlockdown_beta=np.random.uniform(0.7, 0.8, size=(nb_samples, 1)),
        rel_beta_as=0.5,
        time_incubate=5.1,
        prop_as=0.75,
        prop_m=(1 - 0.75) * 0.1,
        time_infectious=time_infectious,
        time_s_to_h=6,
        time_h_to_c=8,
        time_h_to_r=10,
        time_c_to_r=10,
        time_c_to_d=6,
        prop_h_to_c=0.2133,
        prop_c_to_d=0.6,
        y0=y0
    )
    logging.info('Solving system')
    t_skip = 10
    ys = []
    ts = []
    y = None
    for t_start in range(t0, 300 - t_skip, t_skip):
        tt = np.arange(t_start, t_start + t_skip + 1)
        logging.info(f'Solving in range {tt}')
        if y is None:
            y = model.solve(tt, y0=y0)
        else:
            y = model.solve(tt, y0=y[-1].reshape(-1))
        ts.append(tt[:-1])
        ys.append(y[:-1])

    tt = np.concatenate(ts)
    y = np.concatenate(ys)

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

    det_samples = ratio_as_detected * (i_as + r_as) + ratio_m_detected * (i_m + r_m) \
                  + ratio_s_detected * (i_s + i_i + i_h + i_icu + r_h + r_icu + d_icu)

    logging.info('Plotting')
    assa_vars = [assa_detected, assa_hospitalised, assa_icu, assa_dead]
    sample_vars = [det_samples, i_h, i_icu, d_icu]
    titles = ['Detected', 'Hospitalised', 'ICU', 'Dead']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flat

    for i, (assa_var, sample_var, title) in enumerate(zip(assa_vars, sample_vars, titles)):
        mu = np.percentile(sample_var, 50, axis=1)[:, 0]
        low = np.percentile(sample_var, 2.5, axis=1)[:, 0]
        high = np.percentile(sample_var, 97.5, axis=1)[:, 0]

        axes[i].set_title(title)
        axes[i].plot(tt, mu, c='C0')
        axes[i].fill_between(tt, low, high, facecolor='C0', alpha=.2)
        axes[i].plot(assa_time, assa_var, c='C1')

    print(r_h[-1, 0, 0])
    print(r_icu[-1, 0, 0] + d_icu[-1, 0, 0])
    plt.tight_layout()
    plt.show()

