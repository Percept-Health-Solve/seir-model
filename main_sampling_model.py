import logging
import argparse
import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import pickle

import matplotlib.pyplot as plt
import seaborn as sns

from seir.sampling.model import SamplingNInfectiousModel

parser = argparse.ArgumentParser()
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of initial samples per run')
parser.add_argument('--ratio_resample', type=float, default=0.05, help='Proportion of resamples per run')
parser.add_argument('--output_dir', type=str, default='data/', help='Base directory in which to save files')
parser.add_argument('--model_name', type=str, default='model', help='Model name')
parser.add_argument('--nb_runs', type=int, default=1, help='Number of runs to perform')
parser.add_argument('--fit_detected', action='store_true', help='Fit the model to detected data')
parser.add_argument('--fit_hospitalised', action='store_true', help='Fit the model to detected data')
parser.add_argument('--fit_icu', action='store_true', help='Fit the model to detected data')
parser.add_argument('--fit_deaths', action='store_true', help='Fit the model to death data')
parser.add_argument('--fit_data', type=str, default='WC', help="Fit the model to 'WC' or 'national' data")
parser.add_argument('--load_prior_file', type=str, help='Load prior distributions from this file')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite any previous model saves')
parser.add_argument('--prop_as_range', type=float, default=[0.6, 0.9], nargs=2,
                    help='Lower and upper bounds for the prop_as uniform distribution')


def main():
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if args.load_prior_file:
        load_prior_file = Path(args.load_prior_file)
        # check if prior file is a file
        if not load_prior_file.is_file():
            raise ValueError(f"Given prior file '{args.load_prior_file}' is not a file or does not exist.")
    else:
        load_prior_file = None

    # check output directory
    if not output_dir.is_dir():
        raise ValueError(f'Given directory "{args.output_dir}" is either not a directory or does not exist.')

    # check if files exist:
    if args.nb_runs > 1:
        model_files = list(output_dir.glob(f'run*_{args.model_name}*'))
        if len(model_files) > 0 and not args.overwrite:
            raise ValueError(f'Given directory "{args.output_dir}" has saved model runs with name "{args.model_name}".'
                             f' Change the model name, output directory, or use --overwrite to overcome.')
    else:
        model_files = list(output_dir.glob(f'{args.model_name}_*'))
        if len(model_files) > 0 and not args.overwrite:
            raise ValueError(f'Given directory "{args.output_dir}" has saved model files with name "{args.model_name}".'
                             f' Change the model name, output directory, or use --overwrite to overcome.')

    # set up logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -- %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.INFO)
    logging.warning(f"Training model for {args.nb_runs} run(s) with {args.nb_samples} samples "
                    f"and {args.ratio_resample * 100:.1f}% resamples.")
    if args.fit_detected or args.fit_hospitalised or args.fit_icu or args.fit_deaths:
        logging.warning(f"Fitting to {'detected, ' if args.fit_detected else ''}"
                        f"{'hospitalised, ' if args.fit_hospitalised else ''}"
                        f"{'ICU, ' if args.fit_icu else ''}"
                        f"{'and ' if args.fit_detected or args.fit_hospitalised or args.fit_icu else ''}"
                        f"{'death' if args.fit_deaths else ''} cases")
    else:
        raise ValueError(f'Not fitting to any data! Use --fit_detected, --fit_icu, --fit_hospitalised, or --fit_deaths')

    # load data
    if args.fit_data.lower() == 'wc':
        t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs = load_data_WC()
        total_pop = 7000000
    elif args.fit_data.lower() == 'national':
        t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs = load_data_national()
        total_pop = 59000000
    else:
        raise ValueError("The --fitting_data flag is not specified correctly. "
                         f"Should be 'WC' or 'national', got '{args.fit_data}' instead.")

    detected_fit = i_d_obs if args.fit_detected else None
    h_fit = i_h_obs if args.fit_hospitalised else None
    icu_fit = i_icu_obs if args.fit_icu else None
    deaths_fit = d_icu_obs if args.fit_deaths else None

    # define the build_and_solve_model function without the output directory
    _build_and_solve_model = lambda save_dir: build_and_solve_model(t_obs,
                                                                    detected_fit,
                                                                    h_fit,
                                                                    icu_fit,
                                                                    deaths_fit,
                                                                    total_pop=total_pop,
                                                                    nb_samples=args.nb_samples,
                                                                    ratio_resample=args.ratio_resample,
                                                                    prop_as_range=args.prop_as_range,
                                                                    load_prior_file=load_prior_file,
                                                                    model_base=save_dir)

    if args.nb_runs > 1:
        for run in range(args.nb_runs):
            model_base = output_dir.joinpath(f'{run:02}_{args.model_name}')
            logging.info(f'Executing run {run + 1}')
            _build_and_solve_model(model_base)
            calculate_resample(t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs, total_pop=total_pop, model_base=model_base)
    else:
        model_base = output_dir.joinpath(f'{args.model_name}')
        _build_and_solve_model(model_base)
        calculate_resample(t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs, total_pop=total_pop, model_base=model_base)


def build_and_solve_model(t_obs,
                          i_d_obs=None,
                          i_h_obs=None,
                          i_icu_obs=None,
                          d_icu_obs=None,
                          nb_groups: int = 1,
                          total_pop: int = 1,
                          nb_samples: int = 1000000,
                          ratio_resample: float = 0.05,
                          load_prior_file: Path = None,
                          model_base: Path = Path('data/model'),
                          prop_as_range=None):
    if prop_as_range is None:
        prop_as_range = [0.6, 0.9]

    if not load_prior_file:
        logging.info('Setting priors')
        time_infectious = np.random.uniform(1.5, 2.6, size=(nb_samples, 1))
        prop_a = _uniform_from_range(prop_as_range, size=(nb_samples, 1))
        prop_m = (1 - prop_a) * 0.957  # ferguson gives approx 95.7 % of WC symptomatic not requiring hospitalisation
        prop_s_to_h = np.random.uniform(0.86, 0.91, size=(nb_samples, 1))
        prop_h_to_c = np.random.beta(119, 825, size=(nb_samples, 1))
        prop_h_to_d = np.random.beta(270, 1434, size=(nb_samples, 1))
        prop_c_to_d = np.random.beta(54, 65, size=(nb_samples, 1))

        # e0 = np.random.uniform(1E-8, 1e-6, size=(nb_samples, 1))
    else:
        # load df
        logging.info(f"Loading priors from {load_prior_file}")
        df_priors = pd.read_csv(load_prior_file)
        nb_prior_groups = int(df_priors['group'].max() + 1)
        nb_prior_samples = int(len(df_priors) / nb_prior_groups)
        nb_repeats = int(nb_samples / nb_prior_samples)

        # fix number of samples accordingly
        nb_samples = nb_repeats * nb_prior_samples

        # get mean variables
        get_mean = lambda x: df_priors[x].to_numpy() \
            .reshape(nb_prior_samples, nb_prior_groups).repeat(nb_repeats, axis=0)

        # set random vars
        random_scale = 0.01
        time_infectious = np.random.normal(get_mean('time_infectious'), scale=random_scale)
        prop_a = np.random.normal(get_mean('prop_a'), scale=random_scale).clip(min=0, max=1)
        prop_m = (1 - prop_a) * 0.957  # ferguson gives approx 95.7 % of WC symptomatic not requiring hospitalisation
        prop_s_to_h = np.random.normal(get_mean('prop_s_to_h'), scale=random_scale).clip(min=0, max=1)
        prop_h_to_c = np.random.normal(get_mean('prop_h_to_c'), scale=random_scale).clip(min=0, max=1)
        prop_h_to_d = np.random.normal(get_mean('prop_h_to_d'), scale=random_scale).clip(min=0, max=1)
        prop_c_to_d = np.random.normal(get_mean('prop_c_to_d'), scale=random_scale).clip(min=0, max=1)
        # e0 = np.random.normal(get_mean('e0'), scale=random_scale) / 7000000

    r0 = np.random.uniform(1.5, 3.5, size=(nb_samples, 1))
    beta = r0 / time_infectious
    rel_lockdown_beta = np.random.uniform(0.4, 1, size=(nb_samples, 1))
    rel_beta_as = np.random.uniform(0.3, 1, size=(nb_samples, 1))

    y0 = np.zeros((nb_samples, nb_groups, SamplingNInfectiousModel.nb_states))
    e0 = np.random.uniform(1e-8, 1e-6, size=(nb_samples, 1))
    y0[:, :, 0] = (1 - e0) * total_pop
    y0[:, :, 1] = e0 * total_pop
    y0 = y0.reshape(-1)
    t0 = -50

    model = SamplingNInfectiousModel(
        nb_groups=nb_groups,
        beta=beta,
        rel_lockdown_beta=rel_lockdown_beta,
        rel_postlockdown_beta=0.8,
        rel_beta_as=rel_beta_as,
        prop_a=prop_a,
        prop_m=prop_m,
        prop_s_to_h=prop_s_to_h,
        prop_h_to_c=prop_h_to_c,
        prop_h_to_d=prop_h_to_d,
        prop_c_to_d=prop_c_to_d,
        time_incubate=5.1,
        time_infectious=time_infectious,
        time_s_to_h=6,
        time_s_to_c=6,
        time_h_to_c=10,
        time_h_to_r=10.1,
        time_h_to_d=9.9,
        time_c_to_r=18.3,
        time_c_to_d=18.8,
        y0=y0
    )

    # get y_t.min() from y0
    # logging.info('Solving for y at minimum data time')
    # tt = np.linspace(t0, t_obs.min(), 20)
    # y_tmin = model.solve(tt)[-1]
    # y_tmin = y_tmin.reshape(-1)

    # fit to data

    ratio_as_detected = 0
    ratio_m_detected = 0.3
    ratio_s_detected = 1

    model.calculate_sir_posterior(t0, t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs,
                                  ratio_as_detected=ratio_as_detected,
                                  ratio_m_detected=ratio_m_detected,
                                  ratio_s_detected=ratio_s_detected,
                                  ratio_resample=ratio_resample,
                                  smoothing=1)

    sample_vars = model.sample_vars
    resample_vars = model.resample_vars
    scalar_vars = model.scalar_vars
    group_vars = model.group_vars

    calc_sample_vars = model.calculated_sample_vars
    calc_resample_vars = model.calculated_resample_vars

    sample_vars['e0'] = e0
    e0_resample = e0[model.resample_indices]  # TODO: Make y0 resampling a thing

    resample_vars['e0'] = e0_resample
    scalar_vars['t0'] = t0

    # save variables
    save_vars_to_csv(model, base=model_base)

    # plot variables of interest
    logging.info('Plotting prior and posterior distributions')
    sns.set(style='darkgrid')
    fig, axes = plt.subplots(4, 5, figsize=(11, 11))
    i = 0
    axes = axes.flat
    # reshape to a dataframe for pair plotting
    reshaped_resample_vars = {}
    for key, value in resample_vars.items():
        reshaped_resample_vars[key] = value.reshape(-1)
    reshaped_resample_vars['group'] = np.asarray([[i] * model.nb_resamples for i in range(nb_groups)]).reshape(-1)
    df_resample = pd.DataFrame(reshaped_resample_vars)
    try:
        for key, value in resample_vars.items():
            # TODO: plot variables for multiple groups
            logging.info(f'{key}: mean = {value.mean():.3f} - std = {value.std():.3f}')
            sns.distplot(value[:, 0], ax=axes[i], color='C0')
            sns.distplot(sample_vars[key][:, 0], ax=axes[i], color='C1')
            axes[i].set_title(key)
            i += 1
        logging.info('Adding calculated variables')
        for key, value in calc_resample_vars.items():
            logging.info(f'{key}: mean = {value.mean():.3f} - std = {value.std():.3f}')
            sns.distplot(value[:, 0], ax=axes[i], color='C0')
            sns.distplot(calc_sample_vars[key][:, 0], ax=axes[i], color='C1')
            axes[i].set_title(key)
            i += 1
    except np.linalg.LinAlgError:
        logging.warning(f'Plotting of priors and posteriors failed due to posterior collapse')

    plt.tight_layout()
    fig.savefig(f'{model_base}_priors_posterior.png')
    plt.clf()

    logging.info('Building joint distribution plot')
    g = sns.PairGrid(df_resample, corner=True, hue="group")
    try:
        g = g.map_lower(sns.kdeplot, colors='C0')
        g = g.map_diag(sns.distplot)
    except np.linalg.LinAlgError:
        logging.warning(f'Plotting of joint distribution failed due to posterior collapse')
    g.savefig(f'{model_base}_joint_posterior.png')
    plt.clf()

    del model, fig, g


def save_vars_to_csv(model: SamplingNInfectiousModel, base='data/samples'):
    nb_groups = model.nb_groups
    scalar_vars = model.scalar_vars
    group_vars = model.group_vars
    sample_vars = model.sample_vars
    resample_vars = model.resample_vars
    log_weights = model.log_weights

    nb_samples = model.nb_samples
    nb_resamples = model.nb_resamples

    logging.info(f'Saving model variables to {base}_*.csv')
    # need to reshape sample vars
    reshaped_sample_vars = {}
    for key, value in sample_vars.items():
        reshaped_sample_vars[key] = value.reshape(-1)
    reshaped_sample_vars['group'] = np.asarray([[i] * nb_samples for i in range(nb_groups)]).reshape(-1)
    # need to reshape resample vars
    reshaped_resample_vars = {}
    for key, value in resample_vars.items():
        reshaped_resample_vars[key] = value.reshape(-1)
    reshaped_resample_vars['group'] = np.asarray([[i] * nb_resamples for i in range(nb_groups)]).reshape(-1)

    # define dfs
    df_sample = pd.DataFrame(reshaped_sample_vars)
    df_sample['log_weights'] = log_weights
    df_resample = pd.DataFrame(reshaped_resample_vars)

    # save files
    df_sample.to_csv(f"{base}_sample.csv", index=False)
    df_resample.to_csv(f"{base}_resample.csv", index=False)
    with open(f'{base}_scalar.pkl', 'wb') as f:
        pickle.dump(scalar_vars, f)
    with open(f'{base}_group.pkl', 'wb') as f:
        pickle.dump(group_vars, f)


def load_data_WC(remove_small: bool = True):
    # get data
    logging.info('Loading WC data')
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
    df_hosp_icu = pd.read_csv('data/WC_data.csv',
                              parse_dates=['date'],
                              date_parser=lambda t: pd.to_datetime(t, format='%Y-%m-%d'))

    # the WC reporting has some lag, so choose a date to set as the maximum date for each of the dfs
    max_date = pd.to_datetime('2020/05/14')
    df_deaths = df_deaths[df_deaths['date'] <= max_date]
    df_confirmed = df_confirmed[df_confirmed['date'] <= max_date]
    df_hosp_icu = df_hosp_icu[df_hosp_icu['date'] <= max_date]

    df_deaths = df_deaths.sort_values('date')
    df_confirmed = df_confirmed.sort_values('date')
    df_hosp_icu = df_hosp_icu.sort_values('date')

    logging.info('Taking intersection of dates in all dataframes')
    max_date = np.min([df_confirmed['date'].max(), df_hosp_icu['date'].max()])
    logging.info(f'Maximum date at which all data sources had data: {max_date}')
    df_confirmed = df_confirmed[df_confirmed['date'] < max_date]
    df_hosp_icu = df_hosp_icu[df_hosp_icu['date'] < max_date]

    df_deaths = df_deaths[['date', 'WC']]
    df_confirmed = df_confirmed[['date', 'WC']]

    logging.info('Linearly interpolating missing data')
    df_confirmed = df_confirmed.interpolate(method='linear')

    logging.info('Setting date of lockdown 2020-03-27 to day 0')
    df_deaths['Day'] = (df_deaths['date'] - pd.to_datetime('2020-03-27')).dt.days
    df_confirmed['Day'] = (df_confirmed['date'] - pd.to_datetime('2020-03-27')).dt.days
    df_hosp_icu['Day'] = (df_hosp_icu['date'] - pd.to_datetime('2020-03-27')).dt.days

    logging.info('Merging data sources')
    df_merge = df_confirmed.merge(df_deaths, on='Day', how='left', suffixes=('_confirmed', '_deaths'))
    df_merge = df_merge.merge(df_hosp_icu, on='Day', how='left')
    df_merge = df_merge.interpolate(method='linear')
    df_merge = df_merge[
        ['date_confirmed', 'WC_confirmed', 'Cum Deaths', 'Current Hospitalisations', 'Current ICU', 'Day']]
    df_merge = df_merge.fillna(0)

    logging.info('Casting data')
    df_merge['WC_confirmed'] = df_merge['WC_confirmed'].astype(int)
    df_merge['Cum Deaths'] = df_merge['Cum Deaths'].astype(int)
    df_merge['Day'] = df_merge['Day'].astype(int)

    # remove small observations
    if remove_small:
        logging.info('Filtering out data that contains small counts (as not to bias the poisson model)')
        df_merge = df_merge[df_merge['Cum Deaths'] > 5]
        logging.info(f"Minimum data day after filtering: {df_merge['Day'].min()}")

    t_obs = df_merge['Day'].to_numpy()
    i_d_obs = df_merge['WC_confirmed'].to_numpy()
    i_h_obs = df_merge['Current Hospitalisations'].to_numpy()
    i_icu_obs = df_merge['Current ICU'].to_numpy()
    d_icu_obs = df_merge['Cum Deaths'].to_numpy()

    return t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs


def load_data_national(remove_small: bool = True):
    logging.info('Loading national data')
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

    df_deaths = df_deaths.sort_values('date')
    df_confirmed = df_confirmed.sort_values('date')

    df_deaths = df_deaths[['date', 'total']]
    df_confirmed = df_confirmed[['date', 'total']]

    logging.info('Linearly interpolating missing data')
    df_confirmed = df_confirmed.interpolate(method='linear')
    df_deaths = df_deaths.interpolate(method='linear')

    logging.info('Setting date of lockdown 2020-03-27 to day 0')
    df_deaths['Day'] = (df_deaths['date'] - pd.to_datetime('2020-03-27')).dt.days
    df_confirmed['Day'] = (df_confirmed['date'] - pd.to_datetime('2020-03-27')).dt.days

    logging.info('Merging data sources')
    df_merge = df_confirmed.merge(df_deaths, on='Day', how='left', suffixes=('_confirmed', '_deaths'))
    df_merge = df_merge.interpolate(method='linear')
    df_merge = df_merge[
        ['date_confirmed', 'total_confirmed', 'total_deaths', 'Day']]
    df_merge = df_merge.fillna(0)

    df_merge['Day'] = df_merge['Day'].astype(int)
    df_merge['total_confirmed'] = df_merge['total_confirmed'].astype(int)
    df_merge['total_deaths'] = df_merge['total_deaths'].astype(int)

    if remove_small:
        logging.info('Filtering out data that contains small counts (as not to bias the poisson model)')
        df_merge = df_merge[df_merge['total_deaths'] > 5]
        logging.info(f"Minimum data day after filtering: {df_merge['Day'].min()}")

    t_obs = df_merge['Day'].to_numpy()
    i_d_obs = df_merge['total_confirmed'].to_numpy()
    d_icu_obs = df_merge['total_deaths'].to_numpy()

    logging.warning('No ICU or hospital national data, will only be able to fit to detected and death cases.')

    return t_obs, i_d_obs, None, None, d_icu_obs


def calculate_resample(t_obs,
                       i_d_obs,
                       i_h_obs,
                       i_icu_obs,
                       d_icu_obs,
                       total_pop: int = 1,
                       model_base='data/model'):
    with open(f'{model_base}_scalar.pkl', 'rb') as f:
        scalar_vars = pickle.load(f)
    with open(f'{model_base}_group.pkl', 'rb') as f:
        group_vars = pickle.load(f)
    df_resample = pd.read_csv(f'{model_base}_resample.csv')
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
    y0[:, :, 0] = (1 - e0_resample) * total_pop
    y0[:, :, 1] = e0_resample * total_pop
    y0 = y0.reshape(-1)

    logging.info('Creating resampled model')
    model = SamplingNInfectiousModel(
        nb_groups=nb_groups,
        **resample_vars,
        **group_vars,
        **scalar_vars,
        y0=y0
    )

    logging.info('Solving resampled model')
    # have to do this nonsense fiesta to prevent segmentation faults
    t_skip = 10
    ys = []
    ts = []
    y = None
    for t_start in range(t0, 300 - t_skip, t_skip):
        tt = np.arange(t_start, t_start + t_skip + 1)
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
    t_date = [datetime.date(2020, 3, 27) + datetime.timedelta(days=int(day)) for day in t_obs]

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
    logging.info('Generating death percentile samples')
    d_med = np.median(d)
    d_25 = np.percentile(d, 2.5)
    d_975 = np.percentile(d, 97.5)

    arg_med = np.argmin((d - d_med) ** 2)
    arg_25 = np.argmin((d - d_25) ** 2)
    arg_975 = np.argmin((d - d_975) ** 2)

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
    logging.info(f'Saving death percentile parameters to {model_base}_death_percentile_params.csv')
    df_percentiles.to_csv(f'{model_base}_death_percentile_params.csv', index=False)

    # plot final deaths as a kdeplot
    fig = plt.figure(figsize=(8, 5))
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
    fig.savefig(f'{model_base}_death_distribution.png')
    plt.clf()

    # TODO: use code from utils
    ratio_as_detected = 0
    ratio_m_detected = 0.3
    ratio_s_detected = 1

    cum_detected_samples = ratio_as_detected * (i_a + r_a) + ratio_m_detected * (i_m + r_m) \
                           + ratio_s_detected * (i_s + i_h + i_c + h_c + h_r + h_d + c_r + c_d + r_h + r_c + d_h + d_c)

    logging.info('Plotting solutions')

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    pred_vars = [cum_detected_samples, h_r + h_d + h_c, c_r + c_d, d_c + d_h]
    obs_vars = [i_d_obs, i_h_obs, i_icu_obs, d_icu_obs]
    titles = ['Detected', 'Hospitalised', 'ICU', 'Deaths']

    logging.info('Generating timeseries summary stats and plotting')
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
                if obs_vars[i] is not None:
                    axes[j, i].plot(t_date, obs_vars[i], 'x', c='C1')
                    axes[j, i].set_ylim((-1, np.max(obs_vars[i]) * 1.05))
                else:
                    axes[j, i].set_ylim((-1, np.max(mu[110]) * 1.05))
                axes[j, i].set_xlim(
                    (pd.to_datetime('2020/03/27'), np.max(t_date) + datetime.timedelta(days=1))
                )
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
    logging.info(f'Saving plot to {model_base}_prediction.png')
    plt.savefig(f'{model_base}_prediction.png')
    plt.clf()

    logging.info(f'Saving timeseries summary stats to {model_base}_prediction.csv')
    groups = np.asarray([[i] * len(tt) for i in range(nb_groups)]).reshape(-1)
    summary_stats['group'] = groups
    df_stats = pd.DataFrame(summary_stats)
    df_stats.insert(0, 'Date', tt_date)
    df_stats.to_csv(f'{model_base}_prediction.csv', index=False)


def _uniform_from_range(range, size=(1,)):
    if range[0] == range[1]:
        return range[0]
    else:
        return np.random.uniform(range[0], range[1], size=size)


if __name__ == '__main__':
    main()
