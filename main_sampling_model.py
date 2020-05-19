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
parser.add_argument('--model_dir', type=str, default='data/', help='Base directory in which to save files')
parser.add_argument('--model_name', type=str, default='model', help='Model name')
parser.add_argument('--nb_runs', type=int, default=1, help='Number of runs to perform')
parser.add_argument('--fit_detected', action='store_true', help='Fits the model to detected data')
parser.add_argument('--fit_hospitalised', action='store_true', help='Fits the model to detected data')
parser.add_argument('--fit_icu', action='store_true', help='Fits the model to detected data')
parser.add_argument('--fit_deaths', action='store_true', help='Fits the model to death data')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite any previous model saves')
parser.add_argument('--prop_as_range', type=float, default=[0.6, 0.9], nargs=2,
                    help='Lower and upper bounds for the prop_as uniform distribution')


def main():
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

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
                        f"{'and, ' if args.fit_detected or args.fit_hospitalised or args.fit_icu else ''}"
                        f"{'death' if args.fit_deaths else ''} cases")
    else:
        raise ValueError(f'Not fitting to any data! Use --fit_detected, --fit_icu, --fit_hospitalised, or --fit_deaths')

    # load data
    t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs = load_data()

    if not args.fit_detected:
        i_d_obs = None
    if not args.fit_hospitalised:
        i_h_obs = None
    if not args.fit_icu:
        i_icu_obs = None
    if not args.fit_deaths:
        d_icu_obs = None

    # define the build_and_solve_model function without the output directory
    _build_and_solve_model = lambda save_dir: build_and_solve_model(t_obs,
                                                                    i_d_obs,
                                                                    i_h_obs,
                                                                    i_icu_obs,
                                                                    d_icu_obs,
                                                                    nb_samples=args.nb_samples,
                                                                    ratio_resample=args.ratio_resample,
                                                                    prop_as_range=args.prop_as_range,
                                                                    model_dir=save_dir)

    if args.nb_runs > 1:
        for run in range(args.nb_runs):
            logging.info(f'Executing run {run + 1}')
            _build_and_solve_model(output_dir.joinpath(f'{run:02}_{args.model_name}'))
    else:
        _build_and_solve_model(output_dir.joinpath(f'{args.model_name}'))


def build_and_solve_model(t_obs,
                          i_d_obs=None,
                          i_h_obs=None,
                          i_icu_obs=None,
                          d_icu_obs=None,
                          nb_groups: int = 1,
                          nb_samples: int = 1000000,
                          ratio_resample: float = 0.05,
                          model_dir: Path = Path('data/model'),
                          prop_as_range=None):
    if prop_as_range is None:
        prop_as_range = [0.6, 0.9]

    r0 = np.random.uniform(2, 3.5, size=(nb_samples, 1))
    time_infectious = np.random.uniform(2, 2.6, size=(nb_samples, 1))
    e0 = np.random.uniform(0.001, 5, size=(nb_samples, 1))
    rel_lockdown_beta = np.random.uniform(0, 1, size=(nb_samples, 1))

    # set prop_a
    prop_a = _uniform_from_range(prop_as_range, size=(nb_samples, 1))

    y0 = np.zeros((nb_samples, nb_groups, SamplingNInfectiousModel.nb_states))
    y0[:, :, 0] = 7000000 - e0
    y0[:, :, 1] = e0
    y0 = y0.reshape(-1)
    t0 = -50

    model = SamplingNInfectiousModel(
        nb_groups=nb_groups,
        beta=r0 / time_infectious,
        rel_lockdown_beta=rel_lockdown_beta,
        rel_postlockdown_beta=0.8,
        rel_beta_as=np.random.uniform(0.3, 1, size=(nb_samples, 1)),
        prop_a=prop_a,
        prop_m=(1 - prop_a) * 0.957,  # ferguson gives approx 95.7 % of WC symptomatic not requiring hospitalisation
        prop_s_to_h=np.random.uniform(0.8, 0.95, size=(nb_samples, 1)),
        prop_h_to_c=np.random.beta(34, 191, size=(nb_samples, 1)),
        prop_h_to_d=np.random.beta(41, 150, size=(nb_samples, 1)),
        prop_c_to_d=np.random.beta(20, 14, size=(nb_samples, 1)),
        time_incubate=5.1,
        time_infectious=time_infectious,
        time_s_to_h=6,
        time_s_to_c=6,
        time_h_to_c=10,
        time_h_to_r=9.8,
        time_h_to_d=11.1,
        time_c_to_r=18.1,
        time_c_to_d=17.2,
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
    save_vars_to_csv(model, base=model_dir)

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
    fig.savefig(f'{model_dir}_priors_posterior.png')

    logging.info('Building joint distribution plot')
    g = sns.PairGrid(df_resample, corner=True, hue="group")
    try:
        g = g.map_lower(sns.kdeplot, colors='C0')
        g = g.map_diag(sns.distplot)
        g.savefig(f'{model_dir}_joint_posterior.png')
    except np.linalg.LinAlgError:
        logging.warning(f'Plotting of joint distribution failed due to posterior collapse')

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


def load_data(remove_small: bool = True):
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
    df_hosp_icu = pd.read_csv('data/WC_data.csv',
                              parse_dates=['date'],
                              date_parser=lambda t: pd.to_datetime(t, format='%Y-%m-%d'))

    df_deaths = df_deaths.sort_values('date')
    df_confirmed = df_confirmed.sort_values('date')
    df_hosp_icu = df_hosp_icu.sort_values('Date')

    logging.info('Taking intersection of dates in all dataframes')
    max_date = np.min([df_confirmed['date'].max(), df_hosp_icu['Date'].max()])
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
        ['date_confirmed', 'WC_confirmed', 'Cum Deaths', 'Current hospitalisations', 'Current ICU', 'Day']]
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

    t = df_merge['Day'].to_numpy()
    i_d_obs = df_merge['WC_confirmed'].to_numpy()
    i_h_obs = df_merge['Current hospitalisations'].to_numpy()
    i_icu_obs = df_merge['Current ICU'].to_numpy()
    d_icu_obs = df_merge['Cum Deaths'].to_numpy()

    return t, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs


def _uniform_from_range(range, size=(1,)):
    if range[0] == range[1]:
        return range[0]
    else:
        return np.random.uniform(range[0], range[1], size=size)


if __name__ == '__main__':
    main()
