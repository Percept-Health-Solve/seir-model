import logging
import argparse
import datetime
import json
import pickle

import numpy as np
import pandas as pd
from scipy.special import softmax

import matplotlib.pyplot as plt
import seaborn as sns

from seir.sampling.model import SamplingNInfectiousModel
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--nb_samples', type=int, default=1000000, help='Number of initial samples per run')
parser.add_argument('--ratio_resample', type=float, default=0.05, help='Proportion of resamples per run')
parser.add_argument('--output_dir', type=str, default='data/', help='Base directory in which to save files')
parser.add_argument('--model_name', type=str, default='model', help='Model name')
parser.add_argument('--nb_runs', type=int, default=1, help='Number of runs to perform')
parser.add_argument('--age_groups', action='store_true', help='Split the population into age bands when fitting')

parser.add_argument('--fit_detected', action='store_true', help='Fit the model to detected data')
parser.add_argument('--fit_hospitalised', action='store_true', help='Fit the model to hospitalised data')
parser.add_argument('--fit_icu', action='store_true', help='Fit the model to ICU data')
parser.add_argument('--fit_deaths', action='store_true', help='Fit the model to death data')
parser.add_argument('--fit_data', type=str, default='WC', help="Fit the model to 'WC' or 'national' data")

parser.add_argument('--load_prior_file', type=str, help='Load prior distributions from this file')
parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite any previous model saves')
parser.add_argument('--from_config', type=str, help='Load model config from given json file')
parser.add_argument('--only_process_runs', action='store_true')
parser.add_argument('--only_plot', action='store_true')

parser.add_argument('--t0', type=int, default=-50, help='Day relative to the start of lockdown to seed the model.')
parser.add_argument('--e0_range', type=float, default=[0, 1e-5], nargs=2,
                    help='Lower and upper bounds to for the uniform prior distribution of e0.')

parser.add_argument('--r0_range', type=float, default=[1.5, 3.5], nargs=2,
                    help='Lower and upper bounds to for the uniform prior distribution of R0.')
parser.add_argument('--rel_beta_as_range', type=float, default=[0.3, 1], nargs=2,
                    help='Lower and upper bounds to for the uniform prior distribution of the relative infectivity '
                         'level of asymptomatic cases.')
parser.add_argument('--rel_lockdown5_beta_range', type=float, default=[0.4, 1], nargs=2,
                    help='Lower and upper bounds for the uniform prior distribution of the relative beta '
                         'experience during level 5 lockdown.')
parser.add_argument('--rel_postlockdown_beta', type=float, default=0.8,
                    help='The relative infectivity post all levels of lockdown.')

parser.add_argument('--prop_as_range', type=float, default=[0.5, 0.5], nargs=2,
                    help='Lower and upper bounds to for the uniform prior distribution of the proportion asymptomatic.')
parser.add_argument('--prop_s_to_h_range', type=float, default=[0.8875, 0.8875], nargs=2,
                    help='Lower and upper bounds to for the uniform prior distribution of proportion severe moving '
                         'to hospital.')

parser.add_argument('--time_infectious_range', type=float, default=[1.5, 2.6], nargs=2,
                    help='Lower and upper bounds to for the uniform prior distribution of time of infectiousness.')

parser.add_argument('--fit_interval', type=int, default=0,
                    help='Number of days between which to consider fitting. Zero indicates fitting to all data.')
parser.add_argument('--fit_new_deaths', action='store_true', help='Fit to new deaths instead of cumulative deaths')
parser.add_argument('--contact_heterogeneous', action='store_true',
                    help='Use Kong et al (2016) method of employing contact heterogeneity in susceptible population')
parser.add_argument('--contact_k', type=float, default=0.25,
                    help='Value of k describing contact heterogenity in Kong et al 2016.')

parser.add_argument('--likelihood', type=str, default='lognormal',
                    help="Method of calculating likehood function. Currently, only supports 'lognormal' and 'poisson'.")

parser.add_argument('--mort_loading_range', default=[0.9, 1.1], type=float, nargs=2,
                    help='Mortality loading uniform distribution range')

parser.add_argument('--log_to_file', type=str, default='', help="Log to a file. If empty, logs to stdout instead.")
parser.add_argument('--prop_immune', type=float, default=0)


def main():
    """Main script executing all required functionality. Use the command line option `-h` to see all options. This
    script checks the given command arguments for errors, and then sets up a sampling model and solves it according
    to the given arguemnts. If valid, the arguments are saved to a config json in the '--output_dir' directiory. Once
    the model is saved, it saves the parameters of the model according to the '--output_dir' and '--model_name' flags.
    It will then plot and save the plots of the model.
    """
    args = parser.parse_args()  # parse command line arguments

    if args.from_config:
        # load arguments from config, but allow them to be overwritten by the command prompt
        json_dir = Path(args.from_config)
        if not json_dir.is_file():
            raise ValueError(f"Given configuration file '{args.from_config}' is either not a file or does not exist.")
        with open(json_dir, 'rt') as f:
            json_args = argparse.Namespace()
            json_args.__dict__.update(json.load(f))  # load variables from json file
            args = parser.parse_args(namespace=json_args)  # overwrite the loaded variable with command arguments

    if args.load_prior_file:
        load_prior_file = Path(args.load_prior_file)
        # check if prior file is a file
        if not load_prior_file.is_file():
            raise ValueError(f"Given prior file '{args.load_prior_file}' is not a file or does not exist.")
    else:
        load_prior_file = None

    # check if output directory is valid
    output_dir = Path(args.output_dir)
    if not output_dir.is_dir():
        raise ValueError(f'Given directory "{args.output_dir}" is either not a directory or does not exist.')

    # check if files exist in output directory, and if those files are relted to our model
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
                        level=logging.INFO,
                        filename=None if args.log_to_file == '' else args.log_to_file)

    # log training runs and samples
    logging.warning(f"Training model for {args.nb_runs} run(s) with {args.nb_samples} samples "
                    f"and {args.ratio_resample * 100:.1f}% resamples.")

    # log variables we are fitting to
    if args.fit_detected or args.fit_hospitalised or args.fit_icu or args.fit_deaths:
        logging.warning(f"Fitting to {'detected, ' if args.fit_detected else ''}"
                        f"{'hospitalised, ' if args.fit_hospitalised else ''}"
                        f"{'ICU, ' if args.fit_icu else ''}"
                        f"{'and ' if args.fit_detected or args.fit_hospitalised or args.fit_icu else ''}"
                        f"{'death' if args.fit_deaths else ''} cases")
    else:
        logging.warning(f'Not fitting to any data! Use --fit_detected, --fit_icu, --fit_hospitalised, or --fit_deaths')

    # load data
    if args.fit_data.lower() == 'wc':
        t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs = load_data_WC()
    elif args.fit_data.lower() == 'national':
        t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs = load_data_national()
    else:
        raise ValueError("The --fitting_data flag is not specified correctly. "
                         f"Should be 'WC' or 'national', got '{args.fit_data}' instead.")

    # save model args to config file
    if not args.only_process_runs and not args.only_plot:
        with open(output_dir.joinpath(f"{args.model_name}_config.json"), 'wt') as f:
            # save the json, but don't include the overwrite or from_json commands
            cmds = vars(args).copy()
            cmds.pop('overwrite', None)
            cmds.pop('from_json', None)
            cmds.pop('only_process_runs', None)
            cmds.pop('only_plot')
            json.dump(cmds, f, indent=4)

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
                                                                    args=args,
                                                                    load_prior_file=load_prior_file,
                                                                    model_base=save_dir)

    # set seaborn plotting style
    sns.set(style='darkgrid')

    model_base = output_dir.joinpath(f'{args.model_name}')
    if args.nb_runs > 1:
        if not args.only_process_runs:
            for run in range(args.nb_runs):
                model_run_base = output_dir.joinpath(f'{run:02}_{args.model_name}')
                logging.info(f'Executing run {run + 1}')
                if not args.only_plot:
                    _build_and_solve_model(model_run_base)
                calculate_resample(t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs, args=args, model_base=model_run_base)
        # process runs to single output
        logging.info(f'Processing results from {args.nb_runs} runs')
        nb_process_resamples = int(args.ratio_resample * args.nb_runs * args.nb_samples)
        process_multi_run(args.nb_runs, nb_process_resamples, output_dir, args.model_name, args)
        calculate_resample(t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs, args=args, model_base=model_base)
    else:
        if not args.only_plot:
            _build_and_solve_model(model_base)
        calculate_resample(t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs, args=args, model_base=model_base)


def process_multi_run(nb_runs, nb_resamples, output_dir, model_name, args):
    full_samples = None
    for i in range(nb_runs):
        model_run_base = output_dir.joinpath(f'{i:02}_{model_name}')
        fp = f'{model_run_base}_sample.pkl'
        logging.info(f'Loading samples from {fp}')
        with open(fp, 'rb') as f:
            samples = pickle.load(f)
            assert isinstance(samples, dict)
        if full_samples is None:
            full_samples = samples
        else:
            for key, value in full_samples.items():
                full_samples[key] = np.concatenate([value, samples[key]], axis=0)

    log_weights = full_samples['log_weights']
    weights = softmax(log_weights)
    nb_samples = len(weights)
    logging.info(f'Processed {nb_samples} total samples from {nb_runs} runs')

    resample_indices = np.random.choice(nb_samples, nb_resamples, p=weights)

    resample_vars = {}
    for key, value in full_samples.items():
        resample_vars[key] = value[resample_indices]
    resample_vars.pop('log_weights')

    model_base = output_dir.joinpath(f'{model_name}')

    with open(f'{model_base}_resample.pkl', 'wb') as f:
        pickle.dump(resample_vars, f)
    with open(f'{model_base}_sample.pkl', 'wb') as f:
        pickle.dump(full_samples, f)

    with open(f'{model_run_base}_scalar.pkl', 'rb') as f:
        scalar_vars = pickle.load(f)
    with open(f'{model_run_base}_group.pkl', 'rb') as f:
        group_vars = pickle.load(f)

    with open(f'{model_base}_scalar.pkl', 'wb') as f:
        pickle.dump(scalar_vars, f)
    with open(f'{model_base}_group.pkl', 'wb') as f:
        pickle.dump(group_vars, f)

    nb_groups = 1
    for key, value in group_vars.items():
        nb_groups = np.max([nb_groups, value.shape[-1]])
    for key, value in resample_vars.items():
        nb_groups = np.max([nb_groups, value.shape[-1]])

    scalar_vars.pop('t0')
    e0 = resample_vars.pop('e0', None)

    y0, e0 = create_y0(args, nb_resamples, nb_groups, e0=e0)

    # TODO: Create a static method that returns the deterministic variables
    model = SamplingNInfectiousModel(
        nb_groups=nb_groups,
        **scalar_vars,
        **group_vars,
        **resample_vars,
        y0=y0
    )

    resample_vars['e0'] = e0

    plot_prior_posterior(model_base, full_samples, resample_vars,
                         model.calculated_sample_vars, model.calculated_resample_vars)


def build_and_solve_model(t_obs,
                          i_d_obs=None,
                          i_h_obs=None,
                          i_icu_obs=None,
                          d_icu_obs=None,
                          args=None,
                          load_prior_file: Path = None,
                          model_base: Path = Path('data/model')):
    """Build and solve a sampling model, fitting to the given truth variables at the truth time.

    :value t_obs: Time at which observations are made.
    :value i_d_obs: Detected truth cases.
    :value i_h_obs: Hospitalised truth cases.
    :value i_icu_obs: ICU truth cases.
    :value d_icu_obs: Deceased truth cases.
    :value args: Command line arguments.
    :value total_pop: Total population to consider.
    :value load_prior_file: Loads proportions from a prior csv file. This should be generated from a previous fit.
    :value model_base: The model base directory. Defaults to 'data/model', where 'data/' is the output_dir and 'model'
    is the model name. Saves plots to '{model_base}_priors_posterior.png', and '{model_base}_joint_posterior.png'.
    We also save the models variables to '{model_base}_*.pkl'. See the model documentation for more information on the
    model variables.
    """
    if args.age_groups:
        logging.warning('Splitting the population into 10 year age bands when fitting')
        nb_groups = 9  # 0-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70-79, 80+ making 9 age groups
    else:
        logging.warning('Treating population age groups homogenously')
        nb_groups = 1
    nb_samples = args.nb_samples
    ratio_resample = args.ratio_resample
    rel_postlockdown_beta = args.rel_postlockdown_beta
    contact_heterogeneous = args.contact_heterogeneous
    contact_k = args.contact_k

    # inform survival times from KM lifetime analysis of WC data
    time_h_to_c = 2.6
    time_h_to_r = 8
    time_h_to_d = 8
    time_c_to_r = 16
    time_c_to_d = 13

    if not load_prior_file:
        logging.info('Setting priors_params')
        time_infectious = _uniform_from_range(args.time_infectious_range, size=(nb_samples, 1))
        prop_a = _uniform_from_range(args.prop_as_range, size=(nb_samples, 1))
        prop_s_to_h = _uniform_from_range(args.prop_s_to_h_range, size=(nb_samples, 1))

        r0 = _uniform_from_range(args.r0_range, size=(nb_samples, 1))
        beta = r0 / time_infectious
        rel_lockdown5_beta = _uniform_from_range(args.rel_lockdown5_beta_range, size=(nb_samples, 1))
        rel_lockdown4_beta = np.random.uniform(rel_lockdown5_beta - 0.05, (rel_lockdown5_beta+0.2).clip(max=1), size=(nb_samples, 1))
        rel_lockdown3_beta = np.random.uniform(rel_lockdown4_beta - 0.05, (rel_lockdown4_beta+0.2).clip(max=0.9), size=(nb_samples, 1))
        rel_lockdown2_beta = np.random.uniform(rel_lockdown3_beta - 0.05, (rel_lockdown3_beta+0.2).clip(max=0.8), size=(nb_samples, 1))
        rel_postlockdown_beta = np.random.uniform(rel_lockdown2_beta - 0.01, (rel_lockdown2_beta+0.1), size=(nb_samples, 1))
        rel_beta_as = np.random.uniform(0.3, 1, size=(nb_samples, 1))

        e0 = _uniform_from_range(args.e0_range, size=(nb_samples, 1))

        hospital_loading = _uniform_from_range(args.hospital_loading_range, size=(nb_samples, 1))

        if not args.age_groups:
            # inform variables from the WC experience, not controlling for age
            prop_s_base = 0.043 * hospital_loading
            prop_m = (1 - prop_a) * (1 - prop_s_base)
            mort_loading = _uniform_from_range(args.mort_loading_range, size=(nb_samples, 1))
            prop_h_to_c = 6/1238
            prop_h_to_d = mort_loading * 103 / 825
            prop_c_to_d = mort_loading * 54 / 119
        else:
            logging.info('Using 9 age groups, corresponding to 10 year age bands.')
            # from ferguson
            # prop_s_base = np.array([[0.006, 0.003, 0.02, 0.038, 0.06, 0.092, 0.111, 0.148, 0.196]])
            prop_s_base = np.array([[0.005, 0.0025, 0.0167, 0.0317, 0.0501, 0.0768, 0.0927, 0.1236, 0.1637]])
            prop_m = (1 - prop_a) * (1 - prop_s_base * hospital_loading)
            # inform variables from the WC experience, controlling for age
            # these are calculated from WC data, where the proportions are found from patients with known outcomes
            # TODO: Change beta distributions to dirichlet distributions
            mort_loading = _uniform_from_range(args.mort_loading_range, size=(nb_samples, 1))
            prop_h_to_d = mort_loading * np.array([[0.011, 0.042, 0.045, 0.063, 0.096, 0.245, 0.408, 0.448, 0.526]])
            prop_c_to_d = mort_loading * np.array([[0.011, 0.042, 0.410, 0.540, 0.590, 0.650, 0.660, 0.670, .710]])
            prop_h_to_c = 6/1238  # np.array([[1 / 81, 1 / 81, 1 / 81, 7 / 184, 32 / 200, 38 / 193, 24 / 129, 10 / 88, 5 / 31]])
    else:
        # load df
        logging.info(f"Loading proportion priors_params from {load_prior_file}")

        if load_prior_file.suffix == '.csv':
            logging.info('Loading csv file')
            df_priors = pd.read_csv(load_prior_file)
            nb_prior_groups = int(df_priors['group'].max() + 1)
            nb_prior_samples = int(len(df_priors) / nb_prior_groups)

            # get mean variables
            load_and_randomise = lambda x: np.random.normal(
                df_priors[x].to_numpy().reshape(nb_prior_samples, nb_prior_groups).repeat(nb_repeats, axis=0),
                df_priors[x].to_numpy().reshape(nb_prior_samples, nb_prior_groups).std(axis=0)/10
            )

        elif load_prior_file.suffix == '.pkl':
            logging.info('Loading pkl file')
            df_priors = pickle.load(load_prior_file.open('rb'))
            nb_prior_groups = 1
            nb_prior_samples = None
            for key, value in df_priors.items():
                nb_prior_groups = np.max([nb_prior_groups, value.shape[-1]])
                if nb_prior_samples is None:
                    nb_prior_samples = value.shape[0]
                else:
                    assert nb_prior_samples == value.shape[0]

            # get mean variables
            load_and_randomise = lambda x: np.random.normal(df_priors[x].repeat(nb_repeats, axis=0),
                                                            scale=df_priors[x].std(axis=0)/10)

        nb_repeats = int(nb_samples / nb_prior_samples)

        # fix number of samples accordingly
        nb_samples = nb_repeats * nb_prior_samples

        # set random vars
        print(df_priors['time_infectious'].std(axis=0))
        time_infectious = load_and_randomise('time_infectious').clip(min=0)
        # TODO: Change these if else statements, rather use a default value in the function instead?
        if 'prop_a' in df_priors:
            prop_a = load_and_randomise('prop_a').clip(min=0, max=1)
        else:
            prop_a = _uniform_from_range(args.prop_as_range, size=(nb_samples, 1))
        if args.age_groups:
            prop_m = (1 - prop_a) * np.array([[0.999, 0.997, 0.988, 0.968, 0.951, 0.898, 0.834, 0.757, 0.727]])
        else:
            prop_m = (1 - prop_a) * 0.957  # ferguson
        if 'prop_s_to_h' in df_priors:
            prop_s_to_h = load_and_randomise('prop_s_to_h').clip(min=0, max=1)
        else:
            prop_s_to_h = _uniform_from_range(args.prop_s_to_h, size=(nb_samples, 1))
        beta = load_and_randomise('beta').clip(min=0)
        rel_lockdown5_beta = load_and_randomise('rel_lockdown5_beta').clip(min=0, max=1)
        rel_lockdown4_beta = load_and_randomise('rel_lockdown4_beta').clip(min=rel_lockdown5_beta-0.05, max=1)
        rel_lockdown3_beta = load_and_randomise('rel_lockdown3_beta').clip(min=rel_lockdown4_beta-0.05, max=1)
        rel_lockdown2_beta = load_and_randomise('rel_lockdown2_beta').clip(min=rel_lockdown3_beta-0.05, max=1)
        rel_beta_as = load_and_randomise('rel_beta_as').clip(min=0, max=1)
        if 'prop_h_to_c' in df_priors:
            prop_h_to_c = load_and_randomise('prop_h_to_c').clip(min=0, max=1)
        else:
            prop_h_to_c = 119 / 825 if args.age_groups else 6/119
        prop_h_to_d = load_and_randomise('prop_h_to_d').clip(min=0, max=1)
        prop_c_to_d = load_and_randomise('prop_c_to_d').clip(min=0, max=1)
        e0 = load_and_randomise('e0').clip(min=1e-20)

        mort_loading = prop_h_to_d / np.array([[0.011, 0.042, 0.045, 0.063, 0.096, 0.245, 0.408, 0.448, 0.526]])
        mort_loading = mort_loading[:, 0:1]

    y0, e0 = create_y0(args, nb_samples, nb_groups, e0=e0)
    t0 = args.t0

    model = SamplingNInfectiousModel(
        nb_groups=9 if args.age_groups else 1,
        beta=beta,
        rel_lockdown5_beta=rel_lockdown5_beta,
        rel_lockdown4_beta=rel_lockdown4_beta,
        rel_lockdown3_beta=rel_lockdown3_beta,
        rel_lockdown2_beta=rel_lockdown2_beta,
        rel_postlockdown_beta=rel_postlockdown_beta,
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
        time_h_to_c=time_h_to_c,
        time_h_to_r=time_h_to_r,
        time_h_to_d=time_h_to_d,
        time_c_to_r=time_c_to_r,
        time_c_to_d=time_c_to_d,
        contact_heterogeneous=contact_heterogeneous,
        contact_k=contact_k,
        y0=y0
    )

    # fit to data

    ratio_as_detected = 0
    ratio_m_detected = 0.3
    ratio_s_detected = 1

    model.calculate_sir_posterior(t0, t_obs, i_d_obs, i_h_obs, i_icu_obs, d_icu_obs,
                                  ratio_as_detected=ratio_as_detected,
                                  ratio_m_detected=ratio_m_detected,
                                  ratio_s_detected=ratio_s_detected,
                                  ratio_resample=ratio_resample,
                                  smoothing=1,
                                  group_total=True,
                                  likelihood=args.likelihood,
                                  fit_interval=args.fit_interval,
                                  fit_new_deaths=args.fit_new_deaths)

    # get dictionaries from model after solving
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

    # hack mort loading into plot
    if np.asarray(mort_loading).ndim > 0:
        calc_sample_vars['mort_loading'] = mort_loading
        calc_resample_vars['mort_loading'] = mort_loading[model.resample_indices]

    # hack hospital loading into plot
    if np.asarray(hospital_loading).ndim > 0:
        calc_sample_vars['hospital_loading'] = hospital_loading
        calc_resample_vars['hospital_loading'] = hospital_loading[model.resample_indices]

    # save model variables
    save_model_variables(model, base=model_base)

    # reshape to a dataframe for pair plotting
    df_resample = pd.DataFrame(index=range(model.nb_resamples))
    for key, value in resample_vars.items():
        for i in range(value.shape[-1]):
            df_resample[f'{key}_{i}'] = value[:, i]

    # plot variables of interest
    plot_prior_posterior(model_base, sample_vars, resample_vars, calc_sample_vars, calc_resample_vars)

    if len(df_resample.columns) <= 20:
        logging.info('Building joint distribution plot')
        g = sns.PairGrid(df_resample, corner=True)
        try:
            g = g.map_lower(sns.kdeplot, colors='C0')
            g = g.map_diag(sns.distplot)
        except np.linalg.LinAlgError:
            logging.warning(f'Plotting of joint distribution failed due to posterior collapse')
        g.savefig(f'{model_base}_joint_posterior.png')
        plt.clf()
        del g

    del model


def plot_prior_posterior(model_base, sample_vars, resample_vars, calc_sample_vars=None, calc_resample_vars=None):
    logging.info('Plotting prior and posterior distributions')

    n = 0
    for value in resample_vars.values():
        n += value.shape[-1]
    if calc_resample_vars is not None:
        for value in calc_resample_vars.values():
            n += value.shape[-1]
    n = int(np.ceil(np.sqrt(n)))

    fig, axes = plt.subplots(n, n, figsize=(n * 3, n * 3))
    axes = axes.flat

    ax_idx = 0
    for key, value in resample_vars.items():
        # TODO: plot variables for multiple groups
        for i in range(value.shape[-1]):
            logging.info(f'{key}_{i}: mean = {value[:, i].mean():.3f} - std = {value[:, i].std():.3f}')
            try:
                sns.distplot(value[:, i], ax=axes[ax_idx], color='C0')
            except np.linalg.LinAlgError:
                logging.warning(f'Plotting of posterior failed for {key}_{i} due to posterior collapse')
            sns.distplot(sample_vars[key][:, i], ax=axes[ax_idx], color='C1')
            axes[ax_idx].axvline(value[:, i].mean(), ls='--')
            axes[ax_idx].set_title(f'{key}_{i}')
            ax_idx += 1
    if calc_resample_vars is not None and calc_sample_vars is not None:
        logging.info('Adding calculated variables')
        for key, value in calc_resample_vars.items():
            for i in range(value.shape[-1]):
                logging.info(f'{key}_{i}: mean = {value[:, i].mean():.3f} - std = {value[:, i].std():.3f}')
                try:
                    sns.distplot(value[:, i], ax=axes[ax_idx], color='C0')
                except np.linalg.LinAlgError:
                    logging.warning(f'Plotting of posterior failed for {key}_{i} due to posterior collapse')
                sns.distplot(calc_sample_vars[key][:, i], ax=axes[ax_idx], color='C1')
                axes[ax_idx].axvline(value[:, i].mean(), ls='--')
                axes[ax_idx].set_title(f'{key}_{i}')
                ax_idx += 1

    plt.tight_layout()
    fig.savefig(f'{model_base}_priors_posterior.png')
    plt.clf()


def create_y0(args, nb_samples=1, nb_groups=1, e0=None):
    if e0 is None:
        e0 = np.random.uniform(0, 1e-5, size=(nb_samples, 1))
    y0 = np.zeros((nb_samples, nb_groups, SamplingNInfectiousModel.nb_states))
    if not args.age_groups:
        # single population group, so we set starting population accordingly
        logging.info('Treating population homogeneously.')
        df_pop = pd.read_csv('data/population.csv')
        if args.fit_data.lower() == 'wc':
            df_pop = df_pop['Western Cape']
        elif args.fit_data.lower() == 'national':
            df_pop = df_pop['Grand Total']
        total_pop = df_pop.sum()
        y0[:, :, 0] = (1 - e0) * total_pop * (1 - args.prop_immune)
        y0[:, :, 1] = e0 * total_pop * (1 - args.prop_immune)
    else:
        # multiple population groups as a result of age bands
        # have to proportion the starting populations respectively
        logging.info('Treating population heterogenously by age.')
        df_pop = pd.read_csv('data/population.csv')
        df_pop = df_pop.groupby('ageband').sum()
        over_80_rows = ['80-90', '90-100', '100+']
        df_pop.loc['80+'] = df_pop.loc[over_80_rows].sum()
        df_pop.drop(over_80_rows, inplace=True)
        map_agebands_to_idx = {
            '0-10': 0,
            '10-20': 1,
            '20-30': 2,
            '30-40': 3,
            '40-50': 4,
            '50-60': 5,
            '60-70': 6,
            '70-80': 7,
            '80+': 8
        }
        df_pop['idx'] = df_pop.index.map(map_agebands_to_idx).astype(int)
        if args.fit_data.lower() == 'wc':
            filter = 'Western Cape'
        elif args.fit_data.lower() == 'national':
            filter = 'Grand Total'
        if args.fit_data.lower() == 'wc':
            df_pop['Western Cape'] = df_pop['Western Cape'] * 7000000 / df_pop[
                'Western Cape'].sum()  # adjust to Andrew's 7m for now
        for i in range(nb_groups):
            y0[:, i, 0] = (1 - e0[:, 0]) * df_pop[filter][df_pop['idx'] == i].values[0] * (1 - args.prop_immune)
            y0[:, i, 1] = e0[:, 0] * df_pop[filter][df_pop['idx'] == i].values[0] * (1 - args.prop_immune)
    y0 = y0.reshape(-1)
    return y0, e0


def save_model_variables(model: SamplingNInfectiousModel, base='data/samples'):
    """Saves a sampling models varibles (stored as dictionary) for use later.

    :value model: A solved sampling model.
    :value base: The base directory at which to store the variables. Default is 'data/model', where 'data/' is the
    outputdirectory and 'model' is the model name.
    """
    nb_groups = model.nb_groups
    scalar_vars = model.scalar_vars
    group_vars = model.group_vars
    sample_vars = model.sample_vars
    resample_vars = model.resample_vars
    log_weights = model.log_weights

    nb_groups = model.nb_groups
    nb_samples = model.nb_samples
    nb_resamples = model.nb_resamples

    sample_vars['log_weights'] = log_weights

    # logging.info(f'Saving model variables to {base}_*.csv')
    # need to reshape sample vars
    # reshaped_sample_vars = {}
    # for key, value in sample_vars.items():
    #     reshaped_sample_vars[key] = value.reshape(-1)
    # reshaped_sample_vars['group'] = np.asarray([[i] * nb_samples for i in range(nb_groups)]).reshape(-1)
    # # need to reshape resample vars
    # reshaped_resample_vars = {}
    # for key, value in resample_vars.items():
    #     reshaped_resample_vars[key] = value.reshape(-1)
    # reshaped_resample_vars['group'] = np.asarray([[i] * nb_resamples for i in range(nb_groups)]).reshape(-1)

    # define dfs
    # for key, value in reshaped_resample_vars.items():
    #     print(key, value.shape)
    # for key, value in reshaped_sample_vars.items():
    #     print(key, value.shape)
    # df_sample = pd.DataFrame(reshaped_sample_vars)
    # df_sample['log_weights'] = log_weights.repeat(nb_groups, axis=-1)
    # print(df_sample)
    # df_resample = pd.DataFrame(reshaped_resample_vars)

    # save files
    # df_sample.to_csv(f"{base}_sample.csv", index=False)
    # df_resample.to_csv(f"{base}_resample.csv", index=False)
    with open(f'{base}_sample.pkl', 'wb') as f:
        pickle.dump(sample_vars, f)
    with open(f'{base}_resample.pkl', 'wb') as f:
        pickle.dump(resample_vars, f)
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
    max_date = np.min([df_deaths['date'].max(), df_confirmed['date'].max(), df_hosp_icu['date'].max()])
    max_date = max_date - datetime.timedelta(days=5)  # max date set as 5 days prior to shared maximum date
    min_date = max_date - datetime.timedelta(days=45) # min date set to 30 days prior the maximum date, to remove noise

    # filter maximum date
    df_deaths = df_deaths[df_deaths['date'] <= max_date]
    df_confirmed = df_confirmed[df_confirmed['date'] <= max_date]
    df_hosp_icu = df_hosp_icu[df_hosp_icu['date'] <= max_date]

    # filter minimum date
    df_deaths = df_deaths[df_deaths['date'] >= min_date]
    df_confirmed = df_confirmed[df_confirmed['date'] >= min_date]
    df_hosp_icu = df_hosp_icu[df_hosp_icu['date'] >= min_date]

    # sort by date
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
                       args=None,
                       model_base='data/model'):
    with open(f'{model_base}_scalar.pkl', 'rb') as f:
        scalar_vars = pickle.load(f)
    with open(f'{model_base}_group.pkl', 'rb') as f:
        group_vars = pickle.load(f)
    with open(f'{model_base}_resample.pkl', 'rb') as f:
        resample_vars = pickle.load(f)
    nb_groups = 1
    nb_samples = None
    for key, value in group_vars.items():
        nb_groups = np.max([nb_groups, value.shape[-1]])
    for key, value in resample_vars.items():
        nb_groups = np.max([nb_groups, value.shape[-1]])
        if nb_samples is None:
            nb_samples = value.shape[0]
        else:
            assert nb_samples == value.shape[0]

    logging.info(f"Samples: {nb_samples}")
    logging.info(f"Groups: {nb_groups}")

    t0 = scalar_vars.pop('t0')
    e0 = resample_vars.pop('e0', None)

    y0, e0 = create_y0(args, nb_samples, nb_groups, e0=e0)

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

    percentile_vars['e0'] = e0[[arg_25, arg_med, arg_975]]

    mort_loading = resample_vars['prop_h_to_d'] / np.array([[0.011, 0.042, 0.045, 0.063, 0.096, 0.245, 0.408, 0.448, 0.526]])
    prop_s_base = np.array([[0.005, 0.0025, 0.0167, 0.0317, 0.0501, 0.0768, 0.0927, 0.1236, 0.1637]])
    hospital_loading = (1 - resample_vars['prop_m'] / (1 - scalar_vars['prop_a'])) / prop_s_base
    percentile_vars['mort_loading'] = mort_loading[[arg_25, arg_med, arg_975]]
    percentile_vars['hospital_loading'] = hospital_loading[[arg_25, arg_med, arg_975]]

    # reshape this for converting to a df
    df_percentiles = pd.DataFrame(index=range(3))
    for key, value in percentile_vars.items():
        for i in range(value.shape[-1]):
            df_percentiles[f'{key}_{i}'] = value[:, i]
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
    ratio_as_detected = 1
    ratio_m_detected = 1
    ratio_s_detected = 1

    cum_detected_samples = ratio_as_detected * (i_a + r_a) + ratio_m_detected * (i_m + r_m) \
                           + ratio_s_detected * (i_s + i_h + i_c + h_c + h_r + h_d + c_r + c_d + r_h + r_c + d_h + d_c)

    cum_detected_samples = np.sum(cum_detected_samples, axis=2)
    h_tot = np.sum(h_r + h_d + h_c, axis=2)
    c_tot = np.sum(c_r + c_d, axis=2)
    d_tot = np.sum(d_c + d_h, axis=2)
    ifr = d_tot / np.sum(y[:, :, :, 2:], axis=(2, 3))
    hfr = d_tot / np.sum(r_h + r_c + d_h + d_c, axis=2)
    atr = np.sum(y[:, :, :, 2:], axis=(2, 3)) / (model.n.reshape(-1) / (1 - args.prop_immune))

    daily_deaths = np.diff(d_tot, axis=0, prepend=0)
    d_icu_obs_daily = np.diff(d_icu_obs)
    print(daily_deaths.shape)

    logging.info('Plotting solutions')

    pred_vars = [cum_detected_samples, h_tot, c_tot, d_tot, daily_deaths, ifr, hfr, atr]
    obs_vars = [i_d_obs, i_h_obs, i_icu_obs, d_icu_obs, d_icu_obs_daily, None, None, None]
    titles = ['Total Infected', 'Current Hospitalised', 'Current ICU', 'Cum Deaths', 'Daily Deaths',
              'Infection Fatality Ratio', 'Inpatient Fatality Ratio', 'Attack Rate']

    assert len(pred_vars) == len(obs_vars) and len(obs_vars) == len(titles)

    fig, axes = plt.subplots(2, len(pred_vars), figsize=(len(pred_vars) * 4, 8))

    logging.info('Generating timeseries summary stats and plotting')
    summary_stats = {}
    for j, row in enumerate(axes):
        for i in range(len(row)):
            mu = np.median(pred_vars[i], axis=1)
            low = np.percentile(pred_vars[i], 2.5, axis=1)
            high = np.percentile(pred_vars[i], 97.5, axis=1)

            axes[j, i].plot(tt_date, mu, c='C0')
            axes[j, i].fill_between(tt_date, low, high, alpha=.2, facecolor='C0')
            for tick in axes[j, i].get_xticklabels():
                tick.set_rotation(45)
            # axes[i].plot(tt, pred_vars[i][:, :, 0], c='k', alpha=0.1)
            if j == 0:
                if obs_vars[i] is not None:
                    if len(obs_vars[i]) == len(t_date):
                        axes[j, i].plot(t_date, obs_vars[i], 'x', c='C1')
                    else:
                        axes[j, i].plot(t_date[1:], obs_vars[i], 'x', c='C1')
                    axes[j, i].set_ylim((min(np.min(obs_vars[i]), np.min(mu[50])) * 0.5, max(np.max(obs_vars[i]), np.max(mu[120])) * 1.1))
                else:
                    axes[j, i].set_ylim((np.min(mu[50])*0.5, np.max(mu[120]) * 1.5))
                axes[j, i].set_xlim(
                    (pd.to_datetime('2020/03/27'), np.max(t_date) + datetime.timedelta(days=1))
                )
                axes[j, i].set_title(titles[i])
            if j == 1:
                axes[j, i].plot(tt_date, pred_vars[i][:, arg_25], c='C1', ls='--')
                axes[j, i].plot(tt_date, pred_vars[i][:, arg_med], c='C2', ls='--')
                axes[j, i].plot(tt_date, pred_vars[i][:, arg_975], c='C3', ls='--')
                axes[j, i].set_xlabel('Date')

            summary_stats[f'{titles[i]} Mean'] = mu.reshape(-1)
            summary_stats[f'{titles[i]} 2.5CI'] = low.reshape(-1)
            summary_stats[f'{titles[i]} 97.5CI'] = high.reshape(-1)

    plt.tight_layout()
    logging.info(f'Saving plot to {model_base}_prediction.png')
    plt.savefig(f'{model_base}_prediction.png')
    plt.clf()

    logging.info(f'Saving timeseries summary stats to {model_base}_prediction.csv')
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
