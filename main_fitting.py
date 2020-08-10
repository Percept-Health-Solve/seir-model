import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import pickle
import datetime

from pathlib import Path
from dataclasses import dataclass, field, asdict, fields
from typing import List, Union, Iterable

from seir.argparser import DataClassArgumentParser
from seir.cli import MetaCLI, LockdownCLI, OdeParamCLI, FittingCLI, BaseDistributionCLI, BaseCLI
from seir.parameters import FittingParams
from seir.ode import CovidSeirODE
from seir.solvers import ScipyOdeIntSolver
from seir.data import DsfsiData, CovidData, TimestampData, extend_data_samples, append_data_time
from seir.fitting import BayesSIRFitter


@dataclass
class DataCLI(BaseCLI):

    data_source: str = field(
        default='dsfsi/total',
        metadata={
            "help": "Selects the data source for the fitting procedure. If a csv file, it will look for a deaths, "
                    "hospitalised, critical, infected, and recovered columns and fit the output of the model to those "
                    "(if selected for fitting). Can also point to dsfsi/<province> to load the DSFSI data for a "
                    "particular province Defaults to dsfsi/total."
        }
    )

    population_source: str = field(
        default=None,
        metadata={
            "help": "A csv file containing a column labeled 'ageband' and 'population' that yields the number of "
                    "people in each ten year age group, from 0-9, 10-19, ..., 80+. This should correspond to the data "
                    "selected for fitting. NOTE: This is not needed when DSFSI data are selected as the data source."
        }
    )

    lockdown_date: str = field(
        default=None,
        metadata={
            "help": "The day of the start of a lockdown period. The model internally computes time values (such as "
                    "the seeding time t0) relative to this date. Should be in %Y/%m/%d format. This is not needed when "
                    "DSFSI data is used, as it is set to 2020/03/27)."
        }
    )

    min_date: str = field(
        default='2020/04/05',
        metadata={
            "help": "Minimum date from which to fit. Can help reduce noise due to early reporting faults."
        }
    )

    max_date: str = field(
        default=None,
        metadata={
            "help": "Maximum date from which to fit. Can help reduce noise at the end of reported data, due to lag."
        }
    )

    lockdown_date_dt: datetime.datetime = field(init=False)
    min_date_dt: datetime.datetime = field(init=False)
    max_date_dt: datetime.datetime = field(init=False)
    dsfsi_province: str = field(default=None, init=False)
    mode: str = field(default=None, init=False)

    def __post_init__(self):
        if self.data_source.split('/')[0] == 'dsfsi':
            self.mode = 'dsfsi'
            self.lockdown_date = '2020/03/27'
            if len(self.data_source.split('/')) == 1:
                self.dsfsi_province = 'total'
            else:
                province = self.data_source.split('/')[1]
                self.dsfsi_province = province.upper() if not province.lower() == 'total' else province.lower()
        elif Path(self.data_source).is_file():
            self.mode = 'file'
            if not Path(self.population_source).is_file():
                raise ValueError('--population_source not correctly specified for given data source.')
        else:
            raise ValueError('--data_source flag does not point to a dsfsi dataset or a local file.')
        self.lockdown_date = '2020/03/27' if self.lockdown_date is None else self.lockdown_date
        self.lockdown_date_dt = pd.to_datetime(self.lockdown_date, format='%Y/%m/%d') if self.lockdown_date is not None else None
        self.min_date_dt = pd.to_datetime(self.min_date, format='%Y/%m/%d') if self.min_date is not None else None
        self.max_date_dt = pd.to_datetime(self.max_date, format='%Y/%m/%d') if self.max_date is not None else None


@dataclass
class InitialCLI(BaseDistributionCLI):
    _defaults_dict = {
        't0': -50,
        'prop_e0': [0, 1e-5],
        'prop_immune': [0],
    }

    t0: int = field(
        default=-50,
        metadata={
            "help": "Initial time at which to process y0"
        }
    )

    prop_e0: List[float] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Proportion of exposed individuals at t0. Used to seed the SEIR model."
        }
    )

    prop_immune: List[float] = field(
        default_factory=lambda: None,
        metadata={
            "help": "Proportion of initial population who are immune to the disease and will never contract it."
        }
    )


@dataclass
class InputOutputCLI:

    from_config: str = field(
        default=None,
        metadata={
            "help": "Load parameter values from a config file instead of the command line"
        }
    )

    output_dir: str = field(
        default=None,
        metadata={
            "help": "Location to place output files"
        }
    )

    overwrite: bool = field(
        default=False,
        metadata={
            "help": "Whether to overwrite the contents of the output directory"
        }
    )

    output_path: Path = field(init=False)
    run_path: Path = field(init=False)

    def __post_init__(self):
        if self.output_dir is None:
            self.output_dir = './results'
        self.output_path = Path(self.output_dir)
        if not self.output_path.is_dir():
            self.output_path.mkdir()
        if (
            self.output_path.is_dir()
            and any(self.output_path.iterdir())
            and not self.from_config
            and not self.overwrite
        ):
            raise ValueError('Detected files in output directory. Define a new output directory or use --overwrite to '
                             'overcome.')
        self.run_path = self.output_path.joinpath('runs/')
        if not self.run_path.is_dir():
            self.run_path.mkdir()


def save_all_cli_to_config(clis: Union[BaseCLI, Iterable[BaseCLI]], directory: Union[str, Path], exclude: list = None):
    if exclude is None:
        exclude = []
    if isinstance(clis, BaseCLI):
        clis = [clis]
    json_data = {}
    for cli in clis:
        x = asdict(cli)
        f = [f.name for f in fields(cli) if f.init and f.name not in exclude]
        xx = {k: v for k, v in x.items() if k in f}
        json_data.update(xx)

    if isinstance(directory, str):
        directory = Path(directory)
    if not directory.is_dir():
        directory.mkdir()

    json.dump(json_data, directory.joinpath('config.json').open('w'), indent=4)


def plot_priors_posterior(prior_dict: dict, posterior_dict: dict, params_to_plot: Iterable):
    for param in params_to_plot:
        assert param in prior_dict, \
            f"Parameter {param} not found in given prior dictionary"
        assert param in posterior_dict, \
            f"Parameter {param} not found in given posterior dictionary"
        if not param == 'rel_beta_lockdown':
            assert isinstance(prior_dict[param], np.ndarray), \
                f"Parameter in prior dict {param} is not a numpy array"
            assert prior_dict[param].ndim == posterior_dict[param].ndim and 2 >= prior_dict[param].ndim > 0, \
                f"Mismatch of dimensions for parameter {param}."

    nb_plots = 0
    for param in params_to_plot:
        if param == 'rel_beta_lockdown':
            for x in prior_dict[param]:
                if x.ndim > 0:
                    nb_plots += x.shape[0]
        elif prior_dict[param].ndim == 1:
            nb_plots += 1
        else:
            nb_plots += prior_dict[param].shape[0]

    # plot params on a square grid
    n = int(np.ceil(np.sqrt(nb_plots)))
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    axes = axes.flat

    i = 0
    for param in params_to_plot:
        if param == 'rel_beta_lockdown':
            for x in range(len(prior_dict[param])):
                if prior_dict[param][x].ndim == 2:
                    for nb_group in range(prior_dict[param][x].shape[0]):
                        sns.distplot(prior_dict[param][x][nb_group], color='C0', ax=axes[i])
                        sns.distplot(posterior_dict[param][x][nb_group], color='C1', ax=axes[i])
                        axes[i].set_title(f"{param}_{x}_{nb_group}")
                        i += 1
        elif prior_dict[param].ndim == 1:
            sns.distplot(prior_dict[param], color='C0', ax=axes[i])
            sns.distplot(posterior_dict[param], color='C1', ax=axes[i])
            axes[i].set_title(param)
            i += 1
        else:
            for nb_group in range(prior_dict[param].shape[0]):
                sns.distplot(prior_dict[param][nb_group], color='C0', ax=axes[i])
                sns.distplot(posterior_dict[param][nb_group], color='C1', ax=axes[i])
                axes[i].set_title(f"{param}_{nb_group}")
                i += 1
    return fig, axes


def append_samples(a: np.ndarray, b: np.ndarray):
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray)
        if a.ndim > 0 and b.ndim > 0:
            return np.concatenate([a, b], axis=-1)
    return a


def process_runs(run_path: Path, nb_runs: int) -> dict:
    all_priors = None
    for run in range(nb_runs):
        prior_dict = pickle.load(run_path.joinpath(f'run{run:02}_prior_dict.pkl').open('rb'))
        if all_priors is None:
            all_priors = prior_dict
        else:
            for k, v in all_priors.items():
                if isinstance(v, list):
                    for i in range(len(v)):
                        all_priors[k][i] = append_samples(all_priors[k][i], prior_dict[k][i])
                else:
                    all_priors[k] = append_samples(all_priors[k], prior_dict[k])

    return all_priors


def main():
    sns.set(style='darkgrid')
    argparser = DataClassArgumentParser([MetaCLI, LockdownCLI, OdeParamCLI, FittingCLI, InitialCLI, InputOutputCLI, DataCLI])
    meta_cli, lockdown_cli, ode_cli, fitting_cli, initial_cli, output_cli, data_cli = argparser.parse_args_into_dataclasses()
    if output_cli.from_config:
        meta_cli, lockdown_cli, ode_cli, fitting_cli, initial_cli, output_cli, data_cli = argparser.parse_json_file(output_cli.from_config)
    save_all_cli_to_config([meta_cli, lockdown_cli, ode_cli, fitting_cli, initial_cli, output_cli, data_cli],
                           directory=output_cli.output_path, exclude=['from_config'])

    if data_cli.mode == 'dsfsi':
        df_pop = pd.read_csv('data/sa_age_band_population.csv')
        population_band = df_pop[data_cli.dsfsi_province].values
        if not meta_cli.age_heterogeneity:
            population_band = np.sum(population_band)
        population_band = np.expand_dims(population_band, axis=1)

        data = DsfsiData(province=data_cli.dsfsi_province,
                         filter_kwargs={'min_date': data_cli.min_date_dt,
                                        'max_date': data_cli.max_date_dt})
    elif data_cli.mode == 'file':
        data_fp = Path(data_cli.data_source)
        pop_fp = Path(data_cli.population_source)
        if data_fp.suffix != '.csv':
            raise ValueError('Only csv files are supported as data sources')
        if pop_fp.suffix != '.csv':
            raise ValueError('Only csv files are supported as population sources')

        df_pop = pd.read_csv(pop_fp, index_col='ageband')
        population_band = df_pop['population'].values
        if not meta_cli.age_heterogeneity:
            population_band = np.sum(population_band)
        population_band = np.expand_dims(population_band, axis=1)
        data = CovidData.from_csv(data_fp, lockdown_date=data_cli.lockdown_date_dt,
                                  filter_kwargs={'min_date': data_cli.min_date_dt,
                                                 'max_date': data_cli.max_date_dt})
    else:
        raise NotImplementedError

    all_solutions = None
    for run in range(fitting_cli.nb_runs):
        ode_prior = CovidSeirODE.sample_from_cli(meta_cli, lockdown_cli, ode_cli)
        solver = ScipyOdeIntSolver(ode_prior)

        y0 = np.zeros((ode_prior.nb_states, ode_prior.nb_groups, ode_prior.nb_samples))
        e0 = initial_cli.sample_attr('prop_e0', nb_samples=ode_prior.nb_samples)
        y0[1] = e0 * population_band
        y0[0] = (1 - e0) * population_band
        prop_immune = initial_cli.sample_attr('prop_immune', nb_samples=ode_prior.nb_samples)
        y0 = y0 * (1 - prop_immune)

        t = np.arange(max(data.all_timestamps().min(), initial_cli.t0), data.all_timestamps().max()+1)
        if initial_cli.t0 < t.min():
            t = np.concatenate([[initial_cli.t0], t])

        solution, full_sol = solver.solve(y0, t, return_full=True, exclude_t0=True)
        if all_solutions is None:
            all_solutions = solution
        else:
            all_solutions = extend_data_samples(all_solutions, solution)

        fitter = BayesSIRFitter(solution, data, FittingParams.from_cli(fitting_cli))

        prior_dict = {
            **ode_prior.params,
            'full_solution': full_sol,
            'hospitalised': solution.hospitalised.data,
            'critical': solution.critical.data,
            'infected': solution.infected.data,
            'deaths': solution.deaths.data,
            'attack_rate': np.sum(solution.infected.data, axis=1) / np.sum(y0 / (1 - prop_immune), axis=(0, 1)),
            'e0': e0,
            'y0': y0,
            'prop_immune': prop_immune
        }
        posterior_dict = fitter.get_posterior_samples(**prior_dict)

        pickle.dump(prior_dict, output_cli.run_path.joinpath(f'run{run:02}_prior_dict.pkl').open('wb'))
        pickle.dump(posterior_dict, output_cli.run_path.joinpath(f'run{run:02}_posterior_dict.pkl').open('wb'))

        fig, axes = plot_priors_posterior(prior_dict, posterior_dict,
                                          ['rel_beta_lockdown', 'r0', 'time_infectious', 'beta',
                                           'mortality_loading', 'hospital_loading', 'e0'])
        plt.tight_layout()
        fig.savefig(output_cli.run_path.joinpath(f'run{run:02}_prior_posterior.png'))

    prior_dict = process_runs(output_cli.run_path, fitting_cli.nb_runs)
    fitter = BayesSIRFitter(all_solutions, data, FittingParams.from_cli(fitting_cli))
    posterior_dict = fitter.get_posterior_samples(**prior_dict)

    pickle.dump(prior_dict, output_cli.output_path.joinpath(f'prior_dict.pkl').open('wb'))
    pickle.dump(posterior_dict, output_cli.output_path.joinpath(f'posterior_dict.pkl').open('wb'))

    fig, axes = plot_priors_posterior(prior_dict, posterior_dict,
                                      ['rel_beta_lockdown', 'r0', 'time_infectious', 'beta',
                                       'mortality_loading', 'hospital_loading', 'e0'])
    plt.tight_layout()
    fig.savefig(output_cli.output_path.joinpath(f'prior_posterior.png'))

    posterior_solution = CovidData(nb_groups=posterior_dict['nb_groups'], nb_samples=posterior_dict['nb_samples'],
                                   deaths=TimestampData(t[1:], posterior_dict['deaths']),
                                   hospitalised=TimestampData(t[1:], posterior_dict['hospitalised']),
                                   critical=TimestampData(t[1:], posterior_dict['critical']),
                                   infected=TimestampData(t[1:], posterior_dict['infected']))

    ode_post = CovidSeirODE.from_dict(posterior_dict)
    solve_post = ScipyOdeIntSolver(ode_post)
    y0_post = posterior_dict['full_solution'][-1]
    t_post = np.arange(t[-1], 300)
    extended_sol = solve_post.solve(y0_post, t_post, exclude_t0=True)
    projections = append_data_time(posterior_solution, extended_sol)

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    axes = data.plot(axes=axes, plot_daily_deaths=True, group_total=True, plot_kwargs={'color': 'C1'},
                     plot_fmt='x', timestamp_shift=data.lockdown_date)
    axes = posterior_solution.plot(axes=axes, plot_daily_deaths=True, group_total=True,
                                   timestamp_shift=data.lockdown_date, plot_kwargs={'color': 'C0'})
    plt.tight_layout()
    fig.savefig(output_cli.output_path.joinpath('predictions_short_term.png'))

    fig, axes = plt.subplots(1, 6, figsize=(18, 4))
    projections.plot(axes=axes, plot_daily_deaths=True, group_total=True, timestamp_shift=data.lockdown_date,
                      plot_kwargs={'color': 'C0'})
    plt.tight_layout()
    fig.savefig(output_cli.output_path.joinpath('predictions_long_term.png'))

    if meta_cli.age_heterogeneity:
        df_projections_by_age = projections.to_dataframe(
            group_labels=['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+'],
            timestamp_shift=data.lockdown_date
        )
        df_projections_by_age.to_csv(output_cli.output_path.joinpath('projections_by_age.csv'))

    df_projections_total = projections.to_dataframe(
        group_total=True,
        timestamp_shift=data.lockdown_date
    )
    df_projections_total.to_csv(output_cli.output_path.joinpath('projections_total.csv'))


if __name__ == '__main__':
    main()
