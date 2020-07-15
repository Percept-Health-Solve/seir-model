import numpy as np
from seir.parameters import SampleLockdownParams, SampleOdeParams, MetaParams, FittingParams, BaseParams
from seir.cli import LockdownCLI, OdeParamCLI, MetaCLI, FittingCLI


param_classes = [
    SampleLockdownParams,
    SampleOdeParams,
    MetaParams,
    FittingParams
]

cli_classes = [
    LockdownCLI,
    OdeParamCLI,
    MetaCLI,
    FittingCLI
]


def assert_equal(a: BaseParams, b: BaseParams):
    """Pseudo checks if two parameter objects are equal"""
    a_vars = vars(a).copy()
    b_vars = vars(b).copy()
    assert len(a_vars) == len(b_vars)
    for k in a_vars:
        assert k in b_vars
        assert a_vars[k] == b_vars[k]


def test_from_default():
    param_instances = [x.from_default() for x in param_classes]
    for instance in param_instances:
        assert instance is not None


def test_from_cli():
    for param_class, cli_class in zip(param_classes, cli_classes):
        cli_ins = cli_class()
        param_ins = param_class.from_cli(cli_ins)
        assert param_ins is not None


def test_lockdown_params():
    nb_samples_list = [1, 10, 100]
    for nb_samples in nb_samples_list:
        rel_beta_lockdown = [0.8, 0.9, np.random.uniform(0, 1, size=(1, nb_samples))]
        rel_beta_period = [10, 20, 30]
        lockdown_params = SampleLockdownParams(nb_samples, rel_beta_lockdown, rel_beta_period)
        assert lockdown_params is not None
        assert lockdown_params.nb_samples == nb_samples
        assert lockdown_params.rel_beta_lockdown == rel_beta_lockdown
        assert lockdown_params.rel_beta_period == rel_beta_period
        assert np.all(lockdown_params.cum_periods == np.array([10, 30, 60]))


def test_ode_params():
    nb_samples_list = [1, 10, 100]
    for nb_samples in nb_samples_list:
        kwargs = {
            'nb_samples': nb_samples,
            'r0': 2.5,
            'rel_beta_asymptomatic': np.random.uniform(0, 2.5, size=(1, nb_samples)),
            'prop_a': np.random.uniform(0, 1, size=(1, nb_samples)),
            'prop_s': np.random.uniform(0, 1, size=(1, nb_samples)),
            'prop_s_to_h': np.random.uniform(0, 1, size=(1, nb_samples)),
            'prop_h_to_c': np.random.uniform(0, 1, size=(1, nb_samples)),
            'prop_h_to_d': np.random.uniform(0, 1, size=(1, nb_samples)),
            'prop_c_to_d': np.random.uniform(0, 1, size=(1, nb_samples)),
            'time_incubate': np.random.uniform(1, 2, size=(1, nb_samples)),
            'time_infectious': np.random.uniform(1, 2, size=(1, nb_samples)),
            'time_s_to_h': np.random.uniform(1, 2, size=(1, nb_samples)),
            'time_s_to_c': np.random.uniform(1, 2, size=(1, nb_samples)),
            'time_h_to_c': np.random.uniform(1, 2, size=(1, nb_samples)),
            'time_h_to_d': np.random.uniform(1, 2, size=(1, nb_samples)),
            'time_h_to_r': np.random.uniform(1, 2, size=(1, nb_samples)),
            'time_c_to_d': np.random.uniform(1, 2, size=(1, nb_samples)),
            'time_c_to_r': np.random.uniform(1, 2, size=(1, nb_samples)),
            'contact_k': np.random.uniform(0, 2, size=(1, nb_samples)),
        }

        ode_params = SampleOdeParams(**kwargs)

        kwargs['beta'] = kwargs['r0'] / kwargs['time_infectious']
        kwargs['prop_m'] = 1 - kwargs['prop_a'] - kwargs['prop_s']
        kwargs['prop_s_to_c'] = 1 - kwargs['prop_s_to_h']
        kwargs['prop_h_to_r'] = 1 - kwargs['prop_h_to_c'] - kwargs['prop_h_to_d']
        kwargs['prop_c_to_r'] = 1 - kwargs['prop_c_to_d']
        kwargs['time_rsh_to_h'] = kwargs['time_s_to_h'] - kwargs['time_infectious']
        kwargs['time_rsc_to_c'] = kwargs['time_s_to_c'] - kwargs['time_infectious']

        assert ode_params is not None
        ode_vars = vars(ode_params)
        for k in ode_vars:
            assert k in kwargs
            assert np.all(kwargs[k] == ode_vars[k])


def test_meta_params():
    nb_samples_list = [1, 10, 100]
    for nb_samples in nb_samples_list:
        nb_groups = 1
        meta_params = MetaParams(nb_samples, nb_groups)
        assert meta_params is not None
        assert meta_params.nb_samples == nb_samples
        assert  meta_params.nb_groups == nb_groups


def test_fitting_params():
    nb_runs_list = [1, 2, 10]
    ratio_resample_list = [1e-6, 0.2, 1]
    for nb_runs, ratio_resample in zip(nb_runs_list, ratio_resample_list):
        fit_params = FittingParams(nb_runs, ratio_resample)
        assert fit_params is not None
        assert fit_params.nb_runs ==nb_runs
        assert fit_params.ratio_resample == ratio_resample
