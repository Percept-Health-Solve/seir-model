import numpy as np
from seir.parameters import BaseParams
from seir.ode import CovidSeirODE


num_error = 1e-15


def expand_y_for_ode(y, nb_groups, nb_samples):
    y = np.expand_dims(y, axis=(1, 2))
    y = np.repeat(y, nb_groups, axis=1)
    y = np.repeat(y, nb_samples, axis=2)
    return y


def assert_param_equal(a: BaseParams, b: BaseParams):
    """Pseudo checks if two parameter objects are equal"""
    a_vars = vars(a).copy()
    b_vars = vars(b).copy()
    assert len(a_vars) == len(b_vars)
    for k in a_vars:
        assert k in b_vars
        if isinstance(a_vars[k], list):
            assert isinstance(b_vars[k], list)
            assert len(a_vars[k]) == len(b_vars[k])
            for i in range(len(a_vars[k])):
                assert np.all(a_vars[k][i] == b_vars[k][i])
        else:
            assert np.all(a_vars[k] == b_vars[k])


def assert_ode_equal(a: CovidSeirODE, b: CovidSeirODE):
    """Pseudo checks if two parameter objects are equal"""
    a_vars = vars(a).copy()
    b_vars = vars(b).copy()
    assert len(a_vars) == len(b_vars)
    for k in a_vars:
        assert k in b_vars
        assert_param_equal(a_vars[k], b_vars[k])


def test_from_default():
    covid_default = CovidSeirODE.from_default()
    assert covid_default is not None

    meta_params = covid_default.meta_params
    lockdown_params = covid_default.lockdown_params
    ode_params = covid_default.ode_params
    covid_ode = CovidSeirODE(meta_params, lockdown_params, ode_params)

    assert_ode_equal(covid_ode, covid_default)


def test_steady_states():
    covid_steady_states = [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # non infected, all susceptible
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # all dead
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0],  # all recovered
    ]

    ode = CovidSeirODE.from_default()

    for y in covid_steady_states:
        y = expand_y_for_ode(y, ode.nb_groups, ode.nb_samples)
        result = ode(y, 0)
        assert np.all(result == 0)


def test_exposed_seed():
    ode = CovidSeirODE.from_default()

    y = [0.9, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = expand_y_for_ode(y, ode.nb_groups, ode.nb_samples)

    def assert_result(result):
        assert np.all(result[0] == 0)  # susceptible should not change, dS=0
        assert np.all(result[1] < 0)  # exposed should move to infected states, dE < 0
        assert np.all(result[2] > 0) and np.all(result[3] > 0) and np.all(result[4] > 0)  # infected stated increase
        assert np.all(np.abs(np.sum(result, axis=0)) < num_error)  # ode is conserved (up to small numerical error)

    # contact homogeneous
    ode.ode_params.contact_k = 0
    result = ode(y, 0)
    assert_result(result)

    # contact heterogeneous
    ode.ode_params.contact_k = 0
    result = ode(y, 0)
    assert_result(result)


def test_infected_seed():
    ode = CovidSeirODE.from_default()

    y = [0.7, 0, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    y = expand_y_for_ode(y, ode.nb_groups, ode.nb_samples)

    def assert_result(result):
        assert np.all(result[0] < 0)  # susceptible should move to exposed, dS < 0
        assert np.all(result[1] > 0)  # exposed should accept susceptible, dE > 0
        assert np.all(result[0] == -result[1])  # dS = -dE when E=0
        assert np.all(result[2] < 0) and np.all(result[3] < 0) and np.all(result[4] < 0)  # infected states decrease
        assert np.all(result[5] >= 0) and np.all(result[6] >= 0)  # severe removed states increase
        assert np.all(np.abs(result[4] + result[5] + result[6]) < num_error)  # severe to removed states are conserved
        assert np.all(result[12] > 0) and np.all(result[12] > 0)  # asymptomatic and mild removed states increase
        assert np.all(result[2] + result[12] == 0)  # asymptomatic to removed states are conserved
        assert np.all(result[3] + result[13] == 0)  # mild to removed states are conserved
        assert np.all(np.abs(np.sum(result, axis=0)) < num_error)  # ode is conserved to numerical error

    # contact homogeneous
    ode.ode_params.contact_k = 0
    result = ode(y, 0)
    assert_result(result)

    # contact heterogeneous
    ode.ode_params.contact_k = 1
    result = ode(y, 0)
    assert_result(result)


def test_hospital_dynamics():
    ode = CovidSeirODE.from_default()

    y = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    y = expand_y_for_ode(y, ode.nb_groups, ode.nb_samples)

    result = ode(y, 0)
    assert np.all(result[7] < 0) and np.all(result[8] < 0) and np.all(result[9] < 0)
    assert np.all(result[10] > 0) and np.all(result[11] > 0)
    assert np.all(np.abs(result[8] + result[10] + result[11]) < num_error)
    assert np.all(result[14] > 0) and np.all(result[16] > 0)
    assert np.all(result[7] + result[14] == 0)
    assert np.all(result[9] + result[16] == 0)
    assert np.all(np.abs(np.sum(result, axis=0)) < num_error)


def test_icu_dynamics():
    ode = CovidSeirODE.from_default()

    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0]
    y = expand_y_for_ode(y, ode.nb_groups, ode.nb_samples)

    result = ode(y, 0)
    assert np.all(result[10] < 0) and np.all(result[11] < 0)
    assert np.all(result[15] > 0) and np.all(result[17] > 0)
    assert np.all(result[10] + result[15] == 0)
    assert np.all(result[11] + result[17] == 0)
    assert np.all(np.abs(np.sum(result, axis=0)) < num_error)
