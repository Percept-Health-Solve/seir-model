import numpy as np
import pandas as pd
import itertools

from .model import NInfectiousModel


class MultiPopWrapper(NInfectiousModel):

    def __init__(self,
                 pop_categories: dict,
                 inf_labels: list,
                 t_inc: float,
                 alpha,
                 q_se,
                 q_ii,
                 q_ir,
                 q_id,
                 rho_delta,
                 rho_beta,
                 infectious_func=None,
                 imported_func=None,
                 extend_vars:bool = False):
        """A wrapper around the NInfectiousModel that is designed to make working with multiple population groups
        more intuitive. Instead of defining the number of population groups explicitly, here you instead define the
        population/infection categories of interest in a dict. For example, you can define 3 age bands and 2 risk
        factor groups as pop_categories={'age': ['infant', 'adult', 'elderly'], 'risk': ['low', 'high']}. This class
        takes care of the resulting number of population groups (3*2=6) and will label the classes more intuitively
        when saving to a csv.

        Params
        ------

        pop_categories: dictionary of lists
            The categories of population groups to keep track of. The nb_groups variables is calculated as a product of
            the lengths of the lists in the dictionary.

        inf_labels: int
            The infectious labels to assign to the infectious states. The nb_infectious variables is just the length of
            this list.

        t_inc: float
            The incubation time of the disease, transitioning between the exposed and infected states.

        alpha: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The proportion of the population leaving the E state and entering each of the infected states.
            The rows of this array must sum to 1. Can only have shape (nb_infectious,) if extend_vars is True.

        q_se: [nb_infectious X 1] array
            The transition rates from the S state to the E state. This is interpreted as the number of secondary
            infections on the susceptible population caused by a member of the corresponding infectious state.

        q_ii: [nb_group X nb_infectious X nb_infectious] or [nb_infectious X nb_infectious] array
            The transition rates between infectious states. The columns of the each q_ii[j] matrix must sum
            to 0 for the population to be preserved. Can only have shape (nb_infectious, nb_infectious) if extend_vars
            is True.

        q_ir: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The transition rates from the I states to the R states for each population group. Can only have shape
            (nb_infectious) if extend_vars is True.

        q_id: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The transition from the I states to the R states for each population group. Can only have shape
            (nb_infectious,) if extend_vars is True.

        rho_delta: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The proportion of I states undergoing transitions from one I state to another I state for each population
            group. Conversely, 1 - rho represents the proportion of the population that are transitioning to the
            removed states R and D. Can only have shape (nb_infectious,) if extend_vars is True.

        rho_beta: [nb_group X nb_infectious] or [nb_infectious X 1] array
            The mortality rate of the I states undergoing a transition from I to D. Can only have shape (nb_infectious,)
            if extend_vars is True.

        infectious_func: callable, default=None
            A function that takes one input (time) and returns a multiplicative factor on the infectious rate. This can
            be used to model quarantines or periods of high infectivity.

        imported_func: callable, default=None
            A function that takes one input argument (time) and returns the rate of change in infections for that time.
            Used to seed the model with imported cases.

        extend_vars: bool, default=False
            Whether to assume that the given transition rates apply to all population groups. Useful when data on
            certain transition rates between population groups are unknown but data on average transitions rates are
            available.
        """
        # find nb_groups
        nb_groups = 1
        pop_cat_list = []
        unique_elems = set()
        for key in pop_categories:
            pop_categories[key] = np.asarray(pop_categories[key])
            # assert unique elements
            assert len(np.unique(pop_categories[key])) == len(pop_categories[key]), \
                f"Population categories in key '{key}' are not unique."
            assert bool(set(pop_categories[key].tolist()) & unique_elems) is False, \
                f"Population categories in key '{key}' match categories in other population groups."
            # append lists
            unique_elems.update(pop_categories[key].tolist())
            pop_cat_list.append(pop_categories[key].tolist())
            nb_groups *= len(pop_categories[key])
        cat_list_product = itertools.product(*pop_cat_list)

        self.nb_groups = nb_groups
        self.nb_infectious = len(inf_labels)

        self.pop_categories = pop_categories
        self.pop_labels = ['_'.join(x) for x in cat_list_product]
        self.inf_labels = inf_labels.copy()
        self.pop_label_to_idx = {self.pop_labels[i]: i for i in range(self.nb_groups)}
        self.idx_to_pop_label = {i: self.pop_labels[i] for i in range(self.nb_groups)}
        self.inf_label_to_idx = {self.inf_labels[i]: i for i in range(self.nb_infectious)}
        self.idx_to_inf_label = {i: self.inf_labels[i] for i in range(self.nb_infectious)}

        # parse dict/vectors correctly
        alpha = self._parse_dict_or_vector_input(alpha)
        rho_delta = self._parse_dict_or_vector_input(rho_delta)
        rho_beta = self._parse_dict_or_vector_input(rho_beta)

        # parse dict functions correctly
        imported_func = self._parse_dict_or_func(imported_func)

        super(MultiPopWrapper, self).__init__(
            self.nb_groups,
            self.nb_infectious,
            t_inc,
            alpha,
            q_se,
            q_ii,
            q_ir,
            q_id,
            rho_delta,
            rho_beta,
            infectious_func,
            imported_func,
            extend_vars
        )

    def _parse_dict_or_vector_input(self, input):
        assert isinstance(input, list) or isinstance(input, np.ndarray) or isinstance(input, dict), \
            f"Input {input} is not array-like or a dictionary."
        if isinstance(input, dict):
            # check dictionary values
            for key in input:
                assert isinstance(input[key], list) or isinstance(input[key], np.ndarray), \
                    f"Input key {key} contains a non-array-like structure {input[key]}."
                assert len(input[key]) == self.nb_infectious, \
                    f"Input key {key} with value {input[key]} is not of length {self.nb_infectious}."
            # create output
            temp = np.zeros((self.nb_groups, self.nb_infectious))
            for key in input:
                pop_idx_match = np.argwhere([key in x for x in self.pop_labels]).reshape(-1)
                if len(pop_idx_match) == 0:
                    raise ValueError(f'Given key {key} not found in population categories.')
                for pop_idx in pop_idx_match:
                    i = self.pop_label_to_idx[self.pop_labels[pop_idx]]
                    temp[i] = np.asarray(input[key])
            out = temp
            return out
        return np.asarray(input)

    def _parse_dict_or_func(self, input):
        if input is None:
            return lambda t: np.zeros((self.nb_groups, self.nb_infectious))
        assert callable(input), \
            "Input parameter is not callable."

        def wrapped_func(t):
            if isinstance(input(t), dict):
                out = np.zeros((self.nb_groups, self.nb_infectious))
                func_dict = input(t)
                for key in func_dict:
                    idx = self.pop_label_to_idx[key]
                    out[idx] = np.asarray(func_dict[key])
                return out
            else:
                return input(t)

        return wrapped_func



    def _to_csv(self, solution, t, fp):
        pop_label = lambda i: self.idx_to_pop_label[i]
        inf_label = lambda i: self.idx_to_inf_label[i]
        s_cols = [f'S_{pop_label(i)}' for i in range(self.nb_groups)]
        e_cols = [f'E_{pop_label(i)}' for i in range(self.nb_groups)]
        i_cols = [f'I_{pop_label(i)}_{inf_label(j)}' for i in range(self.nb_groups) for j in range(self.nb_infectious)]
        r_cols = [f'R_{pop_label(i)}_{inf_label(j)}' for i in range(self.nb_groups) for j in range(self.nb_infectious)]
        d_cols = [f'D_{pop_label(i)}_{inf_label(j)}' for i in range(self.nb_groups) for j in range(self.nb_infectious)]
        cols = s_cols + e_cols + i_cols + r_cols + d_cols
        df = pd.DataFrame(solution, columns=cols)
        df.insert(0, 'Day', t)
        df.to_csv(fp, index=False)

    def _parse_init_vectors(self, init_vectors: dict) -> tuple:
        s_0 = init_vectors.get('s_0')
        e_0 = init_vectors.get('e_0')
        i_0 = init_vectors.get('i_0')
        r_0 = init_vectors.get('r_0')
        d_0 = init_vectors.get('d_0')

        s_0 = self._parse_pop_vector(s_0)
        e_0 = self._parse_pop_vector(e_0)
        i_0 = self._parse_pop_vector(i_0, states='multi')
        r_0 = self._parse_pop_vector(r_0, states='multi')
        d_0 = self._parse_pop_vector(d_0, states='multi')

        return s_0, e_0, i_0, r_0, d_0

    def _parse_pop_vector(self, dict_vector, states='single'):
        # TODO: Fix coding structure here
        # TODO: Add exception for case when dictionary contains invalid keys
        assert states in ['single', 'multi']
        if isinstance(dict_vector, dict):
            default = 0 if states == 'single' else [0] * self.nb_infectious
            vector = [dict_vector.get(self.idx_to_pop_label[i], default) for i in range(self.nb_groups)]
        else:
            if states == 'single':
                vector = np.zeros(self.nb_groups) if dict_vector is None else dict_vector
            else:
                vector = np.zeros((self.nb_groups, self.nb_infectious)) if dict_vector is None else dict_vector
        vector = np.asarray(vector)
        return vector
