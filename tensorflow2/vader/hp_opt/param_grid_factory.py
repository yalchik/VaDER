import random
import itertools
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from vader.hp_opt.common import ParamsDictType, ParamsGridType


class ParamGridFactory:
    """Contains methods to create parameter dictionaries and parameter grids."""

    def get_randomized_param_grid(self, n_sample: int) -> ParamsGridType:
        """
        Selects <n_sample> random items (excluding 'k'-s) from the full parameter grid.
        If n_sample is greater than the full grid size, returns the full grid.

        Parameters
        ----------
        n_sample : int
            Defines how many sets of hyperparameters (excluding 'k'-s) we select from the full grid.

        Returns
        -------
        Parameter grid (list of dictionaries mapping hyperparameters to certain values)
        """
        param_dict = self.get_full_param_dict()
        k_list = param_dict["k"]
        non_k_params_values = [val for key, val in param_dict.items() if key != "k"]
        all_non_k_params_combinations = list(itertools.product(*non_k_params_values))
        if n_sample >= len(all_non_k_params_combinations):
            return self.get_full_param_grid()
        randomized_non_k_params_combinations = random.sample(all_non_k_params_combinations, n_sample)
        randomized_params_combinations = [
            (c[0], *c[1]) for c in itertools.product(k_list, randomized_non_k_params_combinations)
        ]
        randomized_param_grid = pd.DataFrame(
            randomized_params_combinations, columns=param_dict.keys()
        ).to_dict('records')
        return randomized_param_grid

    def get_full_param_grid(self) -> ParamsGridType:
        """
        Returns the whole parameter grid.

        Returns
        -------
        Parameter grid (list of dictionaries mapping hyperparameters to certain values)
        """
        return ParamGridFactory.map_param_dict_to_param_grid(self.get_full_param_dict())

    def get_full_param_dict(self) -> ParamsDictType:
        """
        Returns the whole parameter dictionary. This method is supposed to be overridden in sub-classes.
        The current implementation returns the parameter dictionary corresponding to the main paper:
            https://academic.oup.com/gigascience/article/8/11/giz134/5626377

        Returns
        -------
        Parameter dictionary (mapping hyperparameters to their ranges of values)
        """
        param_dict = {
            "k": list(range(2, 11)),
            "n_hidden": self.gen_list_of_combinations([0, 1, 2, 3, 4, 5, 6]),
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64, 128],
            "alpha": [1.0]
        }
        return param_dict

    @staticmethod
    def map_param_dict_to_param_grid(param_dict: ParamsDictType) -> ParamsGridType:
        """
        Converts a parameters dictionary to a parameters grid.
        The difference is that the parameters grid is a list of dictionaries mapping each of the hyperparameters to
          some certain values, while the parameters dictionary maps each of the hyperparameters to their possible
          ranges of values.
        Parameters dictionary is easier to change for a human, and a parameters grid is easier to handle in the
          hyperparameters optimization process.

        Parameters
        ----------
        param_dict : dictionary[hyperparameter, range of values]

        Returns
        -------
        Parameter grid (list of dictionaries mapping hyperparameters to certain values)
        """
        all_params_combinations = list(itertools.product(*param_dict.values()))
        param_grid = pd.DataFrame(all_params_combinations, columns=param_dict.keys()).to_dict('records')
        return param_grid

    @staticmethod
    def gen_list_of_combinations(powers: List[int]) -> List[List[int]]:
        """
        Auxiliary method to support extensive lists of combinations for n_hidden hyperparameter
          (including all numbers of layers) as powers of '2'.
        Example:
          Input: 0, 1, 2, ..., 6
          Output: (1), (2), (4), ..., (64), (1, 1), (1, 2), ..., (1, 64), ..., (64, 64)

        Parameters
        ----------
        powers : list
            list of integers representing powers of '2'
        Returns
        -------
        List where each elements represents a certain configuration of hidden layers.
        """
        powers_of_2 = [2 ** p for p in powers]
        list_of_n_hidden_combinations = [[p] for p in powers_of_2]

        for p in itertools.product(powers_of_2, powers_of_2):
            list_of_n_hidden_combinations.append(list(p))
        return list_of_n_hidden_combinations
