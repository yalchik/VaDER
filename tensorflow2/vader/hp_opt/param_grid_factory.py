import random
import itertools
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from .common import ParamsDictType, ParamsGridType


class ParamGridFactory:

    def get_randomized_param_grid(self, n_sample: int) -> ParamsGridType:
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
        return ParamGridFactory.map_param_dict_to_param_grid(self.get_full_param_dict())

    def get_full_param_dict(self) -> ParamsDictType:
        """Parameter dictionary for the full optimization"""
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
        all_params_combinations = list(itertools.product(*param_dict.values()))
        param_grid = pd.DataFrame(all_params_combinations, columns=param_dict.keys()).to_dict('records')
        return param_grid

    @staticmethod
    def gen_list_of_combinations(powers: List[int]) -> List[List[int]]:
        powers_of_2 = [2 ** p for p in powers]
        list_of_n_hidden_combinations = [[p] for p in powers_of_2]
        # (1), (2), (4), ..., (64), (1, 1), (1, 2), ..., (1, 64), ..., (64, 64)
        for p in itertools.product(powers_of_2, powers_of_2):
            list_of_n_hidden_combinations.append(list(p))
        return list_of_n_hidden_combinations
