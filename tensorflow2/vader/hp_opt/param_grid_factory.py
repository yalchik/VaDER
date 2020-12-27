import random
import itertools
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from .setup import ParamsDictType, ParamsGridType


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
        randomized_param_grid = pd.DataFrame(randomized_params_combinations, columns=param_dict.keys()).to_dict('records')
        return randomized_param_grid

    def get_full_param_grid(self) -> ParamsGridType:
        return ParamGridFactory.map_param_dict_to_param_grid(self.get_full_param_dict())

    def get_full_param_dict(self) -> ParamsDictType:
        """Parameter dictionary for the full optimization"""
        param_dict = {
            "k": list(range(2, 16)),
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64, 128],
            "alpha": [1],
            "n_epoch": [20],
            "n_splits": [2],
            "n_perm": [1000]
        }
        return param_dict

    @staticmethod
    def map_param_dict_to_param_grid(param_dict: ParamsDictType) -> ParamsGridType:
        all_params_combinations = list(itertools.product(*param_dict.values()))
        param_grid = pd.DataFrame(all_params_combinations, columns=param_dict.keys()).to_dict('records')
        return param_grid
