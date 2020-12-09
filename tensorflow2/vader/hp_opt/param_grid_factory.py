import itertools
import pandas as pd
from typing import List, Dict, Tuple, Union, Optional
from .constants import ParamsDictType, ParamsGridType


class ParamGridFactory:

    def get_nonvar_param_grid(self) -> ParamsGridType:
        return ParamGridFactory.map_param_dict_to_param_grid(self.get_nonvar_param_dict())

    def get_var_param_grid(self) -> ParamsGridType:
        return ParamGridFactory.map_param_dict_to_param_grid(self.get_var_param_dict())

    def generate_param_dict_for_k_optimization(self, best_hyperparameters):
        best_hyperparameters_dict = eval(best_hyperparameters.name)
        param_dict = self.get_var_param_dict()
        param_dict["n_hidden"] = [best_hyperparameters_dict["n_hidden"]]
        param_dict["learning_rate"] = [best_hyperparameters_dict["learning_rate"]]
        param_dict["batch_size"] = [best_hyperparameters_dict["batch_size"]]
        return param_dict

    def get_nonvar_param_dict(self) -> ParamsDictType:
        """Parameter dictionary for the 1st step of optimization (using non-variational autoencoders)"""
        param_dict = {
            "n_hidden": ParamGridFactory.gen_list_of_combinations([0, 1, 2, 3, 4, 5, 6]),
            "learning_rate": [0.0001, 0.001, 0.01, 0.1],
            "batch_size": [16, 32, 64, 128],
            "alpha": [0],  # 0 value turns on the non-variational AEs mode
            "n_epoch": [10],
            "n_splits": [10]
        }
        return param_dict

    def get_var_param_dict(self) -> ParamsDictType:
        """Parameter dictionary for the 2nd step of optimization (number of clusters 'k')"""
        param_dict = {
            "k": list(range(2, 16)),
            "n_hidden": [[64, 32]],
            "learning_rate": [1e-3],
            "batch_size": [16],
            "alpha": [1.0],
            "n_epoch": [50],
            "n_splits": [2],
            "n_perm": [1000]
        }
        return param_dict

    @staticmethod
    def gen_list_of_combinations(powers: List[int]) -> List[List[int]]:
        powers_of_2 = [2 ** p for p in powers]
        list_of_n_hidden_combinations = [[p] for p in powers_of_2]
        # (1), (2), (4), ..., (64), (1, 1), (1, 2), ..., (1, 64), ..., (64, 64)
        for p in itertools.product(powers_of_2, powers_of_2):
            list_of_n_hidden_combinations.append(list(p))
        return list_of_n_hidden_combinations

    @staticmethod
    def map_param_dict_to_param_grid(param_dict: ParamsDictType) -> ParamsGridType:
        all_params_combinations = list(itertools.product(*param_dict.values()))
        param_grid = pd.DataFrame(all_params_combinations, columns=param_dict.keys()).to_dict('records')
        return param_grid
