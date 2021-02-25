from typing import Dict, List, Union
from vader.hp_opt.interface.abstract_bayesian_params_factory import AbstractBayesianParamsFactory


class ParamsFactory(AbstractBayesianParamsFactory):

    def get_k_list(self):
        k_list = [2, 3, 4, 5, 6]
        return k_list

    def get_param_limits_dict(self):
        params_limits = {
            "alpha": [0.0, 1.0],
            "learning_rate": [1e-4, 1e-2],
            "batch_size": [8, 128],
            "n_hidden_layers": [1, 2],
            "hidden_layer_size": [1, 128]
        }
        return params_limits
