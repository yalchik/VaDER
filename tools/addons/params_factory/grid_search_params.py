from vader.hp_opt.interface.abstract_grid_search_params_factory import AbstractGridSearchParamsFactory


class ParamsFactory(AbstractGridSearchParamsFactory):

    def get_full_param_dict(self):
        param_dict = {
            "k": list(range(2, 7)),
            "n_hidden": [[128, 8], [64, 8], [32, 8], [128, 16], [64, 16]],
            "learning_rate": [0.0001, 0.001, 0.01],
            "batch_size": [16, 32, 64],
            "alpha": [1.0]
        }
        return param_dict
