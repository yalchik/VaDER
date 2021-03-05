from vader.hp_opt.interface.abstract_grid_search_params_factory import AbstractGridSearchParamsFactory


class ParamsFactory(AbstractGridSearchParamsFactory):

    def get_full_param_dict(self):
        """
        Returns the whole parameter dictionary. This method is supposed to be overridden in sub-classes.
        The current implementation returns the parameter dictionary corresponding to the main paper:
            https://academic.oup.com/gigascience/article/8/11/giz134/5626377

        Returns
        -------
        Parameter dictionary (mapping hyperparameters to their ranges of values)
        """
        param_dict = {
            "k": list(range(2, 7)),
            "n_hidden": self.gen_list_of_combinations([0, 1, 2, 3, 4, 5, 6, 7]),
            "learning_rate": [0.0001, 0.001, 0.01],
            "batch_size": [16, 32, 64],
            "alpha": [1.0]
        }
        return param_dict
