from vader.hp_opt.param_grid_factory import ParamGridFactory


class TestParamGridFactory:

    def test_gen_list_of_combinations(self):
        n_hidden = ParamGridFactory.gen_list_of_combinations([0, 1, 2])
        assert n_hidden == [[1], [2], [4], [1, 1], [1, 2], [1, 4], [2, 1], [2, 2], [2, 4], [4, 1], [4, 2], [4, 4]]

    def test_map_param_dict_to_param_grid(self):
        param_dict = {
            "k": [3, 4],
            "n_hidden": [[32, 8]],
            "learning_rate": [0.01],
            "batch_size": [16],
            "alpha": [1.0]
        }
        param_grid = ParamGridFactory.map_param_dict_to_param_grid(param_dict)
        assert len(param_grid) == 2
        assert param_grid == [
            {
                'alpha': 1.0,
                'batch_size': 16,
                'k': 3,
                'learning_rate': 0.01,
                'n_hidden': [32, 8]
            }, {
                'alpha': 1.0,
                'batch_size': 16,
                'k': 4,
                'learning_rate': 0.01,
                'n_hidden': [32, 8]
            }
        ]

        param_dict = {
            "k": [3, 4],
            "n_hidden": [[32, 8], [64, 16]],
            "learning_rate": [0.01],
            "batch_size": [16, 32],
            "alpha": [1.0]
        }
        param_grid = ParamGridFactory.map_param_dict_to_param_grid(param_dict)
        assert len(param_grid) == 8
