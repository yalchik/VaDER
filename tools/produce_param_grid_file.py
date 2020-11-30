import argparse
import itertools
import pandas as pd
from numpy import arange


# local test grid
PARAM_GRID = {
    "k": list(range(3, 7)),
    "n_hidden": [[64, 32], [32, 16], [16, 8], [32, 64], [8, 16]],
    "learning_rate": list(arange(0.0001, 0.001, 0.0004)),
    "batch_size": [16],
    "alpha": [1.0],
    "n_epoch": [10],
    "n_perm": [100],
    "n_splits": [2],
    "n_repeats": [1]
}

# paper simulation
hidden_layer_pows = [0, 1, 2, 3, 4, 5, 6]
hidden_layer_pows_of_2 = [2**p for p in hidden_layer_pows]
list_of_n_hidden_combinations = [[p] for p in hidden_layer_pows_of_2]
# (1), (2), (4), ..., (64), (1, 1), (1, 2), ..., (1, 64), ..., (64, 64)
for p in itertools.product(hidden_layer_pows_of_2, hidden_layer_pows_of_2):
    list_of_n_hidden_combinations.append(list(p))

PARAM_GRID = {
    "k": list(range(2, 16)),
    "n_hidden": list_of_n_hidden_combinations,
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64, 128],
    "alpha": [1.0],
    "n_epoch": [20],
    "n_perm": [1000],
    "n_splits": [2],
    "n_repeats": [20]
}

# pre-optimization local test grid
PARAM_GRID = {
    "k": [1],
    "n_hidden": [[64, 32], [32, 16], [16, 8], [32, 64], [8, 16]],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64, 128],
    "alpha": [0],
    "n_epoch": [10],
    "n_perm": [10],
    "n_splits": [2],
    "n_repeats": [3],
    "pre_optimize": [1]
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output_param_grid_file", type=str, default="paper_param_grid.json",
                        help="hyperparameters values for grid search")
    args = parser.parse_args()

    all_params_combinations = list(itertools.product(*PARAM_GRID.values()))
    all_params_list = pd.DataFrame(all_params_combinations, columns=PARAM_GRID.keys()).to_dict('records')

    with open(args.output_param_grid_file, 'w') as f:
        f.write(str(all_params_list).replace("'", '"'))
