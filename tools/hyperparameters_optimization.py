import os
import sys
import argparse
import numpy as np
import multiprocessing as mp
from typing import Tuple
from vader.utils.data_utils import read_adni_norm_data, read_nacc_data, read_adni_raw_data
from vader.hp_opt.vader_hyperparameters_optimizer import VADERHyperparametersOptimizer
from vader.hp_opt.param_grid_factory import ParamGridFactory
from vader.hp_opt.common import ParamsDictType


class PlainParamGridFactory(ParamGridFactory):

    def get_full_param_dict(self) -> ParamsDictType:
        """
        EDIT THIS FUNCTION TO OVERRIDE THE PARAMETER GRID
        """
        param_dict = {
            "k": list(range(2, 7)),
            "n_hidden": [[128, 8], [32, 8], [128, 32], [64, 16]],
            "learning_rate": [0.001, 0.01],
            "batch_size": [16, 64],
            "alpha": [1.0]
        }
        return param_dict


def read_custom_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    EDIT THIS FUNCTION TO SUPPORT THE "custom" DATA TYPE.

    Reads a given csv file and produces 2 tensors (X, W), where each tensor has tis structure:
      1st dimension is samples,
      2nd dimension is time points,
      3rd dimension is feature vectors.
    X represents input data
    W contains values 0 or 1 for each point of X.
      "0" means the point should be ignored (e.g. because the data is missing)
      "1" means the point should be used for training

    Implementation examples: vader.utils.read_adni_data or vader.utils.read_nacc_data
    """
    raise NotImplementedError


if __name__ == "__main__":
    """
    The script runs VaDER hyperparameters optimization on given data and produces a pdf report
      showing the performance of different hyperparameters sets for different k-s.
    
    Example:
    python hyperparameters_optimization.py --input_data_file=../data/ADNI/Xnorm.csv --input_data_type=ADNI
                                           --n_repeats=5 --n_sample=5 --n_consensus=1 --n_epoch=10
                                           --n_splits=2 --n_perm=10 --output_folder=../vader_results 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, required=True, help=".csv file with input data")
    parser.add_argument("--input_data_type", type=str, choices=["ADNI", "NACC", "PPMI", "ADNI_RAW", "custom"], required=True)
    parser.add_argument("--input_weights_file", type=str, help=".csv file with flags for missing values")
    parser.add_argument("--input_seed", type=int, help="used both as KFold random_state and VaDER seed")
    parser.add_argument("--n_repeats", type=int, default=10, help="number of repeats, default 10")
    parser.add_argument("--n_proc", type=int, default=6, help="number of processor units that can be used, default 6")
    parser.add_argument("--n_sample", type=int, help="number of hyperparameters set per CV, default - full grid")
    parser.add_argument("--n_consensus", type=int, default=1, help="number of repeats for consensus clustering, default 1")
    parser.add_argument("--n_epoch", type=int, default=10, help="number of epochs for VaDER training, default 10")
    parser.add_argument("--n_splits", type=int, default=2, help="number of splits in KFold per optimization job, default 2")
    parser.add_argument("--n_perm", type=int, default=100, help="number of permutations for prediction strength, default 100")
    parser.add_argument("--output_folder", type=str, default=".", required=True, help="a directory where report will be written")
    args = parser.parse_args()

    if not os.path.exists(args.input_data_file):
        print("ERROR: input data file does not exist")
        sys.exit(1)

    if args.input_weights_file and not os.path.exists(args.input_weights_file):
        print("ERROR: weights data file does not exist")
        sys.exit(2)

    input_data, input_weights = None, None
    if args.input_data_type == "ADNI":
        input_data, weights = read_adni_norm_data(args.input_data_file)
        input_weights, _ = read_adni_norm_data(args.input_weights_file) if args.input_weights_file else weights, None
    elif args.input_data_type == "NACC":
        input_data, weights = read_nacc_data(args.input_data_file)
        input_weights, _ = read_nacc_data(args.input_weights_file) if args.input_weights_file else weights, None
    elif args.input_data_type == "PPMI":
        print("ERROR: Sorry, PPMI data processing has not been implemented yet.")
        exit(3)
    elif args.input_data_type == "ADNI_RAW":
        input_data, weights = read_adni_raw_data(args.input_data_file)
        input_weights, _ = read_adni_raw_data(args.input_weights_file) if args.input_weights_file else weights, None
    elif args.input_data_type == "custom":
        input_data, weights = read_custom_data(args.input_data_file)
        input_weights, _ = read_custom_data(args.input_weights_file) if args.input_weights_file else weights, None
    else:
        print("ERROR: Unknown data type.")
        exit(4)

    optimizer = VADERHyperparametersOptimizer(
        param_grid_factory=PlainParamGridFactory(),
        n_repeats=args.n_repeats,
        n_proc=args.n_proc if args.n_proc else mp.cpu_count(),
        n_sample=args.n_sample,
        n_consensus=args.n_consensus,
        n_epoch=args.n_epoch,
        n_splits=args.n_splits,
        n_perm=args.n_perm,
        seed=args.input_seed,
        output_folder=args.output_folder
    )

    optimizer.run(input_data, input_weights)
