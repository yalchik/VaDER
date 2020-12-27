import os
import sys
import argparse
import numpy as np
import multiprocessing as mp
from vader.utils import read_adni_data
from vader.hp_opt import VADERHyperparametersOptimizer, ParamGridFactory
from vader.hp_opt.setup import ParamsDictType


class MyParamGridFactory(ParamGridFactory):

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


def read_custom_data(filename: str) -> np.ndarray:
    """
    EDIT THIS FUNCTION TO SUPPORT THE "custom" DATA TYPE.

    Reads a given csv file and produces a tensor, where:
      1st dimension is samples,
      2nd dimension is time points,
      3rd dimension is feature vectors.
    @param filename: input data file in .csv format.
    @return: tensor.
    """
    raise NotImplementedError


if __name__ == "__main__":
    """
   
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, help="a .csv file with input data", required=True)
    parser.add_argument("--input_weights_file", type=str, help="a .csv file with flags for missing values")
    parser.add_argument("--input_seed", type=int, help="used both as random_state and VaDER seed")
    parser.add_argument("--input_data_type", choices=["ADNI", "PPMI", "custom"], help="data type", required=True)
    parser.add_argument("--n_repeats", type=int, default=1, help="number of repeats")
    parser.add_argument("--n_proc", type=int, help="number of processor units that can be used")
    parser.add_argument("--n_sample", type=int, help="number of hyperparameters set per CV")
    parser.add_argument("--n_consensus", type=int, default=1, help="number of repeats for consensus clustering")
    parser.add_argument("--n_epoch", type=int, default=10, help="number of epochs for VaDER training")
    parser.add_argument("--n_splits", type=int, default=2, help="number of splits in KFold per optimization job")
    parser.add_argument("--n_perm", type=int, default=10, help="number of permutations for prediction strength")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--log_dir", type=str, help="a directory where logs will be written")
    parser.add_argument("--output_folder", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_data_file):
        print("ERROR: input data file does not exist")
        sys.exit(1)

    if args.input_weights_file and not os.path.exists(args.input_weights_file):
        print("ERROR: weights data file does not exist")
        sys.exit(2)

    input_data, input_weights = None, None
    if args.input_data_type == "ADNI":
        input_data = read_adni_data(args.input_data_file)
        input_weights = read_adni_data(args.input_weights_file) if args.input_weights_file else None
    elif args.input_data_type == "PPMI":
        print("ERROR: Sorry, PPMI data processing has not been implemented yet.")
        exit(3)
    elif args.input_data_type == "custom":
        input_data = read_custom_data(args.input_data_file)
        input_weights = read_custom_data(args.input_weights_file) if args.input_weights_file else None
    else:
        print("ERROR: Unknown data type.")
        exit(4)

    optimizer = VADERHyperparametersOptimizer(
        param_grid_factory=MyParamGridFactory(),
        seed=args.input_seed,
        n_repeats=args.n_repeats,
        n_proc=args.n_proc if args.n_proc else mp.cpu_count(),
        n_sample=args.n_sample,
        n_consensus=args.n_consensus,
        n_epoch=args.n_epoch,
        n_splits=args.n_splits,
        n_perm=args.n_perm,
        output_cv_path=args.output_folder,
        verbose=args.verbose,
        log_dir=args.log_dir
    )

    optimizer.run(input_data, input_weights)
