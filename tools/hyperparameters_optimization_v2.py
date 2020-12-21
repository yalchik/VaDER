import os
import sys
import argparse
import multiprocessing as mp
from vader.hp_opt import VADERHyperparametersOptimizer, ParamGridFactory
from vader.hp_opt.constants import ParamsDictType


class MyParamGridFactory(ParamGridFactory):
    def get_nonvar_param_dict(self) -> ParamsDictType:
        """Parameter dictionary for the 1st step of optimization (using non-variational autoencoders)"""
        param_dict = {
            "n_hidden": ParamGridFactory.gen_list_of_combinations([5, 6]),  # should be 0-6 for paper results
            "learning_rate": [0.001, 0.01],  # should be 0.0001, 0.001, 0.01, 0.1 for paper results
            "batch_size": [16, 64],  # should be 16, 32, 64, 128 for paper results
            "alpha": [0],     # 0 value turns on the non-variational AEs mode
            "n_epoch": [10],  # should be 10 for paper results
            "n_splits": [2]   # should be 10 for paper results
        }
        return param_dict

    def get_var_param_dict(self) -> ParamsDictType:
        """Parameter dictionary for the 2nd step of optimization (number of clusters 'k')"""
        param_dict = {
            "k": list(range(2, 5)),   # should be range(2, 16) for paper results
            "n_hidden": [[64, 32]],   # just an example, will be re-assigned from step 1 results
            "learning_rate": [1e-3],  # just an example, will be re-assigned from step 1 results
            "batch_size": [16],       # just an example, will be re-assigned from step 1 results
            "alpha": [1.0],           # should be 1.0 for paper results
            "n_epoch": [10],          # should be 50 for paper results
            "n_splits": [2],          # should be 2 for paper results
            "n_perm": [3]             # should be 1000 for paper results
        }
        return param_dict

    def get_full_param_dict(self) -> ParamsDictType:
        """Parameter dictionary for the full optimization"""
        param_dict = {
            "k": list(range(2, 5)),
            "learning_rate": [0.001, 0.01],
            "batch_size": [16, 64],
            "alpha": [1],
            "n_epoch": [10],
            "n_splits": [2],
            "n_perm": [10]
        }
        return param_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_file", type=str, help="a .csv file with input data")
    parser.add_argument("--input_weights_file", type=str, help="a .csv file with flags for missing values")
    parser.add_argument("--input_param_grid_file", type=str, help="hyperparameters values for grid search")
    parser.add_argument("--input_seed", type=int, help="used both as random_state and VaDER seed")
    parser.add_argument("--n_repeats", type=int, default=1, help="number of repeats")
    parser.add_argument("--n_proc", type=int, help="number of processor units that can be used")
    parser.add_argument("--n_sample", type=int, help="number of hyperparameters set per CV")
    parser.add_argument("--output_save_path", type=str, help="a directory where all models will be saved")
    parser.add_argument("--stage", type=str, help="it allows to skip some parts of the process")
    parser.add_argument("--verbose", action='store_true')
    parser.add_argument("--log_dir", type=str, help="a directory where logs will be written")
    parser.add_argument("--full", action='store_true')
    parser.add_argument("output_evaluation_path", type=str, help="a .csv file where cross-validation results will be "
                                                                 "written")
    args = parser.parse_args()

    if not os.path.exists(args.input_data_file):
        print("ERROR: input data file does not exist")
        sys.exit(1)

    if args.input_weights_file and not os.path.exists(args.input_weights_file):
        print("ERROR: weights data file does not exist")
        sys.exit(2)

    if args.input_param_grid_file and not os.path.exists(args.input_param_grid_file):
        print("ERROR: param grid file does not exist")
        sys.exit(3)

    input_data_file = args.input_data_file
    input_weights_file = args.input_weights_file
    input_param_grid_file = args.input_param_grid_file
    input_seed = args.input_seed if args.input_seed else None
    n_repeats = args.n_repeats
    n_proc = args.n_proc if args.n_proc else mp.cpu_count()
    n_sample = args.n_sample
    output_save_path = args.output_save_path
    output_evaluation_path = args.output_evaluation_path
    stage = args.stage
    verbose = args.verbose
    log_dir = args.log_dir
    full_optimization = args.full

    optimizer = VADERHyperparametersOptimizer(
        param_grid_file=input_param_grid_file,
        param_grid_factory=MyParamGridFactory(),
        seed=input_seed,
        n_repeats=n_repeats,
        n_proc=n_proc,
        n_sample=n_sample,
        output_model_path=output_save_path,
        output_cv_path=output_evaluation_path,
        verbose=verbose,
        log_folder=log_dir,
        full_optimization=full_optimization
    )
    if stage:
        optimizer.run_certain_steps(input_data_file, input_weights_file, stage)
    else:
        optimizer.run_all_steps(input_data_file, input_weights_file)
