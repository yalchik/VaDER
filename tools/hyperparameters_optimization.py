import os
import sys
import argparse
import numpy as np
import multiprocessing as mp
import importlib.util
from vader.utils.data_utils import generate_wtensor_from_xtensor
from vader.hp_opt.vader_hyperparameters_optimizer import VADERHyperparametersOptimizer
from vader.hp_opt.vader_bayesian_optimizer import VADERBayesianOptimizer


if __name__ == "__main__":
    """
    The script runs VaDER hyperparameters optimization on given data and produces a pdf report
      showing the performance of different hyperparameters sets for different k-s.
    
    Example:
    python hyperparameters_optimization.py --input_data_file=../data/ADNI/Xnorm.csv
                                           --param_factory_script=addons/params_factory/grid_search_params.py
                                           --data_reader_script=addons/data_reader/adni_norm_data.py
                                           --n_repeats=5 --n_sample=5 --n_consensus=1 --n_epoch=10
                                           --n_splits=2 --n_perm=10 --output_folder=../vader_results 
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, required=True, help=".csv file with input data")
    parser.add_argument("--input_weights_file", type=str, help=".csv file with flags for missing values")
    parser.add_argument("--input_seed", type=int, help="used both as KFold random_state and VaDER seed")
    parser.add_argument("--param_factory_script", type=str, help="python script declaring param grid factory")
    parser.add_argument("--data_reader_script", type=str, help="python script declaring data reader class")
    parser.add_argument("--n_repeats", type=int, default=10, help="number of repeats, default 10")
    parser.add_argument("--n_proc", type=int, default=6, help="number of processor units that can be used, default 6")
    parser.add_argument("--n_sample", type=int, help="number of hyperparameters set per CV, default - full grid")
    parser.add_argument("--n_consensus", type=int, default=1, help="number of repeats for consensus clustering, default 1")
    parser.add_argument("--n_epoch", type=int, default=10, help="number of epochs for VaDER training, default 10")
    parser.add_argument("--early_stopping_ratio", type=float, help="early stopping ratio")
    parser.add_argument("--early_stopping_batch_size", type=int, default=5, help="early stopping batch size")
    parser.add_argument("--n_splits", type=int, default=2, help="number of splits in KFold per optimization job, default 2")
    parser.add_argument("--n_perm", type=int, default=100, help="number of permutations for prediction strength, default 100")
    parser.add_argument("--type", type=str, choices=["gridsearch", "bayesian"], default="gridsearch")
    parser.add_argument("--n_trials", type=int, default=100, help="number of trials (for bayesian optimization only), default 100")
    parser.add_argument("--output_folder", type=str, default=".", required=True, help="a directory where report will be written")
    parser.add_argument("--enable_cv_loss_reports", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.input_data_file):
        print("ERROR: input data file does not exist")
        sys.exit(1)

    if args.input_weights_file and not os.path.exists(args.input_weights_file):
        print("ERROR: weights data file does not exist")
        sys.exit(2)

    if args.param_factory_script and not os.path.exists(args.param_factory_script):
        print("ERROR: param grid factory file does not exist")
        sys.exit(3)

    if args.data_reader_script and not os.path.exists(args.data_reader_script):
        print("ERROR: data reader file does not exist")
        sys.exit(4)

    # dynamically import param factory
    param_factory_spec = importlib.util.spec_from_file_location("params_factory", args.param_factory_script)
    param_factory_module = importlib.util.module_from_spec(param_factory_spec)
    param_factory_spec.loader.exec_module(param_factory_module)

    # dynamically import data reader
    data_reader_spec = importlib.util.spec_from_file_location("data_reader", args.data_reader_script)
    data_reader_module = importlib.util.module_from_spec(data_reader_spec)
    data_reader_spec.loader.exec_module(data_reader_module)
    data_reader = data_reader_module.DataReader()

    x_tensor = data_reader.read_data(args.input_data_file)
    w_tensor = generate_wtensor_from_xtensor(x_tensor)
    input_data = np.nan_to_num(x_tensor)
    input_weights = w_tensor

    optimizer = None
    if args.type == "gridsearch":
        optimizer = VADERHyperparametersOptimizer(
            params_factory=param_factory_module.ParamsFactory(),
            n_repeats=args.n_repeats,
            n_proc=args.n_proc if args.n_proc else mp.cpu_count(),
            n_sample=args.n_sample,
            n_consensus=args.n_consensus,
            n_epoch=args.n_epoch,
            n_splits=args.n_splits,
            n_perm=args.n_perm,
            seed=args.input_seed,
            early_stopping_ratio=args.early_stopping_ratio,
            early_stopping_batch_size=args.early_stopping_batch_size,
            enable_cv_loss_reports=args.enable_cv_loss_reports,
            output_folder=args.output_folder
        )
    elif args.type == "bayesian":
        optimizer = VADERBayesianOptimizer(
            params_factory=param_factory_module.ParamsFactory(),
            n_repeats=args.n_repeats,
            n_proc=args.n_proc if args.n_proc else mp.cpu_count(),
            n_trials=args.n_trials,
            n_consensus=args.n_consensus,
            n_epoch=args.n_epoch,
            n_splits=args.n_splits,
            n_perm=args.n_perm,
            seed=args.input_seed,
            early_stopping_ratio=args.early_stopping_ratio,
            early_stopping_batch_size=args.early_stopping_batch_size,
            enable_cv_loss_reports=args.enable_cv_loss_reports,
            output_folder=args.output_folder
        )
    else:
        print("ERROR: Unknown optimization type.")
        exit(5)

    optimizer.run(input_data, input_weights)
