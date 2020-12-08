import itertools
import os
import sys
import argparse
import json
import random
import pandas as pd
import multiprocessing as mp
import numpy as np
import uuid
from vader import VADER
from vader.utils import read_adni_data
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.special import comb
from typing import List, Dict, Tuple, Union, Optional
from numpy import ndarray


# Type aliases
ParamsDictType = Dict[str, Union[int, float, List[Union[int, float]]]]
ParamsGridType = List[ParamsDictType]


def get_nonvar_param_dict() -> ParamsDictType:
    """OPTIMIZATION STEP 1"""
    hidden_layer_pows = [0, 1, 2, 3, 4, 5, 6]
    hidden_layer_pows_of_2 = [2 ** p for p in hidden_layer_pows]
    list_of_n_hidden_combinations = [[p] for p in hidden_layer_pows_of_2]
    # (1), (2), (4), ..., (64), (1, 1), (1, 2), ..., (1, 64), ..., (64, 64)
    for p in itertools.product(hidden_layer_pows_of_2, hidden_layer_pows_of_2):
        list_of_n_hidden_combinations.append(list(p))

    param_dict = {
        "k": [1],
        "n_hidden": list_of_n_hidden_combinations,
        "learning_rate": [0.0001, 0.001, 0.01, 0.1],
        "batch_size": [16, 32, 64, 128],
        "alpha": [0],
        "n_epoch": [10],
        "n_perm": [1000],
        "n_splits": [10],
        "n_repeats": [20],
        "pre_optimize": [1]
    }
    return param_dict


def get_var_param_dict() -> ParamsDictType:
    """OPTIMIZATION STEP 2"""
    param_dict = {
        "k": list(range(2, 16)),
        "n_hidden": [],
        "learning_rate": [],
        "batch_size": [],
        "alpha": [1.0],
        "n_epoch": [50],
        "n_perm": [1000],
        "n_splits": [2],
        "n_repeats": [20],
        "pre_optimize": [0]
    }
    return param_dict


def run_hyperparameters_optimization(input_data_file: str, input_weights_file: str, input_param_grid_file: str,
                                     input_seed: int, n_repeats: int, n_proc: int, n_sample: int, output_save_path: str,
                                     output_evaluation_path: str, stage: str, verbose: bool = False) -> None:
    if verbose:
        print("=> start run_hyperparameters_optimization with", locals())

    # read files
    input_data = read_adni_data(input_data_file)
    input_weights = read_adni_data(input_weights_file) if input_weights_file else None
    step1_output_evaluation_path = output_evaluation_path + "_"

    if not stage or stage == "1":
        if input_param_grid_file:
            input_param_grid = read_param_grid(input_param_grid_file)
        else:
            input_param_grid = map_param_dict_to_param_grid(get_nonvar_param_dict())

        # randomize grid search
        if n_sample and n_sample < len(input_param_grid):
            param_grid = random.sample(input_param_grid, n_sample)
        else:
            param_grid = input_param_grid

        # cross-validation
        with mp.Pool(n_proc) as pool:
            cv_results_list = pool.map(run_cv_parallel, [(
                input_data, input_weights, input_seed, output_save_path, params_dict, verbose)
                for params_dict in param_grid
                for _ in range(n_repeats)
            ])

        # output
        cv_results_df = pd.DataFrame(cv_results_list)
        cv_results_df.to_csv(step1_output_evaluation_path)
        if verbose:
            print(f"=> run_hyperparameters_optimization - stage #1 finished. See: {step1_output_evaluation_path}")

    if stage == "2":
        cv_results_df = pd.read_csv(step1_output_evaluation_path, index_col=0)

    if not stage or stage == "2":
        # 2nd step (k-optimization) preparation
        best_hyperparameters = extract_best_parameters(cv_results_df)
        if verbose:
            print(f"BEST HYPERPARAMETERS ARE {best_hyperparameters.name}"
                  f"with loss={best_hyperparameters.test_reconstruction_loss}")
        best_hyperparameters_dict = generate_param_grid_for_k_optimization(best_hyperparameters)
        param_grid = map_param_dict_to_param_grid(best_hyperparameters_dict)

        # 2nd step (k-optimization) cross-validation
        with mp.Pool(n_proc) as pool:
            cv_results_list = pool.map(run_cv_parallel, [(
                input_data, input_weights, input_seed, output_save_path, params_dict, verbose)
                for params_dict in param_grid
                for _ in range(n_repeats)
            ])

        # 2nd step (k-optimization) output
        cv_results_df = pd.DataFrame(cv_results_list)
        cv_results_df.to_csv(output_evaluation_path)
        if verbose:
            print(f"=> run_hyperparameters_optimization - stage #2 finished. See: {output_evaluation_path}")

    if verbose:
        print("<= finish run_hyperparameters_optimization")


def extract_best_parameters(cv_results_df, minimize_metric: str = "test_reconstruction_loss"):
    cv_results_df_means = cv_results_df.groupby("params").mean()
    best_hyperparameters_index = cv_results_df_means[minimize_metric] == cv_results_df_means[minimize_metric].min()
    best_hyperparameters = cv_results_df_means[best_hyperparameters_index].iloc[0]
    return best_hyperparameters


def generate_param_grid_for_k_optimization(best_hyperparameters):
    best_hyperparameters_dict = eval(best_hyperparameters.name)
    param_dict = get_var_param_dict()
    param_dict["n_hidden"] = [best_hyperparameters_dict["n_hidden"]]
    param_dict["learning_rate"] = [best_hyperparameters_dict["learning_rate"]]
    param_dict["batch_size"] = [best_hyperparameters_dict["batch_size"]]
    return param_dict


def map_param_dict_to_param_grid(param_dict):
    all_params_combinations = list(itertools.product(*param_dict.values()))
    param_grid = pd.DataFrame(all_params_combinations, columns=param_dict.keys()).to_dict('records')
    return param_grid


def read_param_grid(param_grid_file: str) -> ParamsGridType:
    with open(param_grid_file, 'r') as json_file:
        input_param_grid = json.load(json_file)
    return input_param_grid


def run_cv_parallel(params: Tuple[np.ndarray, np.ndarray, int, str, ParamsDictType, bool]) -> pd.Series:
    return run_cv(*params)


def run_cv(data: ndarray, weights: ndarray, seed: int, save_path: str, params_dict: ParamsDictType,
           verbose: bool = False) -> pd.Series:
    cv_id = uuid.uuid4()
    if verbose:
        print(f"=> start run_cv id={cv_id} with seed={seed}, params_dict={params_dict}")

    n_splits = params_dict["n_splits"]
    pre_optimize = params_dict["pre_optimize"]

    cv_folds_results_list = []
    for train_index, val_index in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(data):
        X_train, X_val = data[train_index], data[val_index]
        W_train, W_val = (weights[train_index], weights[val_index]) if weights else None, None
        if pre_optimize:
            cv_fold_result = cv_fold_step_pre_optimize(X_train, X_val, W_train, W_val, seed, save_path, params_dict)
        else:
            cv_fold_result = cv_fold_step(X_train, X_val, W_train, W_val, seed, save_path, params_dict)
        cv_folds_results_list.append(cv_fold_result)

    cv_folds_results_df = pd.DataFrame(cv_folds_results_list)
    cv_mean_results_series = cv_folds_results_df.mean()
    cv_mean_results_series["params"] = str(params_dict)
    if verbose:
        print(f"<= finish run_cv id={cv_id} with seed={seed}, params_dict={params_dict}")
    return cv_mean_results_series


def cv_fold_step(X_train: ndarray, X_val: ndarray, W_train: Optional[ndarray], W_val: Optional[ndarray],
                 seed: int, save_path: str, params_dict: ParamsDictType) -> Dict[str, Union[int, float]]:
    # calculate y_pred
    vader = fit_vader(X_train, W_train, seed, save_path, params_dict)
    # noinspection PyTypeChecker
    test_loss_dict = vader.get_loss(X_val)
    train_reconstruction_loss, train_latent_loss = vader.reconstruction_loss[-1], vader.latent_loss[-1]
    test_reconstruction_loss, test_latent_loss = test_loss_dict["reconstruction_loss"], test_loss_dict["latent_loss"]
    # noinspection PyTypeChecker
    effective_k = len(Counter(vader.cluster(X_train)))
    # noinspection PyTypeChecker
    y_pred = vader.cluster(X_val)

    # calculate total loss
    alpha = params_dict["alpha"] if params_dict["alpha"] else 1
    train_total_loss = train_reconstruction_loss + alpha * train_latent_loss
    test_total_loss = test_reconstruction_loss + alpha * test_latent_loss

    # calculate y_true
    vader = fit_vader(X_val, W_val, seed, save_path, params_dict)
    # noinspection PyTypeChecker
    y_true = vader.cluster(X_val)

    # evaluate clustering
    adj_rand_index = calc_adj_rand_index(y_pred, y_true)
    rand_index = calc_rand_index(y_pred, y_true)
    prediction_strength = calc_prediction_strength(y_pred, y_true)
    permuted_clustering_evaluation_metrics = calc_permuted_clustering_evaluation_metrics(y_pred, y_true,
                                                                                         params_dict["n_perm"])
    return {
        "train_reconstruction_loss": train_reconstruction_loss,
        "train_latent_loss": train_latent_loss,
        "train_total_loss": train_total_loss,
        "test_reconstruction_loss": test_reconstruction_loss,
        "test_latent_loss": test_latent_loss,
        "test_total_loss": test_total_loss,
        "effective_k": effective_k,
        "rand_index": rand_index,
        "rand_index_null": permuted_clustering_evaluation_metrics["rand_index"],
        "adj_rand_index": adj_rand_index,
        "adj_rand_index_null": permuted_clustering_evaluation_metrics["adj_rand_index"],
        "prediction_strength": prediction_strength,
        "prediction_strength_null": permuted_clustering_evaluation_metrics["prediction_strength"],
    }


def cv_fold_step_pre_optimize(X_train: ndarray, X_val: ndarray, W_train: Optional[ndarray], W_val: Optional[ndarray],
                              seed: int, save_path: str, params_dict: ParamsDictType) -> Dict[str, Union[int, float]]:

    vader = fit_nonvar_vader(X_train, W_train, seed, save_path, params_dict)
    # noinspection PyTypeChecker
    test_loss_dict = vader.get_loss(X_val)
    train_reconstruction_loss, train_latent_loss = vader.reconstruction_loss[-1], vader.latent_loss[-1]
    test_reconstruction_loss, test_latent_loss = test_loss_dict["reconstruction_loss"], test_loss_dict["latent_loss"]

    return {
        "train_reconstruction_loss": train_reconstruction_loss,
        "train_latent_loss": train_latent_loss,
        "test_reconstruction_loss": test_reconstruction_loss,
        "test_latent_loss": test_latent_loss
    }


def fit_vader(X_train: ndarray, W_train: Optional[ndarray], seed: int, save_path: str,
              params_dict: ParamsDictType) -> VADER:
    k = params_dict["k"]
    n_hidden = params_dict["n_hidden"]
    learning_rate = params_dict["learning_rate"]
    batch_size = params_dict["batch_size"]
    alpha = params_dict["alpha"]
    n_epoch = params_dict["n_epoch"]
    # noinspection PyTypeChecker
    vader = VADER(X_train=X_train, W_train=W_train, save_path=save_path, n_hidden=n_hidden, k=k, seed=seed,
                  learning_rate=learning_rate, recurrent=True, batch_size=batch_size, alpha=alpha)

    vader.pre_fit(n_epoch=n_epoch, verbose=False)
    vader.fit(n_epoch=n_epoch, verbose=False)
    return vader


def fit_nonvar_vader(X_train: ndarray, W_train: Optional[ndarray], seed: int, save_path: str,
                     params_dict: ParamsDictType) -> VADER:
    k = 1
    n_hidden = params_dict["n_hidden"]
    learning_rate = params_dict["learning_rate"]
    batch_size = params_dict["batch_size"]
    alpha = 0
    n_epoch = params_dict["n_epoch"]
    # noinspection PyTypeChecker
    vader = VADER(X_train=X_train, W_train=W_train, save_path=save_path, n_hidden=n_hidden, k=k, seed=seed,
                  learning_rate=learning_rate, recurrent=True, batch_size=batch_size, alpha=alpha)

    vader.pre_fit(n_epoch=n_epoch, verbose=False)
    return vader


def calc_rand_index(y_pred: ndarray, y_true: ndarray) -> float:
    clusters = y_true
    classes = y_pred
    # See: https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation
    tp_plus_fp = comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(comb(np.bincount(A[A[:, 0] == i, 1]), 2).sum()
             for i in set(clusters))
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def calc_adj_rand_index(y_pred: ndarray, y_true: ndarray) -> float:
    return adjusted_rand_score(y_true, y_pred)


def calc_prediction_strength(y_pred: ndarray, y_true: ndarray) -> float:
    # TODO: investigate strange behaviour (e.g. [1,1,2,2,3], [1,1,2,2,3])
    return calc_prediction_strength_legacy(y_pred, y_true)


def calc_prediction_strength_legacy(p: ndarray, q: ndarray) -> float:
    def f(y: ndarray) -> ndarray:
        m = [y for _ in range(len(y))]
        return m == np.transpose(m)

    n = len(p)
    mp = f(p)
    mq = f(q)
    mpq = mp & mq
    pr_str_vector = pd.DataFrame(range(n)).groupby(q).apply(
        lambda ii: (np.sum(mpq[:, ii]) - len(ii)) / len(ii) / (n - 1)
    )
    return min(pr_str_vector)


def calc_permuted_clustering_evaluation_metrics(y_pred: ndarray, y_true: ndarray, n_perm: int):
    metrics_dict = {}
    for i in range(n_perm):
        sample_y_pred = np.random.permutation(y_pred)
        adj_rand_index = calc_adj_rand_index(sample_y_pred, y_true)
        rand_index = calc_rand_index(sample_y_pred, y_true)
        prediction_strength = calc_prediction_strength(sample_y_pred, y_true)
        metrics_dict[i] = [adj_rand_index, rand_index, prediction_strength]
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index',
                                        columns=['adj_rand_index', 'rand_index', 'prediction_strength'])
    return metrics_df.mean()


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

    run_hyperparameters_optimization(
        input_data_file=input_data_file,
        input_weights_file=input_weights_file,
        input_param_grid_file=input_param_grid_file,
        input_seed=input_seed,
        n_repeats=n_repeats,
        n_proc=n_proc,
        n_sample=n_sample,
        output_save_path=output_save_path,
        output_evaluation_path=output_evaluation_path,
        stage=stage,
        verbose=verbose
    )
