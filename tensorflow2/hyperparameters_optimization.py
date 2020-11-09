import os
import sys
import argparse
import json
from vader import VADER
from vader.utils import read_adni_data
from collections import Counter
from sklearn.model_selection import KFold
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.special import comb
import pandas as pd
import numpy as np
import multiprocessing as mp


def run_cv(data, weights, seed, save_path, params_dict, verbose=False):
    if verbose:
        print("=> run_cv with:", params_dict)
    n_splits = params_dict["n_splits"]
    cv_folds_results_list = []
    for train_index, val_index in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(data):
        X_train, X_val = data[train_index], data[val_index]
        W_train, W_val = (weights[train_index], weights[val_index]) if weights else None, None
        cv_fold_result = cv_fold_step(X_train, X_val, W_train, W_val, seed, save_path, params_dict)
        cv_folds_results_list.append(cv_fold_result)

    if verbose:
        print("<= run_cv results:", cv_folds_results_list)

    cv_folds_results_df = pd.DataFrame(cv_folds_results_list)
    cv_mean_results_series = cv_folds_results_df.mean()
    cv_mean_results_series["params"] = str(params_dict)
    return cv_mean_results_series


def cv_fold_step(X_train, X_val, W_train, W_val, seed, save_path, params_dict):
    # calculate y_pred
    vader = fit_vader(X_train, W_train, seed, save_path, params_dict)
    test_loss_dict = vader.get_loss(X_train)
    train_reconstruction_loss, train_latent_loss = vader.reconstruction_loss[-1], vader.latent_loss[-1]
    test_reconstruction_loss, test_latent_loss = test_loss_dict["reconstruction_loss"], test_loss_dict["latent_loss"]
    effective_k = len(Counter(vader.cluster(X_train)))
    y_pred = vader.cluster(X_val)

    # calculate y_true
    vader = fit_vader(X_val, W_val, seed, save_path, params_dict)
    y_true = vader.cluster(X_val)

    # evaluate clustering
    adj_rand_index = calc_adj_rand_index(y_pred, y_pred)
    rand_index = calc_rand_index(y_pred, y_pred)
    prediction_strength = calc_prediction_strength(y_pred, y_pred)
    permuted_clustering_evaluation_metrics = calc_permuted_clustering_evaluation_metrics(y_pred, y_true,
                                                                                         params_dict["n_perm"])
    return {
        "train_reconstruction_loss": train_reconstruction_loss,
        "train_latent_loss": train_latent_loss,
        "test_reconstruction_loss": test_reconstruction_loss,
        "test_latent_loss": test_latent_loss,
        "effective_k": effective_k,
        "rand_index": rand_index,
        "rand_index_null": permuted_clustering_evaluation_metrics["rand_index"],
        "adj_rand_index": adj_rand_index,
        "adj_rand_index_null": permuted_clustering_evaluation_metrics["adj_rand_index"],
        "prediction_strength": prediction_strength,
        "prediction_strength_null": permuted_clustering_evaluation_metrics["prediction_strength"],
    }


def fit_vader(X_train, W_train, seed, save_path, params_dict) -> VADER:
    k = params_dict["k"]
    n_hidden = params_dict["n_hidden"]
    learning_rate = params_dict["learning_rate"]
    batch_size = params_dict["batch_size"]
    n_epoch = params_dict["n_epoch"]
    vader = VADER(X_train=X_train, W_train=W_train, save_path=save_path, n_hidden=n_hidden, k=k, seed=seed,
                  learning_rate=learning_rate, recurrent=True, batch_size=batch_size)

    vader.pre_fit(n_epoch=n_epoch, verbose=False)
    vader.fit(n_epoch=n_epoch, verbose=False)
    return vader


def calc_rand_index(y_pred, y_true) -> float:
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


def calc_adj_rand_index(y_pred, y_true) -> float:
    return adjusted_rand_score(y_true, y_pred)


def calc_prediction_strength(y_pred, y_true) -> float:
    # TODO: Not implemented yet
    return 0


def calc_permuted_clustering_evaluation_metrics(y_pred, y_true, n_perm):
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


def read_param_grid(param_grid_file):
    with open(param_grid_file, 'r') as json_file:
        input_param_grid = json.load(json_file)
    return input_param_grid


def run_cv_parallel(params):
    return run_cv(*params)


def run_hyperparameters_optimization(input_data_file, input_weights_file, input_param_grid_file, input_seed,
                                     n_proc, output_save_path, output_evaluation_path):
    # read files
    input_data = read_adni_data(input_data_file)
    input_weights = read_adni_data(input_weights_file) if input_weights_file else None
    input_param_grid = read_param_grid(input_param_grid_file)

    # cross-validation
    with mp.Pool(n_proc) as pool:
        cv_results_list = pool.map(run_cv_parallel, [(input_data, input_weights, input_seed, output_save_path,
                                                      params_dict) for params_dict in input_param_grid])

    print(cv_results_list)
    pd.DataFrame(cv_results_list).to_csv(output_evaluation_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_file", type=str, help="a .csv file with input data")
    parser.add_argument("--input_weights_file", type=str, help="a .csv file with flags for missing values")
    parser.add_argument("input_param_grid_file", type=str, help="hyperparameters values for grid search")
    parser.add_argument("--input_seed", type=int, help="used both as random_state and VaDER seed")
    parser.add_argument("--n_proc", type=int, help="number of processor units that can be used")
    parser.add_argument("--output_save_path", type=str, help="a directory where all models will be saved")
    parser.add_argument("output_evaluation_path", type=str, help="a .csv file where cross-validation results will be "
                                                                 "written")
    args = parser.parse_args()

    if not os.path.exists(args.input_data_file):
        print("ERROR: input data file does not exist")
        sys.exit(1)

    if args.input_weights_file and not os.path.exists(args.input_weights_file):
        print("ERROR: weights data file does not exist")
        sys.exit(2)

    if not os.path.exists(args.input_param_grid_file):
        print("ERROR: param grid file does not exist")
        sys.exit(3)

    input_data_file = args.input_data_file
    input_weights_file = args.input_weights_file
    input_param_grid_file = args.input_param_grid_file
    input_seed = args.input_seed if args.input_seed else None
    n_proc = args.n_proc if args.n_proc else mp.cpu_count()
    output_save_path = args.output_save_path
    output_evaluation_path = args.output_evaluation_path

    run_hyperparameters_optimization(input_data_file, input_weights_file, input_param_grid_file, input_seed,
                                     n_proc, output_save_path, output_evaluation_path)
