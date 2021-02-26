import os
import sys
import argparse
import numpy as np
import importlib.util
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from typing import Tuple
from collections import Counter
from vader import VADER
from vader.utils.data_utils import generate_wtensor_from_xtensor
from vader.utils.plot_utils import plot_z_scores, plot_loss_history
from vader.utils.clustering_utils import ClusteringUtils


if __name__ == "__main__":
    """
    The script runs VaDER model with a given set of hyperparameters on given data.
    It computes clustering for the given data and writes it to a report file.
    
    Example:
    python run_vader.py --input_data_file=../data/ADNI/Xnorm.csv
                        --data_reader_script=addons/data_reader/adni_norm_data.py
                        --output_path=../vader_results/clustering/
                        --k=3 --n_hidden 128 8 --learning_rate=1e-3 --batch_size=64 --alpha=1 --n_epoch=20                        
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, help="a .csv file with input data", required=True)
    parser.add_argument("--input_weights_file", type=str, help="a .csv file with flags for missing values")
    parser.add_argument("--data_reader_script", type=str, help="python script declaring data reader class")
    parser.add_argument("--n_epoch", type=int, default=20, help="number of training epochs")
    parser.add_argument("--early_stopping_ratio", type=float, help="early stopping ratio")
    parser.add_argument("--early_stopping_batch_size", type=int, default=5, help="early stopping batch size")
    parser.add_argument("--n_consensus", type=int, default=1, help="number of repeats for consensus clustering")
    parser.add_argument("--k", type=int, help="number of repeats", required=True)
    parser.add_argument("--n_hidden", nargs='+', help="hidden layers", required=True)
    parser.add_argument("--learning_rate", type=float, help="learning rate", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size", required=True)
    parser.add_argument("--alpha", type=float, help="alpha", required=True)
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_data_file):
        print("ERROR: input data file does not exist")
        sys.exit(1)

    if args.input_weights_file and not os.path.exists(args.input_weights_file):
        print("ERROR: weights data file does not exist")
        sys.exit(2)

    if args.data_reader_script and not os.path.exists(args.data_reader_script):
        print("ERROR: data reader file does not exist")
        sys.exit(3)

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)

    # dynamically import data reader
    data_reader_spec = importlib.util.spec_from_file_location("data_reader", args.data_reader_script)
    data_reader_module = importlib.util.module_from_spec(data_reader_spec)
    data_reader_spec.loader.exec_module(data_reader_module)
    data_reader = data_reader_module.DataReader()

    x_tensor, features, time_points, x_label = None, None, None, None
    x_tensor = data_reader.read_data(args.input_data_file)
    w_tensor = generate_wtensor_from_xtensor(x_tensor)
    input_data = np.nan_to_num(x_tensor)
    input_weights = w_tensor
    features = data_reader.features
    time_points = data_reader.time_points
    x_label = data_reader.time_point_meaning
    n_hidden = [int(layer_size) for layer_size in args.n_hidden]

    report_suffix = f"k{str(args.k)}" \
                    f"_n_hidden{'_'.join(args.n_hidden)}" \
                    f"_learning_rate{str(args.learning_rate)}" \
                    f"_batch_size{str(args.batch_size)}" \
                    f"_n_epoch{str(args.n_epoch)}" \
                    f"_n_consensus{str(args.n_consensus)}"
    report_file_path = os.path.join(args.output_path, f"clustering_{report_suffix}.txt")
    plot_file_path = os.path.join(args.output_path, f"z_scores_trajectories_{report_suffix}.pdf")
    loss_history_file_path = os.path.join(args.output_path, f"loss_history_{report_suffix}.pdf")

    if args.n_consensus and args.n_consensus > 1:
        loss_history_pdf = matplotlib.backends.backend_pdf.PdfPages(loss_history_file_path)
        y_pred_repeats = []
        effective_k_repeats = []
        train_reconstruction_loss_repeats = []
        train_latent_loss_repeats = []
        for j in range(args.n_consensus):
            seed = f"{args.seed}{i}{j}" if args.seed else None
            # noinspection PyTypeChecker
            vader = VADER(X_train=input_data, W_train=input_weights, k=args.k, n_hidden=n_hidden,
                          learning_rate=args.learning_rate, batch_size=args.batch_size, alpha=args.alpha,
                          seed=args.seed, save_path=args.save_path, output_activation=None, recurrent=True)
            vader.pre_fit(n_epoch=10, verbose=False)
            vader.fit(n_epoch=args.n_epoch, verbose=False, early_stopping_ratio=args.early_stopping_ratio,
                      early_stopping_batch_size=args.early_stopping_batch_size)
            fig = plot_loss_history(vader, model_name=f"Model #{j}")
            loss_history_pdf.savefig(fig)
            # noinspection PyTypeChecker
            clustering = vader.cluster(input_data, input_weights)
            with open(report_file_path, "a+") as f:
                f.write(f"Proportion: {Counter(clustering)}\n"
                        f"{list(clustering)}\n\n")
            effective_k = len(Counter(clustering))
            y_pred_repeats.append(clustering)
            effective_k_repeats.append(effective_k)
            train_reconstruction_loss_repeats.append(vader.reconstruction_loss[-1])
            train_latent_loss_repeats.append(vader.latent_loss[-1])
        effective_k = np.mean(effective_k_repeats)
        num_of_clusters = round(float(effective_k))
        clustering = ClusteringUtils.consensus_clustering(y_pred_repeats, num_of_clusters)
        reconstruction_loss = np.mean(train_reconstruction_loss_repeats)
        latent_loss = np.mean(train_latent_loss_repeats)
        loss_history_pdf.close()
    else:
        seed = f"{args.seed}{i}" if args.seed else None
        # noinspection PyTypeChecker
        vader = VADER(X_train=input_data, W_train=input_weights, k=args.k, n_hidden=n_hidden,
                      learning_rate=args.learning_rate, batch_size=args.batch_size, alpha=args.alpha,
                      seed=args.seed, save_path=args.save_path, output_activation=None, recurrent=True)
        vader.pre_fit(n_epoch=10, verbose=False)
        vader.fit(n_epoch=args.n_epoch, verbose=False, early_stopping_ratio=args.early_stopping_ratio,
                  early_stopping_batch_size=args.early_stopping_batch_size)
        fig = plot_loss_history(vader)
        fig.savefig(loss_history_file_path)
        # noinspection PyTypeChecker
        clustering = vader.cluster(input_data, input_weights)
        reconstruction_loss, latent_loss = vader.reconstruction_loss[-1], vader.latent_loss[-1]
    total_loss = reconstruction_loss + args.alpha * latent_loss

    with open(report_file_path, "a+") as f:
        f.write(f"Proportion: {Counter(clustering)}\n"
                f"Reconstruction loss: {reconstruction_loss}\n"
                f"Lat loss: {latent_loss}\n"
                f"Total loss: {total_loss}\n"
                f"{list(clustering)}\n\n")

    if features and time_points:
        fig = plot_z_scores(x_tensor, clustering, list(features), time_points, x_label=x_label)
        fig.savefig(plot_file_path)
