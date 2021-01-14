import os
import sys
import argparse
import numpy as np
from typing import Tuple
from collections import Counter
from vader import VADER
from vader.utils.data_utils import read_adni_norm_data, read_nacc_data
from vader.utils.clustering_utils import ClusteringUtils


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
    The script runs VaDER model with a given set of hyperparameters on given data.
    It computes clustering for the given data and writes it to a report file.
    
    Example:
    python run_vader.py --input_data_file=../data/ADNI/Xnorm.csv
                        --input_data_type=ADNI
                        --save_path=../vader_results/model/
                        --report_file_path=../vader_results/run_vader_report.txt
                        --k=4 --n_hidden 128 8 --learning_rate=1e-3 --batch_size=32 --alpha=1
                        --n_repeats=1 --n_epoch=20                        
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data_file", type=str, help="a .csv file with input data", required=True)
    parser.add_argument("--input_weights_file", type=str, help="a .csv file with flags for missing values")
    parser.add_argument("--input_data_type", choices=["ADNI", "NACC", "PPMI", "custom"], help="data type",
                        required=True)
    parser.add_argument("--n_repeats", type=int, default=1, help="number of repeats")
    parser.add_argument("--n_epoch", type=int, default=20, help="number of training epochs")
    parser.add_argument("--n_consensus", type=int, default=1, help="number of repeats for consensus clustering")
    parser.add_argument("--k", type=int, help="number of repeats", required=True)
    parser.add_argument("--n_hidden", nargs='+', help="hidden layers", required=True)
    parser.add_argument("--learning_rate", type=float, help="learning rate", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size", required=True)
    parser.add_argument("--alpha", type=float, help="alpha", required=True)
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--report_file_path", type=str, required=True)
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
    elif args.input_data_type == "custom":
        input_data, weights = read_custom_data(args.input_data_file)
        input_weights, _ = read_custom_data(args.input_weights_file) if args.input_weights_file else weights, None
    else:
        print("ERROR: Unknown data type.")
        exit(4)

    n_hidden = [int(layer_size) for layer_size in args.n_hidden]

    for i in range(args.n_repeats):
        if args.n_consensus and args.n_consensus > 1:
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
                vader.fit(n_epoch=args.n_epoch, verbose=False)
                clustering = vader.cluster(input_data, input_weights)
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
        else:
            seed = f"{args.seed}{i}" if args.seed else None
            # noinspection PyTypeChecker
            vader = VADER(X_train=input_data, W_train=input_weights, k=args.k, n_hidden=n_hidden,
                          learning_rate=args.learning_rate, batch_size=args.batch_size, alpha=args.alpha,
                          seed=args.seed, save_path=args.save_path, output_activation=None, recurrent=True)
            vader.pre_fit(n_epoch=10, verbose=False)
            vader.fit(n_epoch=args.n_epoch, verbose=False)
            clustering = vader.cluster(input_data, input_weights)
            reconstruction_loss, latent_loss = vader.reconstruction_loss[-1], vader.latent_loss[-1]
        total_loss = reconstruction_loss + args.alpha * latent_loss
        with open(args.report_file_path, "a") as f:
            f.write(f"Proportion: {Counter(clustering)}\n"
                    f"Reconstruction loss: {reconstruction_loss}\n"
                    f"Lat loss: {latent_loss}\n"
                    f"Total loss: {total_loss}\n"
                    f"{clustering}\n\n")
