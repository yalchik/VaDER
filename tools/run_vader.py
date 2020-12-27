import os
import sys
import argparse
import vader.utils
from collections import Counter
from vader import VADER
from vader.utils import read_adni_data


if __name__ == "__main__":
    """
    The script runs VaDER model with a given set of hyperparameters on given data.
    
    Example:
    python run_vader.py --adni_data_file=../data/ADNI/Xnorm.csv
                        --save_path=../vader_results/model/
                        --report_file_path=../vader_results/run_vader_report.txt
                        --k=4 --n_hidden 128 8 --learning_rate=1e-3 --batch_size=32 --alpha=1
                        --n_repeats=1 --n_epoch=20                        
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--adni_data_file", type=str, help="a .csv file with ADNI data", required=True)
    parser.add_argument("--n_repeats", type=int, default=1, help="number of repeats")
    parser.add_argument("--n_epoch", type=int, default=20, help="number of training epochs")
    parser.add_argument("--k", type=int, help="number of repeats", required=True)
    parser.add_argument("--n_hidden", nargs='+', help="hidden layers", required=True)
    parser.add_argument("--learning_rate", type=float, help="learning rate", required=True)
    parser.add_argument("--batch_size", type=int, help="batch size", required=True)
    parser.add_argument("--alpha", type=float, help="alpha", required=True)
    parser.add_argument("--save_path", type=str, help="model save path")
    parser.add_argument("--seed", type=int, help="seed")
    parser.add_argument("--report_file_path", type=str, required=True)
    args = parser.parse_args()

    if args.adni_data_file and not os.path.exists(args.adni_data_file):
        print("ERROR: ADNI data file does not exist")
        sys.exit(1)

    X_train = read_adni_data(args.adni_data_file)
    n_hidden = [int(layer_size) for layer_size in args.n_hidden]

    for _ in range(args.n_repeats):
        # noinspection PyTypeChecker
        vader = VADER(X_train=X_train, k=args.k, n_hidden=n_hidden, learning_rate=args.learning_rate,
                      batch_size=args.batch_size, alpha=args.alpha, seed=args.seed, save_path=args.save_path,
                      output_activation=None, recurrent=True)
        vader.pre_fit(n_epoch=10, verbose=False)
        vader.fit(n_epoch=args.n_epoch, verbose=False)
        clustering = vader.cluster(X_train)
        reconstruction_loss, latent_loss = vader.reconstruction_loss[-1], vader.latent_loss[-1]
        total_loss = reconstruction_loss + args.alpha * latent_loss
        with open(args.report_file_path, "a") as f:
            f.write(f"Proportion: {Counter(clustering)}\n"
                    f"Reconstruction loss: {reconstruction_loss}\n"
                    f"Lat loss: {latent_loss}\n"
                    f"Total loss: {total_loss}\n"
                    f"{clustering}\n\n")
