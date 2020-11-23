import os
import sys
import argparse
import pandas as pd
import scipy.stats


PARAMS_COLUMN_NAME = "params"
PRED_STRENGTH_COLUMN_NAME = "prediction_strength"
PRED_STRENGTH_NULL_COLUMN_NAME = "prediction_strength_null"
WILCOXON_PRED_STRENGTH_COLUMN_NAME = "wilcoxon_prediction_strength"


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_cv_results_file", type=str)
    parser.add_argument("--metric", type=str, default=WILCOXON_PRED_STRENGTH_COLUMN_NAME)
    parser.add_argument("--minimize", action='store_true')
    parser.add_argument("--maximize", action='store_true')
    args = parser.parse_args()

    if not os.path.exists(args.input_cv_results_file):
        print("ERROR: input cv results file does not exist")
        sys.exit(1)

    if args.minimize and args.maximize:
        print("ERROR: cannot minimize and maximize the target metric simultaneously")
        sys.exit(2)

    if not args.minimize and not args.maximize:
        minimize = True
    else:
        minimize = args.minimize

    df = pd.read_csv(args.input_cv_results_file, index_col=0)
    df_means = df.groupby(PARAMS_COLUMN_NAME).mean()
    try:
        df_means[WILCOXON_PRED_STRENGTH_COLUMN_NAME] = df.groupby(PARAMS_COLUMN_NAME).apply(
            # lambda group: sum(group["prediction_strength"]) + sum(group["prediction_strength_null"])
            lambda group: scipy.stats.wilcoxon(group[PRED_STRENGTH_COLUMN_NAME], group[PRED_STRENGTH_NULL_COLUMN_NAME])
        )
    except ValueError:
        print(f"Cannot calculate {WILCOXON_PRED_STRENGTH_COLUMN_NAME}. Unexpected error:", sys.exc_info()[1])

    performance_metric = args.metric
    max_performance_metric_value = df_means[performance_metric].min() if minimize else df_means[performance_metric].max()
    best_parameters_sets = df_means[df_means[performance_metric] == max_performance_metric_value].index.values
    print(f"{best_parameters_sets} with best {performance_metric}={max_performance_metric_value}")
