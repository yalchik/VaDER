import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.special import comb
from sklearn.metrics.cluster import adjusted_rand_score


class ClusteringMetrics:
    @staticmethod
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

    @staticmethod
    def calc_adj_rand_index(y_pred: ndarray, y_true: ndarray) -> float:
        return adjusted_rand_score(y_true, y_pred)

    @staticmethod
    def calc_prediction_strength(y_pred: ndarray, y_true: ndarray) -> float:
        # TODO: investigate strange behaviour (e.g. [1,1,2,2,3], [1,1,2,2,3])
        return ClusteringMetrics.calc_prediction_strength_legacy(y_pred, y_true)

    @staticmethod
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

    @staticmethod
    def calc_permuted_clustering_evaluation_metrics(y_pred: ndarray, y_true: ndarray, n_perm: int):
        metrics_dict = {}
        for i in range(n_perm):
            sample_y_pred = np.random.permutation(y_pred)
            adj_rand_index = ClusteringMetrics.calc_adj_rand_index(sample_y_pred, y_true)
            rand_index = ClusteringMetrics.calc_rand_index(sample_y_pred, y_true)
            prediction_strength = ClusteringMetrics.calc_prediction_strength(sample_y_pred, y_true)
            metrics_dict[i] = [adj_rand_index, rand_index, prediction_strength]
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index',
                                            columns=['adj_rand_index', 'rand_index', 'prediction_strength'])
        return metrics_df.mean()
