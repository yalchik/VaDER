import numpy as np
import pandas as pd
from numpy import ndarray
from scipy.special import comb
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from vader.hp_opt.common import ClusteringType
from typing import List, Tuple, Union, Dict


class ClusteringUtils:
    """Contains static methods that help in the clusterization process."""

    METRICS_LIST = ['adj_rand_index', 'rand_index', 'prediction_strength']

    @staticmethod
    def calc_rand_index(y_pred: ClusteringType, y_true: ClusteringType) -> float:
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
    def calc_adj_rand_index(y_pred: ClusteringType, y_true: ClusteringType) -> float:
        return adjusted_rand_score(y_true, y_pred)

    @staticmethod
    def calc_prediction_strength(y_pred: ClusteringType, y_true: ClusteringType) -> float:
        # TODO: investigate strange behaviour (e.g. [1,1,2,2,3], [1,1,2,2,3])
        return ClusteringUtils.calc_prediction_strength_legacy(y_pred, y_true)

    @staticmethod
    def calc_prediction_strength_legacy(p: ClusteringType, q: ClusteringType) -> float:
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
    def calc_permuted_clustering_evaluation_metrics(y_pred: ClusteringType, y_true: ClusteringType, n_perm: int) \
            -> pd.Series:
        metrics_dict = {}
        for i in range(n_perm):
            sample_y_pred = np.random.permutation(y_pred)
            adj_rand_index = ClusteringUtils.calc_adj_rand_index(sample_y_pred, y_true)
            rand_index = ClusteringUtils.calc_rand_index(sample_y_pred, y_true)
            prediction_strength = ClusteringUtils.calc_prediction_strength(sample_y_pred, y_true)
            metrics_dict[i] = [adj_rand_index, rand_index, prediction_strength]
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=ClusteringUtils.METRICS_LIST)
        return metrics_df.mean()

    @staticmethod
    def consensus_clustering(clusterings_list: List[ClusteringType], num_of_clusters: int) -> ClusteringType:
        distMatrix = ClusteringUtils.calc_distance_matrix(clusterings_list)
        items_linkage = ClusteringUtils.calc_linkage(distMatrix)
        clustering = list(fcluster(items_linkage, num_of_clusters, criterion='maxclust'))
        return clustering

    @staticmethod
    def calc_distance_matrix(clusterings_list: List[ClusteringType]) -> ndarray:
        # TODO: optimize the performance
        m = len(clusterings_list)
        n = len(clusterings_list[0])
        M = np.zeros((n, n))
        for y_pred in clusterings_list:
            for i, y_item_1 in enumerate(y_pred):
                for j, y_item_2 in enumerate(y_pred):
                    if y_item_1 == y_item_2:
                        M[i, j] += 1
        distance_matrix = 1 - M / m
        return distance_matrix

    @staticmethod
    def calc_linkage(distance_matrix: Union[ndarray, List[List[float]]]) -> ndarray:
        distance_array = squareform(distance_matrix)  # convert n*n square matrix form into a condensed nC2 array
        linkage_array = linkage(distance_array, 'complete')
        return linkage_array

    @staticmethod
    def std_diff(df: pd.DataFrame) -> pd.Series:
        """Translation of the 'colSdDiff' function from the R package 'matrixStats'"""
        # std_diff = df.diff().std() / np.sqrt(2)
        std_diff = [df[c].dropna().diff().std() / np.sqrt(2) for c in df.columns]
        return std_diff

    @staticmethod
    def calc_distribution(df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        mu = df.mean()
        sigma = ClusteringUtils.std_diff(df) / np.sqrt(df.notna().sum()) * 1.96  # 95% CI
        return mu, sigma

    @staticmethod
    def calc_distribution_v2(df):
        mu = df.mean()
        std_diff = [df[c].dropna().diff().std() / np.sqrt(2) for c in df.columns]
        sigma = std_diff / np.sqrt(df.notna().sum()) * 1.96  # 95% CI
        return mu, sigma

    @staticmethod
    def calc_z_scores(X_train: ndarray, std_per_feature: ndarray = None) -> ndarray:
        Xnorm = np.zeros(X_train.shape)
        if std_per_feature is None:
            std_per_feature = pd.DataFrame(X_train.reshape(-1, 3)).std().to_numpy()
        n_features = X_train.shape[2]
        for i in range(n_features):
            Xnorm[:, :, i] = (X_train[:, :, i] - np.vstack(X_train[:, 0, i])) / std_per_feature[i]
        return Xnorm

    @staticmethod
    def clustering_to_dict(clustering_list: ClusteringType) -> Dict[int, List[int]]:
        clusters_dict = {}
        for i, cluster in enumerate(clustering_list):
            if cluster not in clusters_dict:
                clusters_dict[cluster] = []
            clusters_dict[cluster].append(i)

        clusters_dict_reordered = {}
        for i, k in enumerate(sorted(clusters_dict, key=lambda k: len(clusters_dict[k]), reverse=True)):
            clusters_dict_reordered[i] = clusters_dict[k]
        return clusters_dict_reordered
