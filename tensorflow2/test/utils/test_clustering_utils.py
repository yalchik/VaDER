import numpy as np
import pandas as pd
from pytest import approx
from pandas.testing import assert_series_equal, assert_frame_equal
from vader.utils.clustering_utils import ClusteringUtils


class TestClusteringUtils:

    def test_calc_rand_index(self):
        assert ClusteringUtils.calc_rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == approx(1)
        assert ClusteringUtils.calc_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == approx(1)
        assert ClusteringUtils.calc_rand_index([0, 0, 1, 2], [0, 0, 1, 1]) == approx(0.8333333)
        assert ClusteringUtils.calc_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == approx(0.8333333)
        assert ClusteringUtils.calc_rand_index([0, 0, 0, 0], [0, 1, 2, 3]) == approx(0)

    def test_calc_adj_rand_index(self):
        assert ClusteringUtils.calc_adj_rand_index([0, 0, 1, 1], [0, 0, 1, 1]) == approx(1)
        assert ClusteringUtils.calc_adj_rand_index([0, 0, 1, 1], [1, 1, 0, 0]) == approx(1)
        assert ClusteringUtils.calc_adj_rand_index([0, 0, 1, 2], [0, 0, 1, 1]) == approx(0.5714285)
        assert ClusteringUtils.calc_adj_rand_index([0, 0, 1, 1], [0, 0, 1, 2]) == approx(0.5714285)
        assert ClusteringUtils.calc_adj_rand_index([0, 0, 0, 0], [0, 1, 2, 3]) == approx(0)

    def test_calc_prediction_strength(self):
        assert ClusteringUtils.calc_prediction_strength([0, 0, 0, 0], [0, 0, 0, 0]) == approx(1)
        assert ClusteringUtils.calc_prediction_strength([0, 0, 0, 0], [1, 1, 1, 1]) == approx(1)

        assert ClusteringUtils.calc_prediction_strength(
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
        ) == approx(0.46153846)
        assert ClusteringUtils.calc_prediction_strength(
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 1]
        ) == approx(0.32967032)
        assert ClusteringUtils.calc_prediction_strength([0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]) == approx(0.4)
        assert ClusteringUtils.calc_prediction_strength([0, 1, 0, 1, 0, 1], [0, 1, 0, 1, 0, 1]) == approx(0.4)
        assert ClusteringUtils.calc_prediction_strength([0, 0, 1, 1], [0, 0, 1, 1]) == approx(0.3333333)
        assert ClusteringUtils.calc_prediction_strength([0, 0, 1, 1], [1, 1, 0, 0]) == approx(0.3333333)
        assert ClusteringUtils.calc_prediction_strength([0, 0, 0, 0], [0, 0, 0, 1]) == approx(0)
        assert ClusteringUtils.calc_prediction_strength([0, 0, 1, 2], [0, 0, 1, 1]) == approx(0)
        assert ClusteringUtils.calc_prediction_strength([0, 0, 1, 1], [0, 0, 1, 2]) == approx(0)
        assert ClusteringUtils.calc_prediction_strength([0, 0, 0, 0], [0, 1, 2, 3]) == approx(0)

    def test_calc_permuted_clustering_evaluation_metrics(self):
        np.random.seed(42)
        assert_series_equal(ClusteringUtils.calc_permuted_clustering_evaluation_metrics(
            [0, 0, 0, 0], [0, 0, 0, 0], 3
        ), pd.Series([1.0, 1.0, 1.0], index=ClusteringUtils.METRICS_LIST))
        assert_series_equal(ClusteringUtils.calc_permuted_clustering_evaluation_metrics(
            [0, 0, 1, 1], [0, 0, 1, 1], 3
        ), pd.Series([0.5, 0.777777, 0.222222], index=ClusteringUtils.METRICS_LIST))
        assert_series_equal(ClusteringUtils.calc_permuted_clustering_evaluation_metrics(
            [0, 0, 1, 1], [1, 1, 0, 0], 3
        ), pd.Series([0.5, 0.777777, 0.222222], index=ClusteringUtils.METRICS_LIST))
        assert_series_equal(ClusteringUtils.calc_permuted_clustering_evaluation_metrics(
            [0, 0, 1, 2], [0, 0, 1, 1], 3
        ), pd.Series([0.0, 0.611111, 0.0], index=ClusteringUtils.METRICS_LIST))
        assert_series_equal(ClusteringUtils.calc_permuted_clustering_evaluation_metrics(
            [0, 0, 1, 1], [0, 0, 1, 2], 3
        ), pd.Series([0.0, 0.611111, 0.0], index=ClusteringUtils.METRICS_LIST))
        assert_series_equal(ClusteringUtils.calc_permuted_clustering_evaluation_metrics(
            [0, 0, 0, 0], [0, 1, 2, 3], 3
        ), pd.Series([0.0, 0.0, 0.0], index=ClusteringUtils.METRICS_LIST))

    def test_consensus_clustering(self):
        clusterings_list = [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 1, 1, 1, 2, 2, 2, 0],
            [0, 1, 1, 1, 1, 2, 2, 2, 2]
        ]
        assert ClusteringUtils.consensus_clustering(clusterings_list, 1) == approx([1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert ClusteringUtils.consensus_clustering(clusterings_list, 2) == approx([1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert ClusteringUtils.consensus_clustering(clusterings_list, 3) == approx([2, 2, 3, 3, 3, 1, 1, 1, 1])
        assert ClusteringUtils.consensus_clustering(clusterings_list, 4) == approx([3, 3, 4, 4, 4, 2, 1, 1, 1])

        clusterings_list = [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 1, 1, 1, 2, 2, 2, 0],
            [0, 1, 2, 1, 2, 2, 2, 0, 1],
            [1, 1, 2, 1, 2, 0, 2, 2, 2],
            [2, 0, 1, 1, 2, 2, 1, 0, 2]
        ]
        assert ClusteringUtils.consensus_clustering(clusterings_list, 1) == approx([1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert ClusteringUtils.consensus_clustering(clusterings_list, 2) == approx([1, 1, 1, 1, 1, 1, 1, 1, 1])
        assert ClusteringUtils.consensus_clustering(clusterings_list, 3) == approx([1, 1, 3, 3, 3, 3, 2, 2, 1])
        assert ClusteringUtils.consensus_clustering(clusterings_list, 4) == approx([1, 1, 3, 3, 3, 4, 2, 2, 1])

    def test_calc_distance_matrix(self):
        clusterings_list = [
            [0, 0, 0, 1, 1, 1, 2, 2, 2],
            [0, 0, 1, 1, 1, 2, 2, 2, 0],
            [0, 1, 2, 1, 2, 2, 2, 0, 1],
            [1, 1, 2, 1, 2, 0, 2, 2, 2],
            [2, 0, 1, 1, 2, 2, 1, 0, 2]
        ]
        assert_frame_equal(
            pd.DataFrame(ClusteringUtils.calc_distance_matrix(clusterings_list)),
            pd.DataFrame([
                [0.0, 0.4, 0.8, 0.8, 0.8, 0.8, 1.0, 0.8, 0.6],
                [0.4, 0.0, 0.8, 0.6, 1.0, 1.0, 1.0, 0.8, 0.6],
                [0.8, 0.8, 0.0, 0.6, 0.4, 0.8, 0.4, 0.8, 0.8],
                [0.8, 0.6, 0.6, 0.0, 0.6, 0.8, 0.8, 1.0, 0.8],
                [0.8, 1.0, 0.4, 0.6, 0.0, 0.4, 0.6, 0.8, 0.6],
                [0.8, 1.0, 0.8, 0.8, 0.4, 0.0, 0.6, 0.8, 0.8],
                [1.0, 1.0, 0.4, 0.8, 0.6, 0.6, 0.0, 0.4, 0.6],
                [0.8, 0.8, 0.8, 1.0, 0.8, 0.8, 0.4, 0.0, 0.6],
                [0.6, 0.6, 0.8, 0.8, 0.6, 0.8, 0.6, 0.6, 0.0]
            ])
        )

    def test_calc_linkage(self):
        distance_matrix = [
            [0.0, 0.4, 0.8, 0.8, 0.8, 0.8, 1.0, 0.8, 0.6],
            [0.4, 0.0, 0.8, 0.6, 1.0, 1.0, 1.0, 0.8, 0.6],
            [0.8, 0.8, 0.0, 0.6, 0.4, 0.8, 0.4, 0.8, 0.8],
            [0.8, 0.6, 0.6, 0.0, 0.6, 0.8, 0.8, 1.0, 0.8],
            [0.8, 1.0, 0.4, 0.6, 0.0, 0.4, 0.6, 0.8, 0.6],
            [0.8, 1.0, 0.8, 0.8, 0.4, 0.0, 0.6, 0.8, 0.8],
            [1.0, 1.0, 0.4, 0.8, 0.6, 0.6, 0.0, 0.4, 0.6],
            [0.8, 0.8, 0.8, 1.0, 0.8, 0.8, 0.4, 0.0, 0.6],
            [0.6, 0.6, 0.8, 0.8, 0.6, 0.8, 0.6, 0.6, 0.0]
        ]
        assert_frame_equal(
            pd.DataFrame(ClusteringUtils.calc_linkage(distance_matrix)),
            pd.DataFrame([
                [0.0,  1.0,  0.4, 2.0],
                [2.0,  4.0,  0.4, 2.0],
                [6.0,  7.0,  0.4, 2.0],
                [8.0,  9.0,  0.6, 3.0],
                [3.0,  10.0, 0.6, 3.0],
                [5.0,  13.0, 0.8, 4.0],
                [11.0, 14.0, 1.0, 6.0],
                [12.0, 15.0, 1.0, 9.0]
            ])
        )

    def test_std_diff(self):
        assert_series_equal(ClusteringUtils.std_diff_legacy(
            pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        ), pd.Series([0.0, 0.0, 0.0]))
        assert_series_equal(ClusteringUtils.std_diff_legacy(
            pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [8, 6, 8]]))
        ), pd.Series([0.5, 1.0, 0.5]))
        assert_series_equal(ClusteringUtils.std_diff_legacy(
            pd.DataFrame(np.array([[1.1, 2.4, 3.7], [4.1, 5.5, 6.9], [7.77, 8.88, 9.99]]))
        ), pd.Series([0.335, 0.14, 0.055]))

    def test_calc_distribution(self):
        df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [4, 5, 6]]))
        mu, sigma = ClusteringUtils.calc_distribution(df)
        assert_series_equal(mu, pd.Series([3.0, 4.0, 5.0]))
        assert_series_equal(sigma, pd.Series([1.697409, 1.697409, 1.697409]))

        df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        mu, sigma = ClusteringUtils.calc_distribution(df)
        assert_series_equal(mu, pd.Series([4.0, 5.0, 6.0]))
        assert_series_equal(sigma, pd.Series([0.0, 0.0, 0.0]))

        df = pd.DataFrame(np.array([[1, 2, 3], [4, 5, 6], [8, 6, 8]]))
        mu, sigma = ClusteringUtils.calc_distribution(df)
        assert_series_equal(mu, pd.Series([4.333333, 4.333333, 5.666666]))
        assert_series_equal(sigma, pd.Series([0.565803, 1.131606, 0.565803]))

        df = pd.DataFrame(np.array([[1.1, 2.4, 3.7], [4.1, 5.5, 6.9], [7.77, 8.88, 9.99]]))
        mu, sigma = ClusteringUtils.calc_distribution(df)
        assert_series_equal(mu, pd.Series([4.323333, 5.593333, 6.863333]))
        assert_series_equal(sigma, pd.Series([0.379088, 0.158424, 0.062238]))
