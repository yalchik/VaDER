from numpy import ndarray
from collections import Counter
from typing import Dict, Union, Optional
from .abstract_optimization_job import AbstractOptimizationJob
from .clustering_metrics import ClusteringMetrics
from vader import VADER


class Step2OptimizationJob(AbstractOptimizationJob):

    def _cv_fold_step(self, X_train: ndarray, X_val: ndarray, W_train: Optional[ndarray],
                      W_val: Optional[ndarray]) -> Dict[str, Union[int, float]]:
        # calculate y_pred
        vader = self._fit_vader(X_train, W_train)
        # noinspection PyTypeChecker
        test_loss_dict = vader.get_loss(X_val)
        train_reconstruction_loss, train_latent_loss = vader.reconstruction_loss[-1], vader.latent_loss[-1]
        test_reconstruction_loss, test_latent_loss = test_loss_dict["reconstruction_loss"], test_loss_dict[
            "latent_loss"]
        # noinspection PyTypeChecker
        effective_k = len(Counter(vader.cluster(X_train)))
        # noinspection PyTypeChecker
        y_pred = vader.cluster(X_val)

        # calculate total loss
        alpha = self.params_dict["alpha"]
        train_total_loss = train_reconstruction_loss + alpha * train_latent_loss
        test_total_loss = test_reconstruction_loss + alpha * test_latent_loss

        # calculate y_true
        vader = self._fit_vader(X_val, W_val)
        # noinspection PyTypeChecker
        y_true = vader.cluster(X_val)

        # evaluate clustering
        adj_rand_index = ClusteringMetrics.calc_adj_rand_index(y_pred, y_true)
        rand_index = ClusteringMetrics.calc_rand_index(y_pred, y_true)
        prediction_strength = ClusteringMetrics.calc_prediction_strength(y_pred, y_true)
        permuted_clustering_evaluation_metrics = ClusteringMetrics.calc_permuted_clustering_evaluation_metrics(
            y_pred, y_true, self.params_dict["n_perm"]
        )
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

    def _fit_vader(self, X_train: ndarray, W_train: Optional[ndarray]) -> VADER:
        k = self.params_dict["k"]
        n_hidden = self.params_dict["n_hidden"]
        learning_rate = self.params_dict["learning_rate"]
        batch_size = self.params_dict["batch_size"]
        alpha = self.params_dict["alpha"]

        # noinspection PyTypeChecker
        vader = VADER(X_train=X_train, W_train=W_train, save_path=None, n_hidden=n_hidden, k=k, seed=self.seed,
                      learning_rate=learning_rate, recurrent=True, batch_size=batch_size, alpha=alpha)

        vader.pre_fit(n_epoch=self.n_epoch, verbose=False)
        vader.fit(n_epoch=self.n_epoch, verbose=False)
        return vader
