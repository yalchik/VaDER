from numpy import ndarray
from typing import Dict, Union, Optional
from vader.hp_opt.job.abstract_optimization_job import AbstractOptimizationJob
from vader import VADER


class PreFitOptimizationJob(AbstractOptimizationJob):
    """
    Hyperparameter optimization using non-variational AEs
    It can be used to speed up the selection of non-'k' hyperparameters.
    """

    def _cv_fold_step(self, X_train: ndarray, X_val: ndarray, W_train: Optional[ndarray],
                      W_val: Optional[ndarray], split_id: int = None) -> Dict[str, Union[int, float]]:
        vader = self._fit_vader(X_train, W_train)
        # noinspection PyTypeChecker
        test_loss_dict = vader.get_loss(X_val)
        train_reconstruction_loss, train_latent_loss = vader.reconstruction_loss[-1], vader.latent_loss[-1]
        test_reconstruction_loss, test_latent_loss = test_loss_dict["reconstruction_loss"], test_loss_dict[
            "latent_loss"]

        return {
            "train_reconstruction_loss": train_reconstruction_loss,
            "train_latent_loss": train_latent_loss,
            "test_reconstruction_loss": test_reconstruction_loss,
            "test_latent_loss": test_latent_loss
        }

    def _fit_vader(self, X_train: ndarray, W_train: Optional[ndarray]) -> VADER:
        k = 1
        n_hidden = self.params_dict["n_hidden"]
        learning_rate = self.params_dict["learning_rate"]
        batch_size = self.params_dict["batch_size"]
        alpha = 0

        # noinspection PyTypeChecker
        vader = VADER(X_train=X_train, W_train=W_train, save_path=None, n_hidden=n_hidden, k=k, seed=self.seed,
                      learning_rate=learning_rate, recurrent=True, batch_size=batch_size, alpha=alpha)

        vader.pre_fit(n_epoch=self.n_epoch, verbose=False)
        return vader
