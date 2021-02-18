import uuid
import pandas as pd
from numpy import ndarray
from abc import ABC, abstractmethod
from vader import VADER
from vader.hp_opt import common
from typing import List, Dict, Union, Optional
from sklearn.model_selection import KFold


class AbstractOptimizationJob(ABC):
    """Holds cross-validation logic."""

    def __init__(self, data: ndarray, weights: ndarray, params_dict: common.ParamsDictType, seed: int,
                 n_consensus: int, n_epoch: int, n_splits: int, n_perm: int, early_stopping_ratio: float = None,
                 early_stopping_batch_size: int = 5):
        self.data = data
        self.weights = weights
        self.params_dict = params_dict
        self.seed = seed
        self.n_consensus = n_consensus
        self.n_epoch = n_epoch
        self.n_splits = n_splits
        self.n_perm = n_perm
        self.early_stopping_ratio = early_stopping_ratio
        self.early_stopping_batch_size = early_stopping_batch_size
        self.cv_id = uuid.uuid4()
        self.logger = common.log_manager.get_logger(__name__)
        self.logger.info(f"Job is initialized with id={self.cv_id}, seed={seed}, n_consensus={n_consensus},"
                         f" n_epoch={n_epoch}, n_splits={n_splits}, n_perm={n_perm}, params_dict={params_dict}")

    @abstractmethod
    def _cv_fold_step(self, X_train: ndarray, X_val: ndarray, W_train: Optional[ndarray],
                      W_val: Optional[ndarray]) -> Dict[str, Union[int, float]]:
        """
        Processes a single data fold when input data has been already split into a train and a validation subsets.

        Parameters
        ----------
        X_train : tensor
        X_val : tensor
        W_train : tensor
        W_val : tensor

        Returns
        -------
        Dictionary mapping performance metrics names to their values.
        """
        pass

    @abstractmethod
    def _fit_vader(self, X_train: ndarray, W_train: Optional[ndarray]) -> VADER:
        """
        Fits VaDER model with given data and weights tensors.

        Parameters
        ----------
        X_train : tensor
            1st dimension is samples,
            2nd dimension is time points,
            3rd dimension is feature vectors.
        W_train : tensor
            Has the same structure as input_data tensor, but defines if the corresponding value in X tensor is missing.
            If None, VaDER will use its own logic to calculate weights from the X_train data.

        Returns
        -------
        VaDER object with calculated weights, that can be used to cluster data.
        """
        pass

    def run(self) -> pd.Series:
        """
        Main entry point of the job. Handles cross-validation process.
        Splits data into training and validation data subsets, calls an abstract method to measure the performance
          for each fold and aggregates the results in the end.

        Returns
        -------
        Pandas Series with evaluation metrics values for a certain set of hyperparameters.
        """
        # self.logger.debug(f"=> optimization_job started id={self.cv_id}")
        cv_folds_results_list = []
        data_split = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed).split(self.data)
        for train_index, val_index in data_split:
            X_train, X_val = self.data[train_index], self.data[val_index]
            if self.weights is not None:
                W_train, W_val = self.weights[train_index], self.weights[val_index]
            else:
                W_train, W_val = None, None
            cv_fold_result = self._cv_fold_step(X_train, X_val, W_train, W_val)
            cv_folds_results_list.append(cv_fold_result)

        cv_folds_results_df = pd.DataFrame(cv_folds_results_list)
        cv_mean_results_series = cv_folds_results_df.mean()
        cv_params_series = pd.Series(self.params_dict)
        cv_result_series = cv_params_series.append(cv_mean_results_series)
        # self.logger.debug(f"<= optimization_job finished id={self.cv_id}")
        return cv_result_series
