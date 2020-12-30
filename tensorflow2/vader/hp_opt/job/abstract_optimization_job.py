import uuid
import pandas as pd
from numpy import ndarray
from abc import ABC, abstractmethod
from vader import VADER
from vader.hp_opt import common
from typing import List, Dict, Union, Optional
from sklearn.model_selection import KFold


class AbstractOptimizationJob(ABC):
    def __init__(self, data: ndarray, weights: ndarray, params_dict: common.ParamsDictType, seed: int,
                 n_consensus: int, n_epoch: int, n_splits: int, n_perm: int):
        self.data = data
        self.weights = weights
        self.params_dict = params_dict
        self.seed = seed
        self.n_consensus = n_consensus
        self.n_epoch = n_epoch
        self.n_splits = n_splits
        self.n_perm = n_perm
        self.cv_id = uuid.uuid4()
        self.logger = common.log_manager.get_logger(__name__)

    @abstractmethod
    def _cv_fold_step(self, X_train: ndarray, X_val: ndarray, W_train: Optional[ndarray],
                      W_val: Optional[ndarray]) -> Dict[str, Union[int, float]]:
        pass

    @abstractmethod
    def _fit_vader(self, X_train: ndarray, W_train: Optional[ndarray]) -> VADER:
        pass

    def run(self) -> pd.Series:
        self.logger.debug(f"=> optimization_job started id={self.cv_id}")
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
        self.logger.debug(f"<= optimization_job finished id={self.cv_id}")
        return cv_result_series
