import uuid
import pandas as pd
from numpy import ndarray
from abc import ABC, abstractmethod
from .constants import ParamsDictType, PARAMS_COLUMN_NAME, logger
from typing import List, Dict, Union, Optional
from sklearn.model_selection import KFold


class AbstractOptimizationJob(ABC):
    def __init__(self, data: ndarray, weights: ndarray, params_dict: ParamsDictType, seed: int, verbose: bool = False):
        self.params_dict = params_dict
        self.verbose = verbose
        self.seed = seed
        self.weights = weights
        self.data = data
        self.cv_id = uuid.uuid4()

        self.n_splits = params_dict["n_splits"]
        self.n_epoch = params_dict["n_epoch"]

    @abstractmethod
    def _cv_fold_step(self, X_train: ndarray, X_val: ndarray, W_train: Optional[ndarray],
                      W_val: Optional[ndarray]) -> Dict[str, Union[int, float]]:
        pass

    @abstractmethod
    def _fit_vader(self, X_train: ndarray, W_train: Optional[ndarray]):
        pass

    def run(self) -> pd.Series:
        if self.verbose:
            logger.info(f"=> start run_cv id={self.cv_id} with params_dict={self.params_dict}")

        cv_folds_results_list = []
        data_split = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.seed).split(self.data)
        for train_index, val_index in data_split:
            X_train, X_val = self.data[train_index], self.data[val_index]
            W_train, W_val = (self.weights[train_index], self.weights[val_index]) if self.weights else None, None
            cv_fold_result = self._cv_fold_step(X_train, X_val, W_train, W_val)
            cv_folds_results_list.append(cv_fold_result)

        cv_folds_results_df = pd.DataFrame(cv_folds_results_list)
        cv_mean_results_series = cv_folds_results_df.mean()
        cv_mean_results_series[PARAMS_COLUMN_NAME] = str(self.params_dict)
        if self.verbose:
            logger.info(f"<= finish run_cv id={self.cv_id} with params_dict={self.params_dict}")
        return cv_mean_results_series