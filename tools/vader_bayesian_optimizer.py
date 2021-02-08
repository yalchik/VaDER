import os
import traceback
import numpy as np
import pandas as pd
import multiprocessing as mp
from vader.hp_opt import common
from typing import List, Optional
from vader.hp_opt.param_grid_factory import ParamGridFactory
from vader.hp_opt.job.full_optimization_job import FullOptimizationJob
from vader.hp_opt.cv_results_aggregator import CVResultsAggregator
import uuid
import pandas as pd
from numpy import ndarray
from abc import ABC, abstractmethod
from vader import VADER
from vader.hp_opt import common
from typing import List, Dict, Union, Optional
from sklearn.model_selection import KFold
import numpy as np
from numpy import ndarray
from collections import Counter
from typing import Dict, Union, Optional
from vader import VADER
from vader.hp_opt.job.abstract_optimization_job import AbstractOptimizationJob
from vader.utils.clustering_utils import ClusteringUtils
from vader.utils.data_utils import generate_x_w_y, read_adni_norm_data, generate_wtensor_from_xtensor
import optuna


class VADERBayesianOptimizer:
    def __init__(self, param_grid_factory: Optional[ParamGridFactory] = None, n_repeats: int = 10, n_proc: int = 1,
                 n_sample: int = None, n_consensus: int = 1, n_epoch: int = 10, n_splits: int = 2, n_perm: int = 100,
                 seed: Optional[int] = None, output_folder: str = "."):
        self.logger = common.log_manager.get_logger(__name__)

    def run(self, input_data: np.ndarray, input_weights: np.ndarray) -> None:
        results = {}

        for k in range(2, 7):
            study = optuna.create_study(
                study_name=f'VaDER_k{k}',
                # storage='sqlite:///hypopt/optuna__coxtest_01.db',
                direction="maximize",
                load_if_exists=True
            )

            def objective(trial):
                learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1)
                batch_size = trial.suggest_int("batch_size", 8, 128)
                n_hidden_1 = trial.suggest_int("n_hidden_1", 8, 128)
                n_hidden_2 = trial.suggest_int("n_hidden_2", 1, n_hidden_1)
                n_hidden = (n_hidden_1, n_hidden_2)

                params_dict = {
                    "k": k,
                    "n_hidden": n_hidden,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "alpha": 1
                }
                job = FullOptimizationJob(
                    data=input_data,
                    weights=input_weights,
                    params_dict=params_dict,
                    seed=None,
                    n_consensus=1,
                    n_epoch=50,
                    n_splits=2,
                    n_perm=10
                )
                try:
                    self.logger.info(f"Job has started with id={job.cv_id} and job_params_dict={params_dict}")
                    result = job.run()
                    score = result["prediction_strength"] - result["prediction_strength_null"]
                    self.logger.info(f"Job has finished with id={job.cv_id} and job_params_dict={params_dict}")
                except Exception:
                    error_message = f"Job failed: {job.cv_id} and job_params_dict={params_dict}\n{traceback.format_exc()}"
                    # log_file = os.path.join(self.failed_jobs_dir, f"{job.cv_id}.log")
                    # with open(log_file, "w") as f:
                    #     f.write(error_message)
                    self.logger.error(error_message)
                    score = None
                return score

            self.logger.info(f"PROCESS k={k}")
            study.optimize(
                func=objective,
                n_trials=100,
                timeout=3600,
                n_jobs=6
            )
            self.logger.info(f"FOR k={k} best_params={study.best_params} with best_value={study.best_value} and best_trial={study.best_trial}")
            results[k] = {
                "best_params": study.best_params,
                "best_value": study.best_value,
                "best_trial": study.best_trial
            }
        self.logger.info(f"FINAL REPORT: {results}")


if __name__ == "__main__":
    x_tensor_with_nans = read_adni_norm_data("d:\\workspaces\\vader_data\\ADNI\\Xnorm.csv")
    W = generate_wtensor_from_xtensor(x_tensor_with_nans)
    X = np.nan_to_num(x_tensor_with_nans)
    VADERBayesianOptimizer().run(X, W)
