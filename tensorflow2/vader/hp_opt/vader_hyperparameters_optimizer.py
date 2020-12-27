import sys
import json
import random
import traceback
import os
import os.path
from time import time
import logging
from logging.handlers import RotatingFileHandler
import scipy.stats
import numpy as np
import pandas as pd
import multiprocessing as mp
from numpy import ndarray
from typing import List, Dict, Tuple, Union, Optional
from .param_grid_factory import ParamGridFactory, ParamsGridType
from .pre_fit_optimization_job import PreFitOptimizationJob
from .full_optimization_job import FullOptimizationJob
from .setup import logger, formatter
from .cv_results_aggregator import CVResultsAggregator


class VADERHyperparametersOptimizer:
    def __init__(self, param_grid_factory: Optional[ParamGridFactory],
                 seed: Optional[int], n_repeats: int, n_proc: int, n_sample: int, n_consensus: int, n_epoch: int,
                 n_splits: int, n_perm: int, output_cv_path: str, verbose: bool = False,
                 log_dir: str = None):
        self.verbose = verbose
        self.n_sample = n_sample
        self.n_proc = n_proc
        self.n_repeats = n_repeats
        self.n_consensus = n_consensus
        self.n_epoch = n_epoch
        self.n_splits = n_splits
        self.n_perm = n_perm
        self.seed = seed
        self.output_cv_path = output_cv_path
        self.hyperparameters = [key for key in param_grid_factory.get_full_param_dict().keys() if key != "k"]
        self.param_grid = param_grid_factory.get_randomized_param_grid(n_sample)
        self.log_dir = log_dir
        if log_dir:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
            log_file = os.path.join(log_dir, "vader_hyperparameters_optimizer.log")
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.DEBUG)
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        if self.verbose:
            logger.info("init VADEROptimizer with", locals())

    def run(self, input_data, input_weights):
        """Main entry function that does all the job"""
        if self.verbose:
            logger.info("start run_all_steps")

        output_csv_path = os.path.join(self.output_cv_path, "csv")
        _ = self.optimization_step_full(input_data, input_weights, output_csv_path)
        aggregator = CVResultsAggregator.from_files(output_csv_path, self.hyperparameters)
        aggregator.plot_to_pdf(os.path.join(self.output_cv_path, "grid_search_report.pdf"))
        aggregator.save_to_csv(os.path.join(self.output_cv_path, "grid_search_diffs.csv"))

        if self.verbose:
            logger.info("finish run_all_steps")

    def optimization_step_full(self, input_data, input_weights, output_folder):
        if self.verbose:
            number_of_jobs = self.n_repeats * len(self.param_grid)
            logger.info(f"Number of full jobs: {number_of_jobs}")

        # construct jobs parameters
        jobs_params_list = []
        for i in range(self.n_repeats):
            for j, params_dict in enumerate(self.param_grid):
                seed = int(str(self.seed) + str(j) + str(i)) if self.seed else None
                jobs_params_list.append(
                    (input_data, input_weights, params_dict, seed,
                     self.n_consensus, self.n_epoch, self.n_splits, self.n_perm, self.verbose)
                )

        # run jobs in parallel
        with mp.Pool(self.n_proc) as pool:
            cv_results_list = pool.map(self.run_cv_full_job, jobs_params_list)

        # output
        cv_results_df = pd.DataFrame(cv_results_list)
        cv_results_df_list = np.array_split(cv_results_df, self.n_repeats)
        for i in range(self.n_repeats):
            output_file = os.path.join(output_folder, f"grid_search_{i}.csv")
            cv_results_df_list[i].to_csv(output_file, index=False)
        if self.verbose:
            logger.info(f"optimization_step_full finished. See: {self.output_cv_path}")
        return cv_results_df

    def run_cv_full_job(self, params_tuple: tuple) -> pd.Series:
        job = FullOptimizationJob(*params_tuple)
        try:
            result = job.run()
        except Exception as err:
            if self.log_dir:
                log_path = os.path.join(self.log_dir, f"{job.cv_id}.log")
                with open(log_path, "w") as f:
                    f.write(f"Job failed: {job.cv_id} with err={err}, Traceback: {traceback.format_exc()}")
            result = pd.Series(params_tuple[2])
        return result
