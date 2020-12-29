import os
import logging
import traceback
import numpy as np
import pandas as pd
import multiprocessing as mp
from vader.hp_opt import common
from typing import List, Dict, Tuple, Union, Optional
from .param_grid_factory import ParamGridFactory, ParamsGridType
from .full_optimization_job import FullOptimizationJob
from .cv_results_aggregator import CVResultsAggregator


class VADERHyperparametersOptimizer:

    def __init__(self, param_grid_factory: Optional[ParamGridFactory] = None, seed: Optional[int] = None,
                 n_repeats: int = 10, n_proc: int = 1, n_sample: int = None, n_consensus: int = 1,
                 n_epoch: int = 10, n_splits: int = 2, n_perm: int = 100, output_folder: str = ".",
                 verbose: bool = False):
        self.verbose = verbose
        self.n_sample = n_sample
        self.n_proc = n_proc
        self.n_repeats = n_repeats
        self.n_consensus = n_consensus
        self.n_epoch = n_epoch
        self.n_splits = n_splits
        self.n_perm = n_perm
        self.seed = seed

        # Configure output folders
        self.output_folder = output_folder
        self.output_repeats_dir = os.path.join(self.output_folder, "csv_repeats")
        self.failed_jobs_dir = os.path.join(self.output_folder, "failed_jobs")
        if not os.path.exists(self.output_repeats_dir):
            os.makedirs(self.output_repeats_dir)
        if not os.path.exists(self.failed_jobs_dir):
            os.makedirs(self.failed_jobs_dir)

        # Configure param grid
        if not param_grid_factory:
            param_grid_factory = ParamGridFactory()
        self.hyperparameters = [key for key in param_grid_factory.get_full_param_dict().keys() if key != "k"]
        self.param_grid = param_grid_factory.get_randomized_param_grid(n_sample)

        # Configure output files names
        self.run_id = f"n_grid{len(self.param_grid)}_n_sample{n_sample}_n_repeats{n_repeats}_n_splits{n_splits}_" \
                      f"n_consensus{n_consensus}_n_epoch{n_epoch}_n_perm{n_perm}_seed{seed}"
        self.output_pdf_report_file = os.path.join(self.output_folder, f"report_{self.run_id}.pdf")
        self.output_diffs_file = os.path.join(self.output_folder, f"diffs_{self.run_id}.csv")
        self.output_all_repeats_file = os.path.join(self.output_folder, f"all_repeats_{self.run_id}.csv")
        self.output_log_file = os.path.join(self.output_folder, f"{__name__}_{self.run_id}.log")

        # Configure logging
        self.logger = common.log_manager.get_logger(__name__, log_file=self.output_log_file)
        self.logger.info(f"{__name__} is initialized with run_id={self.run_id}")

    def run(self, input_data: np.ndarray, input_weights: np.ndarray) -> None:
        """Main entry function that does all the job"""
        self.logger.info(f"Optimization has started. Data shape: {input_data.shape}")

        jobs_params_list = self.__construct_jobs_params_list(input_data, input_weights)
        self.logger.info(f"Number of jobs: {len(jobs_params_list)}")

        cv_results_df = self.run_parallel_jobs(jobs_params_list)
        self.__save_all_repeats(cv_results_df)
        self.__save_split_repeats(cv_results_df)

        # TODO: refactoring: aggregate directly from cv_results_df instead of files manipulations
        aggregator = CVResultsAggregator.from_files(self.output_repeats_dir, self.hyperparameters)
        aggregator.plot_to_pdf(self.output_pdf_report_file)
        aggregator.save_to_csv(self.output_diffs_file)

        self.logger.info(f"Optimization has finished. See: {self.output_pdf_report_file}")

        number_of_failed_jobs = len(os.listdir(self.failed_jobs_dir))
        if number_of_failed_jobs > 0:
            self.logger.warning(f"There are {number_of_failed_jobs} failed jobs. See: {self.failed_jobs_dir}")

    def run_parallel_jobs(self, jobs_params_list: List[tuple]) -> pd.DataFrame:
        with mp.Pool(self.n_proc) as pool:
            cv_results_list = pool.map(self.run_cv_full_job, jobs_params_list)

        cv_results_df = pd.DataFrame(cv_results_list)
        return cv_results_df

    def run_cv_full_job(self, params_tuple: tuple) -> pd.Series:
        job = FullOptimizationJob(*params_tuple)
        try:
            self.logger.info(f"Job has started with id={job.cv_id} and params_tuple={params_tuple}")
            result = job.run()
            self.logger.info(f"Job has finished with id={job.cv_id} and params_tuple={params_tuple}")
        except Exception as err:
            error_message = f"Job failed: {job.cv_id} with err={err}, Traceback: {traceback.format_exc()}"
            log_file = os.path.join(self.failed_jobs_dir, f"{job.cv_id}.log")
            with open(log_file, "w") as f:
                f.write(error_message)
            self.logger.error(error_message)
            result = pd.Series(params_tuple[2])
        return result

    def __construct_jobs_params_list(self, input_data: np.ndarray, input_weights: np.ndarray) -> List[tuple]:
        jobs_params_list = []
        for i in range(self.n_repeats):
            for j, params_dict in enumerate(self.param_grid):
                seed = int(str(self.seed) + str(j) + str(i)) if self.seed else None
                jobs_params_list.append(
                    (input_data, input_weights, params_dict, seed,
                     self.n_consensus, self.n_epoch, self.n_splits, self.n_perm)
                )
        return jobs_params_list

    def __save_all_repeats(self, cv_results_df: pd.DataFrame) -> None:
        cv_results_df.to_csv(self.output_all_repeats_file, index=False)

    def __save_split_repeats(self, cv_results_df: pd.DataFrame) -> None:
        cv_results_df_list = np.array_split(cv_results_df, self.n_repeats)
        for i in range(self.n_repeats):
            cv_results_df_list[i].to_csv(os.path.join(self.output_repeats_dir, f"repeat_{i}.csv"), index=False)
