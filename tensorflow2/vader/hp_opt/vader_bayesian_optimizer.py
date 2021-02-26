import os
import shutil
import optuna
import traceback
import numpy as np
import pandas as pd
import multiprocessing as mp
from typing import List, Optional
from vader.hp_opt import common
from vader.hp_opt.job.full_optimization_job import FullOptimizationJob
from vader.hp_opt.cv_results_aggregator import CVResultsAggregator
from vader.hp_opt.interface.abstract_bayesian_params_factory import AbstractBayesianParamsFactory
from PyPDF2 import PdfFileMerger


class VADERBayesianOptimizer:
    """Handles the whole VaDER hyperparameters optimization process (bayesian)"""

    SECONDS_IN_DAY = 86400

    def __init__(self, params_factory: AbstractBayesianParamsFactory, n_repeats: int = 10, n_proc: int = 1,
                 n_trials: int = 100, n_consensus: int = 1, n_epoch: int = 10, n_splits: int = 2, n_perm: int = 100,
                 seed: Optional[int] = None, early_stopping_ratio: float = None, early_stopping_batch_size: int = 5,
                 enable_cv_loss_reports: bool = False, output_folder: str = "."):
        """
        Configure output folders, output file names, param grid and logging.

        Parameters
        ----------
        params_factory : AbstractBayesianParamsFactory
            Object that can produce a parameters grid.
        n_repeats : int
            Defines how many times we perform the optimization for the same set of hyperparameters.
            The higher this parameter - the better is optimization, but the worse is performance.
            Optimal values: 10-20.
            Default is 10.
        n_proc : int
            Defines how many processor units can be used to run optimization jobs.
            If the value is too big - maximum number of CPUs will be used.
            Since each jobs splits into some sub-processes too,
              a good approach will be to set n_proc to a maximum number of CPUs divided by 4.
            Default is 1 (no multi-processing).
        n_trials : int
            Defines how many sets of hyperparameters (excluding 'k'-s) we choose to evaluate.
            Each set of hyperparameters is chosen automatically according to the Bayesian optimization rules based on
              the performance of previous hyperparameters.
            The higher this parameter - the better is optimization, but the worse is performance.
            Optimal values: 100-200.
            Default is 100.
        n_consensus : int
            Defines how many times we train vader for each job for each data split.
            If n_consensus > 1, then it runs the "consensus clustering" algorithm
              to determine the final clustering.
            The higher this parameter - the better is optimization, but the worse is performance.
            Optimal values: 1-10.
            Default is 1 (no consensus clustering).
        n_epoch : int
            Defines how many epochs we train during the vader's "fit" step.
            The higher this parameter - the better is optimization, but the worse is performance.
            Optimal values: 10-50.
            Default is 10.
        early_stopping_ratio : float or None
            Defines the relative difference at which the model can stop fitting.
            Optimal value: 0.03 (which means that we stop fitting the model once loss changes less than 3% on average).
            Default is None (no early stopping).
        early_stopping_batch_size : int
            Defines how many epochs we use to calculate average relative difference in loss for early stopping criteria.
            When early_stopping_ratio is None, it does not have any effect.
            Default is 5.
        n_splits : int
            Defines into how many chunks we split the data for the cross-validation step.
            Increase this parameter for bigger data sets.
            Optimal values: 2-10.
            Default is 2.
        n_perm : int
            Defines how many times we permute each clustering during the calculation of the "prediction_strength_null".
            The higher this parameter - the better is optimization, but the worse is performance.
            Optimal values: 100-1000.
            Default is 100.
        seed : int or None
            Initializes the random number generator.
            It can be used to achieve reproducible results.
            If None - the random number generator will use its in-built initialization logic
                (e.g. using the current system time)
            Default is None.
        enable_cv_loss_reports : bool
            If true, produces intermittent reports showing loss changes over epochs during cross-validation.
            Default is False.
        output_folder : str
            Defines a folder where all outputs will be written.
            Outputs include:
              * final pdf report;
              * best hyperparameters for each 'k' with their scores;
              * "csv_repeats" folder with intermediate csv chunks;
              * "csv_trials" folder with intermediate csv chunks;
              * "failed_jobs" folder with stack-traces for all failed jobs;
              * logging file.
            Default: the current folder.
        """
        self.n_trials = n_trials
        self.n_proc = n_proc
        self.n_repeats = n_repeats
        self.n_consensus = n_consensus
        self.n_epoch = n_epoch
        self.n_splits = n_splits
        self.n_perm = n_perm
        self.seed = seed
        self.early_stopping_ratio = early_stopping_ratio
        self.early_stopping_batch_size = early_stopping_batch_size

        # Configure output folders
        self.output_folder = output_folder
        self.output_repeats_dir = os.path.join(self.output_folder, "csv_repeats")
        self.output_trials_dir = os.path.join(self.output_folder, "csv_trials")
        self.failed_jobs_dir = os.path.join(self.output_folder, "failed_jobs")
        if not os.path.exists(self.output_repeats_dir):
            os.makedirs(self.output_repeats_dir, exist_ok=True)
        if not os.path.exists(self.output_trials_dir):
            os.makedirs(self.output_trials_dir, exist_ok=True)
        if not os.path.exists(self.failed_jobs_dir):
            os.makedirs(self.failed_jobs_dir, exist_ok=True)

        self.cv_loss_reports_dir = None
        if enable_cv_loss_reports:
            self.cv_loss_reports_dir = os.path.join(self.output_folder, "cv_loss_reports")
            os.makedirs(self.cv_loss_reports_dir, exist_ok=True)

        # Configure param grid
        self.hyperparameters = ["n_hidden", "learning_rate", "batch_size", "alpha"]
        self.params_limits = params_factory.get_param_limits_dict()
        self.k_list = params_factory.get_k_list()

        # Configure output files names
        self.run_id = f"n_trials{n_trials}_n_repeats{n_repeats}_n_splits{n_splits}_" \
                      f"n_consensus{n_consensus}_n_epoch{n_epoch}_n_perm{n_perm}_seed{seed}"
        self.output_pdf_report_file = os.path.join(self.output_folder, f"report_{self.run_id}.pdf")
        self.output_diffs_file = os.path.join(self.output_folder, f"diffs_{self.run_id}.csv")
        self.output_best_scores_file = os.path.join(self.output_folder, f"best_scores_{self.run_id}.csv")
        self.output_log_file = os.path.join(self.output_folder, f"{__name__}_{self.run_id}.log")
        self.output_cv_loss_report_file = os.path.join(self.output_folder, f"cv_loss_report_{self.run_id}.pdf")

        # Configure logging
        self.logger = common.log_manager.get_logger(__name__, log_file=self.output_log_file)
        self.logger.info(f"{__name__} is initialized with run_id={self.run_id}")

    def __construct_jobs_params_list(self, input_data: np.ndarray, input_weights: np.ndarray) -> List[tuple]:
        jobs_params_list = [(input_data, input_weights, k) for k in self.k_list]
        return jobs_params_list

    def run_parallel_jobs(self, jobs_params_list: List[tuple]) -> pd.DataFrame:
        if self.n_proc > len(jobs_params_list):
            n_proc = len(jobs_params_list)
        else:
            n_proc = self.n_proc
        with mp.Pool(n_proc) as pool:
            cv_results_list = pool.map(self.run_cv_full_job, jobs_params_list)

        cv_results_df = pd.DataFrame(cv_results_list)
        return cv_results_df

    def run_cv_full_job(self, params_tuple: tuple) -> pd.Series:
        input_data = params_tuple[0]
        input_weights = params_tuple[1]
        k = params_tuple[2]

        self.logger.info(f"PROCESS k={k}")
        study_name = f'VaDER_k{k}'
        study = optuna.create_study(
            study_name=study_name,
            # storage=f"sqlite:///{study_name}.db",
            direction="maximize",
            load_if_exists=True
        )
        study.optimize(
            func=lambda trial: self.objective(trial, k, input_data, input_weights),
            n_trials=self.n_trials,
            timeout=self.SECONDS_IN_DAY,
            n_jobs=self.n_proc
        )
        result = {
            "k": k,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "best_trial": study.best_trial.number
        }
        self.logger.info(f"For k={k} best_params={study.best_params} with score={study.best_value}")
        return pd.Series(result)

    def __gen_repeats_files_from_trials_files(self, cv_results_df):
        df_trials_list = []
        for i, row in cv_results_df.iterrows():
            k = row["k"]
            trial = row["best_trial"]
            trial_csv_file = f"k{k}_trial{trial}.csv"
            trial_df = pd.read_csv(os.path.join(self.output_trials_dir, trial_csv_file))
            # Since hyperparameters are different for different 'k's - we need to mask them to support join
            for hp in self.hyperparameters:
                trial_df[hp] = "~"
            df_trials_list.append(trial_df)
        df = pd.concat(df_trials_list, ignore_index=True)

        for i in range(self.n_repeats):
            one_repeat_index = list(range(i, df.shape[0], self.n_repeats))
            df.iloc[one_repeat_index].to_csv(os.path.join(self.output_repeats_dir, f"repeat_{i}.csv"), index=False)

    def __gen_cv_loss_report(self):
        pdf_files = [entry.path for entry in os.scandir(self.cv_loss_reports_dir) if entry.is_file() and entry.path.endswith(".pdf")]
        merger = PdfFileMerger()
        for pdf in pdf_files:
            merger.append(pdf)
        merger.write(self.output_cv_loss_report_file)
        merger.close()
        shutil.rmtree(self.cv_loss_reports_dir)

    def run(self, input_data: np.ndarray, input_weights: np.ndarray) -> None:
        self.logger.info(f"Optimization has started. Data shape: {input_data.shape}")

        jobs_params_list = self.__construct_jobs_params_list(input_data, input_weights)
        self.logger.info(f"Number of jobs: {len(jobs_params_list)}")

        cv_results_df = self.run_parallel_jobs(jobs_params_list)
        cv_results_df.to_csv(self.output_best_scores_file, index=False)

        self.__gen_repeats_files_from_trials_files(cv_results_df)
        aggregator = CVResultsAggregator.from_files(self.output_repeats_dir, self.hyperparameters)
        aggregator.plot_to_pdf(self.output_pdf_report_file)

        if self.cv_loss_reports_dir:
            self.__gen_cv_loss_report()

        self.logger.info(f"Optimization has finished. See: {self.output_best_scores_file}")

        number_of_failed_jobs = len(os.listdir(self.failed_jobs_dir))
        if number_of_failed_jobs > 0:
            self.logger.warning(f"There are {number_of_failed_jobs} failed jobs. See: {self.failed_jobs_dir}")

    def run_cv_single_job(self, input_data, input_weights, params_dict, seed):
        job = FullOptimizationJob(
            data=input_data,
            weights=input_weights,
            params_dict=params_dict,
            seed=seed,
            n_consensus=self.n_consensus,
            n_epoch=self.n_epoch,
            early_stopping_ratio=self.early_stopping_ratio,
            early_stopping_batch_size=self.early_stopping_batch_size,
            n_splits=self.n_splits,
            n_perm=self.n_perm,
            reports_dir=self.cv_loss_reports_dir
        )
        # noinspection PyBroadException
        try:
            self.logger.info(f"Job has started with id={job.cv_id} and job_params_dict={params_dict}")
            result = job.run()
            self.logger.info(f"Job has finished with id={job.cv_id} and job_params_dict={params_dict}")
        except Exception:
            error_message = f"Job failed: {job.cv_id} and job_params_dict={params_dict}\n{traceback.format_exc()}"
            log_file = os.path.join(self.failed_jobs_dir, f"{job.cv_id}.log")
            with open(log_file, "w") as f:
                f.write(error_message)
            self.logger.error(error_message)
            result = pd.Series(params_dict)
        return result

    def objective(self, trial, k, input_data, input_weights):
        trial_id = f"k{k}_trial{trial.number}"
        # learning_rate = trial.suggest_loguniform("learning_rate", 1e-4, 1e-2)
        alpha = trial.suggest_float(
            "alpha",
            self.params_limits["alpha"][0],
            self.params_limits["alpha"][1]
        )
        learning_rate = trial.suggest_float(
            "learning_rate",
            self.params_limits["learning_rate"][0],
            self.params_limits["learning_rate"][1],
            log=True
        )
        batch_size = trial.suggest_int(
            "batch_size",
            self.params_limits["batch_size"][0],
            self.params_limits["batch_size"][1]
        )
        n_hidden_layers = trial.suggest_int(
            "n_hidden_layers",
            self.params_limits["n_hidden_layers"][0],
            self.params_limits["n_hidden_layers"][1]
        )
        n_hidden = []
        for i in range(1, n_hidden_layers + 1):
            max_size = max(self.params_limits["hidden_layer_size"][1], n_hidden[-1]) if n_hidden else self.params_limits["hidden_layer_size"][1]
            n_hidden_i = trial.suggest_int(
                f"n_hidden_{i}",
                self.params_limits["hidden_layer_size"][0],
                max_size
            )
            n_hidden.append(n_hidden_i)

        params_dict = {
            "k": k,
            "n_hidden": n_hidden,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "alpha": alpha
        }
        self.logger.info(f"New trial {trial_id} with params={params_dict}")

        repeats_results = []
        for i in range(self.n_repeats):
            seed = int(str(self.seed) + str(k) + str(trial.number) + str(i)) if self.seed else None
            result = self.run_cv_single_job(input_data, input_weights, params_dict, seed)
            repeats_results.append(result)
        results_df = pd.DataFrame(repeats_results)
        results_df.to_csv(os.path.join(self.output_trials_dir, f"{trial_id}.csv"), index=False)
        score = results_df["prediction_strength_diff"].mean() if "prediction_strength_diff" in results_df.columns else None
        return score
