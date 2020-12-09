import sys
import json
import random
import scipy.stats
import pandas as pd
import multiprocessing as mp
from numpy import ndarray
from vader.utils import read_adni_data
from typing import List, Dict, Tuple, Union, Optional
from .param_grid_factory import ParamGridFactory, ParamsGridType
from .step1_optimization_job import Step1OptimizationJob
from .step2_optimization_job import Step2OptimizationJob
from .constants import PARAMS_COLUMN_NAME


class VADERHyperparametersOptimizer:
    def __init__(self, param_grid_file: Optional[str], param_grid_factory: Optional[ParamGridFactory],
                 seed: Optional[int], n_repeats: int, n_proc: int, n_sample: int, output_model_path: str,
                 output_cv_path: str, verbose: bool = False):
        self.verbose = verbose
        self.n_sample = n_sample
        self.n_proc = n_proc
        self.n_repeats = n_repeats
        self.seed = seed
        self.output_model_path = output_model_path
        self.output_cv_path = output_cv_path
        self._step1_output_cv_path = output_cv_path + "_step1"
        self._step2_output_cv_path = output_cv_path + "_step2"
        self._step3_output_cv_path = output_cv_path
        self.input_param_grid = self.read_param_grid(param_grid_file) if param_grid_file else None
        self.param_grid_factory = param_grid_factory
        if self.verbose:
            print("=> init VADEROptimizer with", locals())

    @staticmethod
    def __read_param_grid(param_grid_file: str) -> ParamsGridType:
        with open(param_grid_file, 'r') as json_file:
            input_param_grid = json.load(json_file)
        return input_param_grid

    def run_all_steps(self, input_data_file: str, input_weights_file: str):
        """Main entry function that does all the job"""
        if self.verbose:
            print("=> start run_all_steps")

        # Stage 0: data loading
        input_data = read_adni_data(input_data_file)
        input_weights = read_adni_data(input_weights_file) if input_weights_file else None

        # Stage 1: non-variational AEs
        step1_results_df = self.optimization_step_1(input_data, input_weights)

        # Stage 2: optimize number of clusters 'k'
        step2_results_df = self.optimization_step_2(input_data, input_weights, step1_results_df)

        # Stage 3: aggregate the results, calculate wilcoxon's stats
        step3_results_df = self.aggregate_repetitions(step2_results_df)
        self.print_optimal_model(step3_results_df, "wilcoxon_prediction_strength")

        if self.verbose:
            print("<= finish run_all_steps")

    def run_certain_steps(self, input_data_file: str, input_weights_file: str, stage: str):
        """Secondary entry point for case when we need to re-run some certain steps"""
        if self.verbose:
            print("=> start run_certain_steps")

        # Stage 0: data loading
        if not stage or stage == "1" or stage == "2":
            input_data = read_adni_data(input_data_file)
            input_weights = read_adni_data(input_weights_file) if input_weights_file else None
        else:
            input_data = None
            input_weights = None

        # Stage 1: non-variational AEs
        if not stage or "1" in stage:
            step1_results_df = self._optimization_step_1(input_data, input_weights)
        elif "2" in stage:
            step1_results_df = pd.read_csv(self._step1_output_cv_path, index_col=0)
        else:
            step1_results_df = None

        # Stage 2: optimize number of clusters 'k'
        if not stage or "2" in stage:
            step2_results_df = self.optimization_step_2(input_data, input_weights, step1_results_df)
        elif "3" in stage:
            step2_results_df = pd.read_csv(self._step2_output_cv_path, index_col=0)
        else:
            step2_results_df = None

        # Stage 3: aggregate the results, calculate wilcoxon's stats
        if not stage or "3" in stage:
            step3_results_df = self.aggregate_repetitions(step2_results_df)
            print_optimal_model(step3_results_df, "wilcoxon_prediction_strength")

        if self.verbose:
            print("<= finish run_certain_steps")

    @staticmethod
    def run_cv_step1_job(params_tuple: tuple) -> pd.Series:
        return Step1OptimizationJob(*params_tuple).run()

    @staticmethod
    def run_cv_step2_job(params_tuple: tuple) -> pd.Series:
        return Step2OptimizationJob(*params_tuple).run()

    def optimization_step_1(self, input_data, input_weights):
        """The 1st step of optimization (using non-variational autoencoders)"""
        if not self.input_param_grid:
            self.input_param_grid = self.param_grid_factory.get_nonvar_param_grid()

        # randomize grid search
        if self.n_sample and self.n_sample < len(self.input_param_grid):
            param_grid = random.sample(self.input_param_grid, self.n_sample)
        else:
            param_grid = self.input_param_grid

        # cross-validation
        with mp.Pool(self.n_proc) as pool:
            cv_results_list = pool.map(self.run_cv_step1_job, [
                (input_data, input_weights, params_dict, self.seed, self.verbose)
                for params_dict in param_grid
                for _ in range(self.n_repeats)
            ])

        # output
        output_path = self._step1_output_cv_path
        cv_results_df = pd.DataFrame(cv_results_list)
        cv_results_df.to_csv(output_path)
        if self.verbose:
            print(f"=> optimization_step_1 finished. See: {output_path}")
        return cv_results_df

    def optimization_step_2(self, input_data, input_weights, step1_results_df: pd.DataFrame):
        # 2nd step (k-optimization) preparation
        best_hyperparameters = self._extract_best_parameters(step1_results_df)
        if self.verbose:
            print(f"BEST HYPERPARAMETERS ARE {best_hyperparameters.name} "
                  f"with loss={best_hyperparameters.test_reconstruction_loss}")
        best_hyperparameters_dict = self.param_grid_factory.generate_param_dict_for_k_optimization(best_hyperparameters)
        param_grid = ParamGridFactory.map_param_dict_to_param_grid(best_hyperparameters_dict)

        # 2nd step (k-optimization) cross-validation
        with mp.Pool(self.n_proc) as pool:
            cv_results_list = pool.map(self.run_cv_step2_job, [
                (input_data, input_weights, params_dict, self.seed, self.verbose)
                for params_dict in param_grid
                for _ in range(self.n_repeats)
            ])

        # 2nd step (k-optimization) output
        output_path = self._step2_output_cv_path
        cv_results_df = pd.DataFrame(cv_results_list)
        cv_results_df.to_csv(output_path)
        if self.verbose:
            print(f"=> run_hyperparameters_optimization - stage #2 finished. See: {output_path}")
        return cv_results_df

    @staticmethod
    def _extract_best_parameters(cv_results_df, minimize_metric: str = "test_reconstruction_loss"):
        cv_results_df_means = cv_results_df.groupby("params").mean()
        best_hyperparameters_index = cv_results_df_means[minimize_metric] == cv_results_df_means[minimize_metric].min()
        best_hyperparameters = cv_results_df_means[best_hyperparameters_index].iloc[0]
        return best_hyperparameters

    def aggregate_repetitions(self, step2_results_df: pd.DataFrame):
        RAND_INDEX_COLUMN_NAME = "rand_index"
        ADJ_RAND_INDEX_COLUMN_NAME = "adj_rand_index"
        PRED_STRENGTH_COLUMN_NAME = "prediction_strength"
        RAND_INDEX_NULL_COLUMN_NAME = f"{RAND_INDEX_COLUMN_NAME}_null"
        ADJ_RAND_INDEX_NULL_COLUMN_NAME = f"{ADJ_RAND_INDEX_COLUMN_NAME}_null"
        PRED_STRENGTH_NULL_COLUMN_NAME = f"{PRED_STRENGTH_COLUMN_NAME}_null"
        WILCOXON_RAND_INDEX_COLUMN_NAME = f"wilcoxon_{RAND_INDEX_COLUMN_NAME}"
        WILCOXON_ADJ_RAND_INDEX_COLUMN_NAME = f"wilcoxon_{ADJ_RAND_INDEX_COLUMN_NAME}"
        WILCOXON_PRED_STRENGTH_COLUMN_NAME = f"wilcoxon_{PRED_STRENGTH_COLUMN_NAME}"

        aggregated_df = step2_results_df.groupby(PARAMS_COLUMN_NAME).mean()
        try:
            aggregated_df[WILCOXON_RAND_INDEX_COLUMN_NAME] = step2_results_df.groupby(PARAMS_COLUMN_NAME).apply(
                lambda group: scipy.stats.wilcoxon(group[RAND_INDEX_COLUMN_NAME], group[RAND_INDEX_NULL_COLUMN_NAME])
            )
            aggregated_df[WILCOXON_ADJ_RAND_INDEX_COLUMN_NAME] = step2_results_df.groupby(PARAMS_COLUMN_NAME).apply(
                lambda group: scipy.stats.wilcoxon(group[ADJ_RAND_INDEX_COLUMN_NAME],
                                                   group[ADJ_RAND_INDEX_NULL_COLUMN_NAME])
            )
            aggregated_df[WILCOXON_PRED_STRENGTH_COLUMN_NAME] = step2_results_df.groupby(PARAMS_COLUMN_NAME).apply(
                lambda group: scipy.stats.wilcoxon(group[PRED_STRENGTH_COLUMN_NAME],
                                                   group[PRED_STRENGTH_NULL_COLUMN_NAME])
            )
        except ValueError:
            print(f"Unexpected error:", sys.exc_info()[1])

        # output
        output_path = self._step3_output_cv_path
        aggregated_df.to_csv(output_path)
        if self.verbose:
            print(f"=> run_hyperparameters_optimization - stage #3 finished. See: {output_path}")
        return aggregated_df

    @staticmethod
    def print_optimal_model(aggregated_df, performance_metric, minimize: bool = False):
        if not performance_metric in aggregated_df:
            return
        max_performance_metric_value = aggregated_df[performance_metric].min() if minimize \
            else aggregated_df[performance_metric].max()
        best_parameters_sets = aggregated_df[aggregated_df[performance_metric] == max_performance_metric_value]\
            .index.values
        print(f"{best_parameters_sets} with best {performance_metric}={max_performance_metric_value}")
