import os
import math
import pandas as pd
import numpy as np
import scipy.stats
import warnings
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.lines import Line2D
from vader.utils.clustering_utils import ClusteringUtils
from typing import List, Type, TypeVar, Tuple

CVResultsAggregatorType = TypeVar('CVResultsAggregatorType', bound='CVResultsAggregator')


class CVResultsAggregator:

    def __init__(self):
        self.df_params:              Optional[pd.DataFrame] = None
        self.df_eff_k:               Optional[pd.DataFrame] = None
        self.df_pred_str:            Optional[pd.DataFrame] = None
        self.df_pred_str_null:       Optional[pd.DataFrame] = None
        self.num_of_repetitions:     Optional[int] = None
        self.num_of_hp_sets:         Optional[int] = None
        self.repetitions_matrix:     Optional[np.ndarray] = None
        self.diff_df:                Optional[pd.DataFrame] = None
        self.pval_df:                Optional[pd.DataFrame] = None
        self.diff_ord_series_sorted: Optional[pd.Series] = None
        self.perf_df:                Optional[pd.DataFrame] = None

    def plot_to_pdf(self, output_file: str) -> None:
        pdf = matplotlib.backends.backend_pdf.PdfPages(output_file)
        for i in self.diff_ord_series_sorted.index:
            fig = self.__plot_one(i)
            pdf.savefig(fig)
        pdf.close()

    def save_to_csv(self, output_file: str) -> None:
        self.perf_df.to_csv(output_file, index=False)

    def __plot_one(self, index: int) -> matplotlib.figure.Figure:
        rows_set = self.repetitions_matrix[index]
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"#{index}: {self.df_params.loc[index].to_dict()}")
        self.plot_1_1(axs[0, 0], self.df_pred_str, self.df_pred_str_null, rows_set)
        self.plot_1_2(axs[0, 1], self.diff_df, index)
        self.plot_2_1(axs[1, 0], self.pval_df, index)
        self.plot_2_2(axs[1, 1], self.df_eff_k, rows_set)
        return fig

    @classmethod
    def from_files(cls: Type[CVResultsAggregatorType], input_dir: str, params: List[str]) -> CVResultsAggregatorType:
        obj = cls()
        obj.df_params, obj.df_eff_k, obj.df_pred_str, obj.df_pred_str_null, obj.num_of_repetitions = cls.read_data(input_dir, params)
        obj.num_of_hp_sets = len(obj.df_eff_k) // obj.num_of_repetitions
        obj.repetitions_matrix = np.stack(
            np.split(np.arange(obj.num_of_hp_sets * obj.num_of_repetitions), obj.num_of_repetitions)
        ).transpose()
        obj.diff_df = cls.calc_diff(obj.df_pred_str, obj.df_pred_str_null, obj.repetitions_matrix)
        obj.pval_df = cls.calc_pval(obj.df_pred_str, obj.df_pred_str_null, obj.repetitions_matrix)
        obj.diff_ord_series_sorted = obj.diff_df.median(axis=1).sort_values(ascending=False)
        obj.perf_df = obj.df_params.loc[obj.diff_ord_series_sorted.index].join(obj.diff_df)
        return obj

    @staticmethod
    def read_data(input_folder: str, params: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, int]:
        df_eff_k_list = []
        df_pred_str_list = []
        df_pred_str_null_list = []
        for entry in os.scandir(input_folder):
            if entry.is_file() and entry.path.endswith(".csv"):
                df = pd.read_csv(entry.path)
                df_eff_k_list.append(
                    df.pivot(values='effective_k', index=params, columns="k").reset_index()
                )
                df_pred_str_list.append(
                    df.pivot(values='prediction_strength', index=params, columns="k").reset_index()
                )
                df_pred_str_null_list.append(
                    df.pivot(values='prediction_strength_null', index=params, columns="k").reset_index()
                )
        df_eff_k = pd.concat(df_eff_k_list, ignore_index=True)
        df_pred_str = pd.concat(df_pred_str_list, ignore_index=True)
        df_pred_str_null = pd.concat(df_pred_str_null_list, ignore_index=True)
        df_params = df_eff_k[params]
        df_eff_k_clean = df_eff_k.drop(columns=df_params.columns)
        df_eff_pred_str_clean = df_pred_str.drop(columns=df_params.columns)
        df_eff_pred_str_null_clean = df_pred_str_null.drop(columns=df_params.columns)
        num_of_files = len(df_eff_k_list)
        return df_params, df_eff_k_clean, df_eff_pred_str_clean, df_eff_pred_str_null_clean, num_of_files

    @staticmethod
    def calc_diff_row(p: pd.DataFrame, q: pd.DataFrame) -> pd.Series:
        pq_diff = (p - q)  # .applymap(lambda val: val if np.abs(val) > eps else None)
        col_means = pq_diff.mean()
        col_sd_diffs = ClusteringUtils.std_diff(pq_diff)
        col_sums = np.sqrt(p.notna().sum())
        diff = col_means / col_sd_diffs * col_sums
        return diff

    @staticmethod
    def calc_diff(df_pred_str: pd.DataFrame, df_pred_str_null: pd.DataFrame, repetitions_matrix: np.ndarray) -> pd.DataFrame:
        diff_series_list = [
            CVResultsAggregator.calc_diff_row(df_pred_str.iloc[i], df_pred_str_null.iloc[i]) for i in repetitions_matrix
        ]
        diff_df = pd.DataFrame(diff_series_list).fillna(0)
        return diff_df

    @staticmethod
    def calc_pval_row(p: pd.DataFrame, q: pd.DataFrame) -> pd.Series:
        pval_list = []
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', "Sample size too small for normal approximation.")
            for col in p.columns:
                try:
                    stats = scipy.stats.wilcoxon(p[col], q[col], mode="approx", correction=True)
                except ValueError:
                    stats = None
                pval_list.append(stats[1] if stats else None)
        return pd.Series(pval_list, index=p.columns)

    @staticmethod
    def calc_pval(df_pred_str: pd.DataFrame, df_pred_str_null: pd.DataFrame, repetitions_matrix: np.ndarray) -> pd.DataFrame:
        pval_series_list = [
            CVResultsAggregator.calc_pval_row(df_pred_str.iloc[i], df_pred_str_null.iloc[i]) for i in repetitions_matrix
        ]
        pval_df = pd.DataFrame(pval_series_list).fillna(1)
        return pval_df

    @staticmethod
    def plot_1_1(ax: matplotlib.axes.SubplotBase, df_pred_str: pd.DataFrame, df_pred_str_null: pd.DataFrame, rows_set: np.ndarray) -> None:
        mu, sigma = ClusteringUtils.calc_distribution(df_pred_str.loc[rows_set])
        mu_null, sigma_null = ClusteringUtils.calc_distribution(df_pred_str_null.loc[rows_set])
        y = pd.DataFrame([
            mu - sigma,
            mu,
            mu + sigma,
            mu_null - sigma_null,
            mu_null,
            mu_null + sigma_null
        ]).transpose()
        x = pd.DataFrame([df_pred_str.columns for _ in range(len(y.columns))]).transpose()
        lty = ["--", "-", "--", "--", "-", "--"]
        lwd = [1, 2, 1, 1, 2, 1]
        colors = ["blue", "blue", "blue", "red", "red", "red"]
        legend_lines = [
            Line2D([0], [0], color="blue", linestyle="-", linewidth=2),
            Line2D([0], [0], color="red", linestyle="-", linewidth=2),
            Line2D([0], [0], color="gray", linestyle="--", linewidth=1)
        ]
        for col in x.columns:
            ax.plot(x[col], y[col], color=colors[col], linestyle=lty[col], linewidth=lwd[col])
            ax.set_xlabel("k")
            ax.set_ylabel("prediction strength")
        ax.legend(legend_lines, ["model", "null", "95% CI"], loc='upper right')

    @staticmethod
    def plot_1_2(ax: matplotlib.axes.SubplotBase, diff_df: pd.DataFrame, row_id: int) -> None:
        ax.bar(diff_df.columns, diff_df.loc[row_id], color="red")
        ax.set_title("difference")
        ax.set_xlabel("k")
        ax.set_ylabel("diff")
        diff_df_min = diff_df.min().min()
        diff_df_max = diff_df.max().max()
        if CVResultsAggregator.check_limit(diff_df_min) and CVResultsAggregator.check_limit(diff_df_max):
            ax.set_ylim(diff_df_min, diff_df_max)

    @staticmethod
    def plot_2_1(ax: matplotlib.axes.SubplotBase, pval_df: pd.DataFrame, row_id: int) -> None:
        ax.bar(pval_df.columns, -np.log10(pval_df.loc[row_id]), color="red")
        ax.set_title("significance of difference")
        ax.set_xlabel("k")
        ax.set_ylabel("-log10(p-value)")
        ax.set_ylim(0, 4)
        # ax.set_ylim(0, max(-np.log10(pval_df)))

    @staticmethod
    def plot_2_2(ax: matplotlib.axes.SubplotBase, df_eff_k: pd.DataFrame, rows_set: np.ndarray) -> None:
        lty = ["--", "-", "--"]
        lwd = [1, 2, 1]
        mu, sigma = ClusteringUtils.calc_distribution(df_eff_k.loc[rows_set])
        y = pd.DataFrame([mu - sigma, mu, mu + sigma]).transpose()
        for col in y.columns:
            ax.plot(df_eff_k.columns, y[col], color="red", linestyle=lty[col], linewidth=lwd[col])
            ax.set_xlabel("k")
            ax.set_ylabel("effective k")
            ax.set_xlim(1, df_eff_k.columns.max())
            ax.set_ylim(1, df_eff_k.columns.max())
        abline_vals = np.array(ax.get_xlim())
        ax.plot(abline_vals, abline_vals, color="grey", linestyle="--")

    @staticmethod
    def check_limit(number):
        return number is not None and not math.isnan(number) and not math.isinf(number)
