import math
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy import ndarray
from typing import List, Union, Tuple
from vader.hp_opt.common import ClusteringType
from vader.utils.clustering_utils import ClusteringUtils


def plot_z_scores(x_tensor: ndarray, clustering_list: ClusteringType, features_list: List[str],
                  time_points_list: Union[List[int], tuple], cmap_name: str = "Set1",
                  x_label: str = "month") -> matplotlib.figure.Figure:
    """
    Generates normalized cluster mean trajectories relative to baseline (x-axis in months).
    Reference: Figure 4 from the paper.

    See: https://github.com/yalchik/VaDER/issues/8
    """
    features_list = ['CDRSB', 'MMSE', 'ADAS11']
    clustering_dict = ClusteringUtils.clustering_to_dict(clustering_list)
    n_features = len(features_list)
    n_clusters = len(clustering_dict)
    n_plots = math.ceil(math.sqrt(n_features))
    colors_list = [plt.get_cmap(cmap_name)(i) for i in range(n_clusters)]

    # get z-scores as tensor, where
    #  1st dimension is features
    #  2nd dimension is samples
    #  3rd dimension is time points
    z_scores_by_feature = x_tensor.transpose((2, 0, 1))

    # start plotting
    fig, axs = plt.subplots(n_plots, n_plots, figsize=(15, 10))
    fig.suptitle(f"Normalized cluster mean trajectories relative to baseline (x-axis in months)")
    for i, z_score_feature in enumerate(z_scores_by_feature):
        plt_index = (i // n_plots, i % n_plots)
        z_score_feature_df = pd.DataFrame(z_score_feature)
        for j, rows_set in clustering_dict.items():
            z_score_feature_cluster_df = z_score_feature_df.iloc[rows_set, :]
            mu, sigma = ClusteringUtils.calc_distribution(z_score_feature_cluster_df)
            axs[plt_index].plot(time_points_list, mu,         color=colors_list[j], linestyle="-",  linewidth=2)
            axs[plt_index].plot(time_points_list, mu - sigma, color=colors_list[j], linestyle="--", linewidth=1)
            axs[plt_index].plot(time_points_list, mu + sigma, color=colors_list[j], linestyle="--", linewidth=1)
        axs[plt_index].set_title(features_list[i])
        axs[plt_index].set_xlabel(x_label)
        axs[plt_index].set_ylabel("z−score relative to baseline")

    # create legend
    clustering_labels = [f"{key} (n = {len(val)})" for key, val in clustering_dict.items()]
    clustering_labels.append("95% CI")
    legend_lines = [Line2D([0], [0], color=c, linestyle="-", linewidth=2) for c in colors_list]
    legend_lines.append(Line2D([0], [0], color="grey", linestyle="--", linewidth=1))
    axs[n_plots - 1, n_plots - 1].legend(legend_lines, clustering_labels, loc='upper right')

    return fig
