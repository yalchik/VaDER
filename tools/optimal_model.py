import pandas as pd
import numpy as np
import scipy.stats
import warnings
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf
from matplotlib.lines import Line2D

params = ["n_layer", "alpha", "learning_rate", "batch_size", "n_hidden1", "n_hidden2"]

# initialization
num_of_files = 0  # 17
num_of_repetitions = 0  # 90
num_of_k = 0  # 9
data_frames_eff_k = []
data_frames_pred_str = []
data_frames_pred_str_null = []
for i in range(1, 20):
    if i in (13, 14):
        continue
    num_of_files += 1
    df = pd.read_csv(f"step2/file {i} .csv")
    data_frames_eff_k.append(df.pivot(values='effective_k', index=params, columns="k").reset_index())
    data_frames_pred_str.append(df.pivot(values='prediction_strength', index=params, columns="k").reset_index())
    data_frames_pred_str_null.append(df.pivot(values='prediction_strength_null', index=params, columns="k").reset_index())
df_eff_k = pd.concat(data_frames_eff_k, ignore_index=True)
df_eff_pred_str = pd.concat(data_frames_pred_str, ignore_index=True)
df_eff_pred_str_null = pd.concat(data_frames_pred_str_null, ignore_index=True)
num_of_repetitions = len(df_eff_k) // num_of_files
#repetitions_matrix = [[i + num_of_repetitions * j for j in range(num_of_files)] for i in range(num_of_repetitions)]
repetitions_matrix = np.stack(np.split(np.arange(num_of_repetitions * num_of_files), num_of_files)).transpose()
num_of_k = len(df_eff_k.columns) - len(params)
k_columns = df_eff_k.columns[-num_of_k:]
df_params = df_eff_k[params]


def std_diff(df):
    std_diff = df.diff().std() / np.sqrt(2)
    return std_diff


def calc_diff_row(p, q):
    pq_diff = (p - q)  # .applymap(lambda val: val if np.abs(val) > eps else None)
    col_means = pq_diff.mean()
    col_sd_diffs = std_diff(pq_diff)
    col_sums = np.sqrt(p.notna().sum())
    diff = col_means / col_sd_diffs * col_sums
    return diff


def calc_diff(df_eff_pred_str, df_eff_pred_str_null, num_of_k):
    diff_series_list = [
        calc_diff_row(df_eff_pred_str.iloc[ii, -num_of_k:], df_eff_pred_str_null.iloc[ii, -num_of_k:]) for ii in I
    ]
    diff_df = pd.DataFrame(diff_series_list).fillna(0)
    return diff_df


def calc_pval_row(p, q):
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


def calc_pval(df_eff_pred_str, df_eff_pred_str_null, num_of_k):
    pval_series_list = [
        calc_pval_row(df_eff_pred_str.iloc[ii, -num_of_k:], df_eff_pred_str_null.iloc[ii, -num_of_k:]) for ii in I
    ]
    pval_df = pd.DataFrame(pval_series_list).fillna(1)
    return pval_df


def plot_1_1(ax, df_eff_pred_str, df_eff_pred_str_null, rows_set):
    p = df_eff_pred_str.iloc[rows_set, -num_of_k:]
    mu = p.mean()
    sigma = std_diff(p) / np.sqrt(p.notna().sum()) * 1.96  # 95% CI
    q = df_eff_pred_str_null.iloc[rows_set, -num_of_k:]
    mu_null = q.mean()
    sigma_null = std_diff(q) / np.sqrt(q.notna().sum()) * 1.96  # 95% CI
    x = pd.DataFrame([k_columns for _ in range(6)]).transpose()
    y = pd.DataFrame([
        mu - sigma,
        mu,
        mu + sigma,
        mu_null - sigma_null,
        mu_null,
        mu_null + sigma_null
    ]).transpose()
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


def plot_1_2(ax, diff_df, rows_set):
    ax.bar(diff_df.columns, diff_df.loc[rows_set], color="red")
    ax.set_title("difference")
    ax.set_xlabel("k")
    ax.set_ylabel("diff")
    ax.set_ylim(diff_df.min().min(), diff_df.max().max())


def plot_2_1(ax, pval_df, rows_set):
    ax.bar(pval_df.columns, -np.log10(pval_df.loc[rows_set]), color="red")
    ax.set_title("significance of difference")
    ax.set_xlabel("k")
    ax.set_ylabel("-log10(p-value)")
    ax.set_ylim(0, 4)


#     ax.set_ylim(0, max(-np.log10(pval_df)))

def plot_2_2(ax, df_eff_k, rows_set):
    lty = ["--", "-", "--"]
    lwd = [1, 2, 1]
    p = df_eff_k.loc[rows_set, k_columns]
    mu = p.mean()
    sigma = std_diff(p) / np.sqrt(p.notna().sum()) * 1.96  # 95% CI
    y = pd.DataFrame([mu - sigma, mu, mu + sigma]).transpose()
    for col in y.columns:
        ax.plot(p.columns, y[col], color="red", linestyle=lty[col], linewidth=lwd[col])
        ax.set_xlabel("k")
        ax.set_ylabel("effective k")
        ax.set_xlim(1, p.columns.max())
        ax.set_ylim(1, p.columns.max())
    abline_vals = np.array(ax.get_xlim())
    ax.plot(abline_vals, abline_vals, color="grey", linestyle="--")


diff_df = calc_diff(df_eff_pred_str, df_eff_pred_str_null, num_of_k)
pval_df = calc_pval(df_eff_pred_str, df_eff_pred_str_null, num_of_k)
diff_ord_series = diff_df.median(axis=1)
diff_ord_series_sorted = diff_ord_series.sort_values(ascending=False)
pdf = matplotlib.backends.backend_pdf.PdfPages("test.pdf")
for q, i in enumerate(diff_ord_series_sorted.index):
    if q > 2:
        break
    rows_set = repetitions_matrix[i]
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"#{i}: {df_eff_k.loc[i, params].to_dict()}")
    plot_1_1(axs[0, 0], df_eff_pred_str, df_eff_pred_str_null, rows_set)
    #     plot_1_2(axs[0, 1], diff_df, rows_set)
    #     plot_2_1(axs[1, 0], pval_df, rows_set)
    plot_2_2(axs[1, 1], df_eff_k, rows_set)
    pdf.savefig(fig)
pdf.close()
