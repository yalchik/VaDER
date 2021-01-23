import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, Optional
from vader.utils.clustering_utils import ClusteringUtils

# Type aliases
XTensorDict = Dict[str, Dict[str, np.ndarray]]


def read_artificial_data(filename: str, class_atr: str = "cluster") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads csv file with artificial data and produces 3 numpy arrays:
    1. X tensor, where:
        1st dimension is samples,
        2nd dimension is time points,
        3rd dimension is feature vectors.
    2. W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is missing.
    3. Y vector, which contains class labels for each sample.

    Parameters
    ----------
    filename : str
        input data file in .csv format.
    class_atr : str
        column name that defines class attribute

    Returns
    -------
    tuple of 3 elements: X tensor, W tensor and Y vector.
    """
    df = pd.read_csv(filename)
    x_dict = defaultdict(dict)
    y_vector = df[class_atr].to_numpy() if class_atr else None
    feature_columns = [column for column in df if column != class_atr and "_" in column]
    for column in feature_columns:
        feature, time = column.split("_", 1)
        x_dict[feature][time] = df[column].to_numpy()
    x_tensor_with_nans = __map_xdict_to_xtensor(x_dict)
    w_tensor = generate_wtensor_from_xtensor(x_tensor_with_nans)
    x_tensor = np.nan_to_num(x_tensor_with_nans)
    return x_tensor, w_tensor, y_vector


def read_adni_norm_data(filename: str, features: tuple = ("CDRSB", "MMSE", "ADAS11"),
                        time_points: tuple = ("0", "6", "12", "24", "36")) -> np.ndarray:
    """
    Reads csv file with ADNI data and produces 2 numpy arrays:
    1. X tensor, where:
        1st dimension is samples,
        2nd dimension is time points,
        3rd dimension is feature vectors.
    2. W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is missing.

    Parameters
    ----------
    filename : str
        input data file in .csv format.
    features : tuple
        list of required features
    time_points : tuple
        list of required time points

    Returns
    -------
    tuple of 2 elements: X tensor, W tensor.
    """
    df = pd.read_csv(filename)

    x_dict = OrderedDict.fromkeys(features)
    for feature in features:
        x_dict[feature] = OrderedDict.fromkeys(time_points)

    feature_columns = [column for column in df if "." in column]
    for column in feature_columns:
        time, feature = column.split(".", 1)
        time = time[1:]  # remove leading 'X'
        if feature in features and time in time_points:
            x_dict[feature][time] = df[column].to_numpy()

    x_tensor_with_nans = __map_xdict_to_xtensor(x_dict)
    return x_tensor_with_nans


def read_nacc_data(filename: str, normalize: bool = True,
                   features: tuple = ("NACCMMSE", "CDRSUM", "NACCFAQ")) -> np.ndarray:
    """
    Reads csv file with NACC data and produces 2 numpy arrays:
    1. X tensor, where:
        1st dimension is samples,
        2nd dimension is time points,
        3rd dimension is feature vectors.
    2. W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is missing.

    Parameters
    ----------
    filename : str
        input data file in .csv format.
    normalize : bool
        if True, the function will normalize the data. Otherwise it will use it without normalization.
    features : tuple
        list of required features

    Returns
    -------
    tuple of 2 elements: X tensor, W tensor.
    """
    df = pd.read_csv(filename, index_col=0)

    features_list = list(features)
    if normalize:
        features_df = df[features_list]
        df[features_list] = (features_df - features_df.mean()) / features_df.std()
    time_points = [str(i) for i in range(1, df["NACCVNUM"].max()+1)]
    pivoted_normalized_df = df.pivot(index="NACCID", columns="NACCVNUM", values=features_list)
    pivoted_normalized_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_normalized_df.columns.values]
    df = pivoted_normalized_df

    x_dict = OrderedDict.fromkeys(features)
    for feature in features:
        x_dict[feature] = OrderedDict.fromkeys(time_points)

    feature_columns = [column for column in df if "_" in column]
    for column in feature_columns:
        feature, time = column.split("_", 1)
        if feature in features and time in time_points:
            x_dict[feature][time] = df[column].to_numpy()

    x_tensor_with_nans = __map_xdict_to_xtensor(x_dict)
    return x_tensor_with_nans


def preprocess_adni_patient(df_p, time_points_list):
    # get first time point with Dementia
    time_points_ad = df_p.loc[df_p.DX == "Dementia"]
    if not len(time_points_ad):
        time_points_ad = df_p.loc[df_p.DX_bl == "AD"]

    first_dementia_tp = time_points_ad.iloc[0].TIME_POINT

    # get offset
    time_points_index_dict = {tp: i for i, tp in enumerate(time_points_list)}
    offset = time_points_index_dict[first_dementia_tp]
    if offset:
        df_p_new = df_p.iloc[offset - 1:].copy()
        df_p_new["TIME_POINT"] = df_p_new["TIME_POINT"].apply(
            lambda x: time_points_list[time_points_index_dict[x] - offset])
    else:
        df_p_new = df_p.copy()
    return df_p_new


def preprocess_adni_data(df):
    patients_list = df.loc[(df.DX_bl == "AD") | (
            df.DX == "Dementia")].PTID.unique()  # select all patients who was at least once diagnosed as AD

    df_ad = df.loc[df.PTID.isin(patients_list)].copy()
    df_ad['TIME_POINT'] = df_ad['VISCODE'].apply(lambda x: int(x[1:]) if x != "bl" else 0)
    df_ad = df_ad.sort_values(by=["PTID", "TIME_POINT"])

    time_points_list = sorted(df_ad.TIME_POINT.unique())  # time points for AD patients

    processed_patients_df_list = []
    for patient_id in patients_list:
        df_p = df_ad.loc[df.PTID == patient_id]
        df_p_processed = preprocess_adni_patient(df_p, time_points_list)
        processed_patients_df_list.append(df_p_processed)

    df_processed = pd.concat(processed_patients_df_list)
    return df_processed


def read_adni_raw_data(filename: str, normalize: bool = True,
                       features: tuple = ("CDRSB", "MMSE", "ADAS11")) -> np.ndarray:
    features_list = list(features)

    df = pd.read_csv(filename).loc[:, ("PTID", "VISCODE", "DX_bl", "DX", "CDRSB", "MMSE", "ADAS11")]
    df_ad_filtered = preprocess_adni_data(df).loc[:, ("PTID", "TIME_POINT", "CDRSB", "MMSE", "ADAS11")]
    time_points: tuple = ("0", "6", "12", "24", "36")

    df_ad_filtered_normalized_pivoted = df_ad_filtered.pivot(index="PTID", columns="TIME_POINT", values=features_list)
    df_ad_filtered_normalized_pivoted.columns = [f"{col[0]}_{col[1]}" for col in
                                                 df_ad_filtered_normalized_pivoted.columns.values]
    df = df_ad_filtered_normalized_pivoted

    x_dict = OrderedDict.fromkeys(features)
    for feature in features:
        x_dict[feature] = OrderedDict.fromkeys(time_points)

    feature_columns = [column for column in df if "_" in column]
    for column in feature_columns:
        feature, time = column.split("_", 1)
        if feature in features and time in time_points:
            x_dict[feature][time] = df[column].to_numpy()

    x_tensor_with_nans = __map_xdict_to_xtensor(x_dict)

    if normalize:
        adni_std = np.array([1.720699, 6.699345, 2.682827])
        x_tensor_with_nans = ClusteringUtils.calc_z_scores(x_tensor_with_nans, adni_std)

    return x_tensor_with_nans


def generate_wtensor_from_xtensor(x_tensor: np.ndarray) -> np.ndarray:
    """
    Generates W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is
    missing.

    Parameters
    ----------
    x_tensor : np.ndarray
        3D numpy array, where 1st dimension is samples, 2nd dimension is time points, 3rd dimension is feature vectors.

    Returns
    -------
    W tensor
    """
    w = ~np.isnan(x_tensor)
    return w.astype(int)


def __map_xdict_to_xtensor(x_dict: XTensorDict) -> np.ndarray:
    # noinspection PyTypeChecker
    return np.array([list(val.values()) for val in x_dict.values()]).transpose(2, 1, 0)


def generate_x_w_y(num_of_time_points: int = 7, num_of_samples: int = 400) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generates some simple random data [ns * 2 samples, nt - 1 time points, 2 variables]

    Parameters
    ----------
    num_of_time_points : int
        number of time points (default is 7)
    num_of_samples : int
        number of samples (default is 400)

    Returns
    -------
    tuple of 3 elements: X tensor, W tensor and Y vector.
    """
    nt = num_of_time_points + 1
    ns = num_of_samples // 2
    sigma = 0.5
    mu1 = -2
    mu2 = 2
    a1 = np.random.normal(mu1, sigma, ns)
    a2 = np.random.normal(mu2, sigma, ns)
    # one variable with two clusters
    X0 = np.outer(a1, 2 * np.append(np.arange(-nt / 2, 0), 0.5 * np.arange(0, nt / 2)[1:]))
    X1 = np.outer(a2, 0.5 * np.append(np.arange(-nt / 2, 0), 2 * np.arange(0, nt / 2)[1:]))
    X_train = np.concatenate((X0, X1), axis=0)
    y_train = np.repeat([0, 1], ns)
    # add another variable as a random permutation of the first one
    # resulting in four clusters in total
    ii = np.random.permutation(ns * 2)
    X_train = np.stack((X_train, X_train[ii, ]), axis=2)
    # we now get four clusters in total
    y_train = y_train * 2 ** 0 + y_train[ii] * 2 ** 1

    # normalize (better for fitting)
    for i in np.arange(X_train.shape[2]):
        X_train[:, :, i] = (X_train[:, :, i] - np.mean(X_train[:, :, i])) / np.std(X_train[:, :, i])

    # randomly re-order the samples
    ii = np.random.permutation(ns * 2)
    X_train = X_train[ii, :]
    y_train = y_train[ii]
    # Randomly set 50% of values to missing (0: missing, 1: present)
    # Note: All X_train[i,j] for which W_train[i,j] == 0 are treated as missing (i.e. their specific value is ignored)
    W_train = np.random.choice(2, X_train.shape)
    return X_train, W_train, y_train


def generate_x_y_for_nonrecur(num_of_time_points: int = 7, num_of_samples: int = 400) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates some simple random data for non-recurrent VaDER (ordinary VAE with GM prior)

    Parameters
    ----------
    num_of_time_points : int
        number of time points (default is 7)
    num_of_samples : int
        number of samples (default is 400)

    Returns
    -------
    tuple of 2 elements: X tensor and Y vector.
    """
    nt = num_of_time_points
    ns = num_of_samples // 2
    sigma = np.diag(np.repeat(2, nt))
    mu1 = np.repeat(-1, nt)
    mu2 = np.repeat(1, nt)
    a1 = np.random.multivariate_normal(mu1, sigma, ns)
    a2 = np.random.multivariate_normal(mu2, sigma, ns)
    X_train = np.concatenate((a1, a2), axis=0)
    y_train = np.repeat([0, 1], ns)
    ii = np.random.permutation(ns * 2)
    X_train = X_train[ii, :]
    y_train = y_train[ii]
    # normalize (better for fitting)
    X_train = (X_train - np.mean(X_train)) / np.std(X_train)
    return X_train, y_train


if __name__ == "__main__":
    pass
