import pandas as pd
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Dict, Tuple, Optional, Callable

# Type aliases
XTensorDict = Dict[str, Dict[str, np.ndarray]]
ColumnMapperFn = Callable[[str, str, list, list], Tuple[str, str]]


def read_artificial_data(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    Returns
    -------
    tuple of 3 elements: X tensor, W tensor and Y vector.
    """
    df = pd.read_csv(filename)
    x_tensor, w_tensor, y_vector = __df_to_numpy_tensor_and_class_vector(df, class_atr="cluster", feature_time_separator="_")
    return x_tensor, w_tensor, y_vector


def read_adni_data(filename: str) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    tuple of 2 elements: X tensor, W tensor.
    """
    def column_mapper_fn(column, feature_time_separator, required_features_list, required_time_points_list):
        time, feature = column.split(feature_time_separator, 1)
        time = time[1:]  # remove leading 'X'
        # ignore all non-specified features and time points
        if feature not in required_features_list or time not in required_time_points_list:
            return None, None
        return feature, time

    df = pd.read_csv(filename)
    x_tensor, w_tensor, _ = __df_to_numpy_tensor_and_class_vector(
        df,
        class_atr=None,
        feature_time_separator=".",
        required_features_list=["CDRSB", "MMSE", "ADAS11"],
        required_time_points_list=["0", "6", "12", "24", "36"],
        column_mapper_fn=column_mapper_fn
    )
    return x_tensor, w_tensor


def read_nacc_data(filename: str, normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns
    -------
    tuple of 2 elements: X tensor, W tensor.
    """
    df = pd.read_csv(filename, index_col=0)
    required_features_list = ["NACCMMSE", "CDRSUM", "NACCFAQ"]
    if normalize:
        features_df = df[required_features_list]
        df[required_features_list] = (features_df - features_df.mean()) / features_df.std()
    required_time_points_list = [str(i) for i in range(1, df["NACCVNUM"].max()+1)]
    pivoted_normalized_df = df.pivot(index="NACCID", columns="NACCVNUM", values=required_features_list)
    pivoted_normalized_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_normalized_df.columns.values]
    x_tensor, w_tensor, _ = __df_to_numpy_tensor_and_class_vector(
        pivoted_normalized_df,
        required_features_list=required_features_list,
        required_time_points_list=required_time_points_list
    )
    return x_tensor, w_tensor


def __df_to_numpy_tensor_and_class_vector(
        df: pd.DataFrame,
        class_atr: Optional[str] = None,
        feature_time_separator: str = "_",
        required_features_list: list = None,
        required_time_points_list: list = None,
        column_mapper_fn: ColumnMapperFn = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y_vector = df[class_atr].to_numpy() if class_atr else None
    x_dict = __map_dataframe_to_xdict(df, class_atr, feature_time_separator, required_features_list,
                                      required_time_points_list, column_mapper_fn)
    x_tensor_with_nans = __map_xdict_to_xtensor(x_dict)
    w_tensor = generate_wtensor_from_xtensor(x_tensor_with_nans)
    x_tensor = np.nan_to_num(x_tensor_with_nans)
    return x_tensor, w_tensor, y_vector


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


def __map_dataframe_to_xdict(
        df: pd.DataFrame,
        class_atr: str,
        feature_time_separator: str,
        required_features_list: list,
        required_time_points_list: list,
        column_mapper_fn: ColumnMapperFn
) -> XTensorDict:
    x_dict = defaultdict(dict)
    if required_features_list:
        x_dict = OrderedDict.fromkeys(required_features_list)
    if required_time_points_list:
        for feature in required_features_list:
            x_dict[feature] = OrderedDict.fromkeys(required_time_points_list)

    for column in df:
        if column == class_atr:
            continue
        if feature_time_separator in column:
            if column_mapper_fn:
                feature, time = column_mapper_fn(column, feature_time_separator, required_features_list,
                                                 required_time_points_list)
            else:
                feature, time = column.split(feature_time_separator, 1)
            if feature and time:
                x_dict[feature][time] = df[column].to_numpy()
    return x_dict


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
