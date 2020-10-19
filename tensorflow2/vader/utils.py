import pandas as pd
import numpy as np
from collections import defaultdict
from typing import Tuple
from typing import Dict


# Type aliases
XTensorDict = Dict[str, Dict[str, np.ndarray]]


def read_x_w_y(filename: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Reads csv file and produces 3 numpy arrays:
    1. X tensor, where 1st dimension is samples, 2nd dimension is time points, 3rd dimension is feature vectors.
    2. W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is nan.
    3. Y vector, which contains class labels for each sample.
    @param filename: input data file in .csv format.
    @return: tuple of 3 elements: X tensor, W tensor and Y vector.
    """
    x_tensor, y_vector = csv_to_numpy_tensor_and_class_vector(filename)
    w_tensor = generate_wtensor_from_xtensor(x_tensor)
    return x_tensor, w_tensor, y_vector


def csv_to_numpy_tensor_and_class_vector(filename: str, class_atr: str = "cluster", feature_time_separator: str = "_")\
        -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads csv file and produces 2 numpy arrays:
    1. X tensor, where 1st dimension is samples, 2nd dimension is time points, 3rd dimension is feature vectors.
    2. Y vector, which contains class labels for each sample.
    @param filename: input data file in .csv format.
    @param class_atr: name of the column that contains class labels.
    @param feature_time_separator: character between feature and time labels in feature columns names.
    @return: tuple of 2 elements: X tensor and Y vector.
    """
    df = pd.read_csv(filename)
    y_vector = df[class_atr].to_numpy()
    x_dict = __map_dataframe_to_xdict(df, class_atr, feature_time_separator)
    x_tensor = __map_xdict_to_xtensor(x_dict)
    return x_tensor, y_vector


def generate_wtensor_from_xtensor(x_tensor: np.ndarray) -> np.ndarray:
    """
    Generates W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is nan.
    @param x_tensor: X tensor
    @return: W tensor
    """
    return np.isnan(x_tensor).astype(int)


def __map_dataframe_to_xdict(df: pd.DataFrame, class_atr: str, feature_time_separator: str) -> XTensorDict:
    x_dict = defaultdict(dict)
    for column in df:
        if column == class_atr:
            continue
        feature, time = column.split(feature_time_separator)
        x_dict[feature][time] = df[column].to_numpy()
    return x_dict


def __map_xdict_to_xtensor(x_dict: XTensorDict) -> np.ndarray:
    # noinspection PyTypeChecker
    return np.array([list(val.values()) for val in x_dict.values()]).transpose(2, 1, 0)


if __name__ == "__main__":
    pass
