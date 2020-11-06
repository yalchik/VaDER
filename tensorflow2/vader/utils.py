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
    1. X tensor, where 1st dimension is samples, 2nd dimension is time points, 3rd dimension is feature vectors.
    2. W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is missing.
    3. Y vector, which contains class labels for each sample.
    @param filename: input data file in .csv format.
    @return: tuple of 3 elements: X tensor, W tensor and Y vector.
    """
    x_tensor, y_vector = csv_to_numpy_tensor_and_class_vector(filename, class_atr="cluster", feature_time_separator="_")
    w_tensor = generate_wtensor_from_xtensor(x_tensor)
    return x_tensor, w_tensor, y_vector


def read_adni_data(filename: str) -> np.ndarray:
    """
    Reads csv file with ADNI data and produces 1 numpy array:
    1. X tensor, where 1st dimension is samples, 2nd dimension is time points, 3rd dimension is feature vectors.
    @param filename: input data file in .csv format.
    @return: X tensor.
    """
    def column_mapper_fn(column, feature_time_separator, required_features_list, required_time_points_list):
        time, feature = column.split(feature_time_separator, 1)
        time = time[1:]  # remove leading 'X'
        # ignore all non-specified features and time points
        if feature not in required_features_list or time not in required_time_points_list:
            return None, None
        return feature, time

    x_tensor, _ = csv_to_numpy_tensor_and_class_vector(
        filename,
        class_atr=None,
        feature_time_separator=".",
        required_features_list=["CDRSB", "MMSE", "ADAS11"],
        required_time_points_list=["0", "6", "12", "24", "36"],
        column_mapper_fn=column_mapper_fn
    )
    return x_tensor


def csv_to_numpy_tensor_and_class_vector(
        filename: str,
        class_atr: Optional[str] = None,
        feature_time_separator: str = "_",
        required_features_list: list = None,
        required_time_points_list: list = None,
        column_mapper_fn: ColumnMapperFn = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reads csv file and produces 2 numpy arrays:
    1. X tensor, where 1st dimension is samples, 2nd dimension is time points, 3rd dimension is feature vectors.
    2. Y vector, which contains class labels for each sample.
    @param filename: input data file in .csv format.
    @param class_atr: name of the column that contains class labels.
    @param feature_time_separator: character between feature and time labels in feature columns names.
    @param required_features_list: list of features that we need to extract from the input file.
    @param required_time_points_list: list of time points that we need to extract form the input file.
    @param column_mapper_fn: mapping from a column name to (feature, time) tuple.
    @return: tuple of 2 elements: X tensor and Y vector.
    """
    df = pd.read_csv(filename)
    y_vector = df[class_atr].to_numpy() if class_atr else None
    x_dict = __map_dataframe_to_xdict(df, class_atr, feature_time_separator, required_features_list,
                                      required_time_points_list, column_mapper_fn)
    x_tensor = __map_xdict_to_xtensor(x_dict)
    return x_tensor, y_vector


def generate_wtensor_from_xtensor(x_tensor: np.ndarray) -> np.ndarray:
    """
    Generates W tensor, which has the same shape as X tensor, but defines if the corresponding value in X tensor is
    missing.
    @param x_tensor: X tensor
    @return: W tensor
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
                x_dict[feature][time] = df[column].replace(np.nan, 0).to_numpy()
    return x_dict


def __map_xdict_to_xtensor(x_dict: XTensorDict) -> np.ndarray:
    # noinspection PyTypeChecker
    return np.array([list(val.values()) for val in x_dict.values()]).transpose(2, 1, 0)


if __name__ == "__main__":
    pass
