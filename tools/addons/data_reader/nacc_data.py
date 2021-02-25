import pandas as pd
import numpy as np
from collections import OrderedDict
from typing import Dict, Tuple, Optional
from vader.utils.data_utils import map_xdict_to_xtensor
from vader.utils.clustering_utils import ClusteringUtils
from vader.hp_opt.interface.abstract_data_reader import AbstractDataReader


class DataReader(AbstractDataReader):
    normalize: bool = True
    features: tuple = ("NACCMMSE", "CDRSUM", "NACCFAQ")
    time_points: tuple = None
    time_point_meaning: str = "visit"

    def read_data(self, filename: str) -> np.ndarray:
        df = pd.read_csv(filename, index_col=0)

        features_list = list(self.features)
        self.time_points = [str(i) for i in range(1, df["NACCVNUM"].max()+1)]
        pivoted_normalized_df = df.pivot(index="NACCID", columns="NACCVNUM", values=features_list)
        pivoted_normalized_df.columns = [f"{col[0]}_{col[1]}" for col in pivoted_normalized_df.columns.values]
        df = pivoted_normalized_df

        x_dict = OrderedDict.fromkeys(self.features)
        for feature in self.features:
            x_dict[feature] = OrderedDict.fromkeys(self.time_points)

        feature_columns = [column for column in df if "_" in column]
        for column in feature_columns:
            feature, time = column.split("_", 1)
            if feature in self.features and time in self.time_points:
                x_dict[feature][time] = df[column].to_numpy()

        x_tensor_with_nans = map_xdict_to_xtensor(x_dict)

        if self.normalize:
            x_tensor_with_nans = ClusteringUtils.calc_z_scores(x_tensor_with_nans)

        return x_tensor_with_nans
