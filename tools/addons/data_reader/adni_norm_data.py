import pandas as pd
from collections import OrderedDict
from vader.utils.data_utils import map_xdict_to_xtensor
from vader.hp_opt.interface.abstract_data_reader import AbstractDataReader


class DataReader(AbstractDataReader):
    features: tuple = ("CDRSB", "MMSE", "ADAS11")
    time_points: tuple = ("0", "6", "12", "24", "36")
    time_point_meaning: str = "month"
    ids_list: list = None

    def read_data(self, filename: str):
        df = pd.read_csv(filename)
        self.ids_list = list(df.index)

        x_dict = OrderedDict.fromkeys(self.features)
        for feature in self.features:
            x_dict[feature] = OrderedDict.fromkeys(self.time_points)

        feature_columns = [column for column in df if "." in column]
        for column in feature_columns:
            time, feature = column.split(".", 1)
            time = time[1:]  # remove leading 'X'
            if feature in self.features and time in self.time_points:
                x_dict[feature][time] = df[column].to_numpy()

        x_tensor_with_nans = map_xdict_to_xtensor(x_dict)
        return x_tensor_with_nans
