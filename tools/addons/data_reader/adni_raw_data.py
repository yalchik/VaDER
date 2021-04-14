import pandas as pd
import numpy as np
from collections import OrderedDict
from vader.utils.data_utils import map_xdict_to_xtensor
from vader.utils.clustering_utils import ClusteringUtils
from vader.hp_opt.interface.abstract_data_reader import AbstractDataReader


class DataReader(AbstractDataReader):
    normalize: bool = True
    features: tuple = ("CDRSB", "MMSE", "ADAS11")
    time_points: tuple = ("0", "6", "12", "24", "36")
    time_point_meaning: str = "month"
    ids_list: list = None

    def read_data(self, filename: str) -> np.ndarray:
        features_list = list(self.features)

        df = pd.read_csv(filename).loc[:, ("PTID", "VISCODE", "DX_bl", "DX", "CDRSB", "MMSE", "ADAS11")]
        df_ad_filtered = self.__preprocess_adni_data(df).loc[:, ("PTID", "TIME_POINT", "CDRSB", "MMSE", "ADAS11")]

        df_ad_filtered_normalized_pivoted = df_ad_filtered.pivot(index="PTID", columns="TIME_POINT", values=features_list)
        df_ad_filtered_normalized_pivoted.columns = [f"{col[0]}_{col[1]}" for col in
                                                     df_ad_filtered_normalized_pivoted.columns.values]
        df = df_ad_filtered_normalized_pivoted
        self.ids_list = list(df.index)

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
            adni_std = np.array([1.720699, 6.699345, 2.682827])
            x_tensor_with_nans = ClusteringUtils.calc_z_scores(x_tensor_with_nans, adni_std)

        return x_tensor_with_nans

    def __preprocess_adni_data(self, df):
        patients_list = df.loc[(df.DX_bl == "AD") | (
                df.DX == "Dementia")].PTID.unique()  # select all patients who was at least once diagnosed as AD

        df_ad = df.loc[df.PTID.isin(patients_list)].copy()
        df_ad['TIME_POINT'] = df_ad['VISCODE'].apply(lambda x: int(x[1:]) if x != "bl" else 0)
        df_ad = df_ad.sort_values(by=["PTID", "TIME_POINT"])

        time_points_list = sorted(df_ad.TIME_POINT.unique())  # time points for AD patients

        processed_patients_df_list = []
        for patient_id in patients_list:
            df_p = df_ad.loc[df.PTID == patient_id]
            df_p_processed = self.__preprocess_adni_patient(df_p, time_points_list)
            processed_patients_df_list.append(df_p_processed)

        df_processed = pd.concat(processed_patients_df_list)
        return df_processed

    @staticmethod
    def __preprocess_adni_patient(df_p, time_points_list):
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
