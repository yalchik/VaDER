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
    time_points = ("1", "2", "3", "4", "5")
    time_point_meaning: str = "visit"

    def read_data(self, filename: str) -> np.ndarray:
        features_list = list(self.features)

        df = pd.read_csv(filename).loc[:, ("NACCID", "NACCVNUM", "Diagnosis_BL", "Diagnosis", "NACCMMSE", "CDRSUM", "NACCFAQ")]
        df_ad_filtered = self.__preprocess_nacc_data(df).loc[:, ("NACCID", "TIME_POINT", "NACCMMSE", "CDRSUM", "NACCFAQ")]
        # time_points = tuple([str(t) for t in sorted(df_ad_filtered.TIME_POINT.unique())])

        df_ad_filtered_normalized_pivoted = df_ad_filtered.pivot(index="NACCID", columns="TIME_POINT", values=features_list)
        df_ad_filtered_normalized_pivoted.columns = [f"{col[0]}_{col[1]}" for col in
                                                     df_ad_filtered_normalized_pivoted.columns.values]
        df = df_ad_filtered_normalized_pivoted

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

    def __preprocess_nacc_data(self, df):
        patients_list = df.loc[(df.Diagnosis_BL == "AD") | (
                df.Diagnosis == "AD")].NACCID.unique()  # select all patients who was at least once diagnosed as AD

        df_ad = df.loc[df.NACCID.isin(patients_list)].copy()
        df_ad['TIME_POINT'] = df_ad['NACCVNUM']
        df_ad = df_ad.sort_values(by=["NACCID", "TIME_POINT"])

        time_points_list = sorted(df_ad.TIME_POINT.unique())  # time points for AD patients

        processed_patients_df_list = []
        for patient_id in patients_list:
            df_p = df_ad.loc[df.NACCID == patient_id]
            df_p_processed = self.__preprocess_nacc_patient(df_p, time_points_list)
            processed_patients_df_list.append(df_p_processed)

        df_processed = pd.concat(processed_patients_df_list)
        return df_processed

    @staticmethod
    def __preprocess_nacc_patient(df_p, time_points_list):
        # get first time point with Dementia
        time_points_ad = df_p.loc[df_p.Diagnosis == "AD"]
        if not len(time_points_ad):
            time_points_ad = df_p.loc[df_p.Diagnosis_BL == "AD"]

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
