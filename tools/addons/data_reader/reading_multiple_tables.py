import pandas as pd
import numpy as np
import os
from vader.hp_opt.interface.abstract_data_reader import AbstractDataReader

class DataReader(AbstractDataReader):
    """A DataReader that takes all csv files stored in a directory and stacks them to a tensor. 
    The format of the csv files should be samples x time points and 
    dimensions need to be consistend across files."""
    
    def read_data(self, directory: str) -> np.ndarray:

        # read all csv files in path into one list of pd.DataFrames
        csvs = [pd.read_csv(os.path.join(directory, csv), index_col=0) for csv in os.listdir(directory)]
        # stack tables into a tensor with dim samples x time points x tables
        tensor = np.stack(csvs, axis=2)
        return tensor
