from numpy import ndarray
from abc import ABC, abstractmethod
from typing import Tuple


class AbstractDataReader(ABC):

    @abstractmethod
    def read_data(self, filename: str) -> Tuple[ndarray, ndarray]:
        """
        Reads a given csv file and produces 2 tensors (X, W), where each tensor has this structure:
          1st dimension is samples,
          2nd dimension is time points,
          3rd dimension is feature vectors.
        X represents input data
        W contains values 0 or 1 for each point of X.
          "0" means the point should be ignored (e.g. because the data is missing)
          "1" means the point should be used for training

        Implementation examples: vader.utils.read_adni_data or vader.utils.read_nacc_data
        """
        pass
