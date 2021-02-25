from abc import ABC, abstractmethod
from typing import Dict, List, Union


class AbstractBayesianParamsFactory(ABC):

    @abstractmethod
    def get_k_list(self) -> List[int]:
        pass

    @abstractmethod
    def get_param_limits_dict(self) -> Dict[str, List[Union[int, float]]]:
        pass
