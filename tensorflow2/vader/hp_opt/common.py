from numpy import ndarray
from typing import List, Dict, Union, Optional
from vader.hp_opt.log_manager import LogManager

# Type aliases
ParamsDictType = Dict[str, Union[int, float, List[Union[int, float]]]]
ParamsGridType = List[ParamsDictType]
ClusteringType = Union[ndarray, List[int]]

# Global variables
log_manager = LogManager()
