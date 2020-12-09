from typing import List, Dict, Union

# Type aliases
ParamsDictType = Dict[str, Union[int, float, List[Union[int, float]]]]
ParamsGridType = List[ParamsDictType]

PARAMS_COLUMN_NAME = "params"
