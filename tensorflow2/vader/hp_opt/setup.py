import logging
from typing import List, Dict, Union

# Type aliases
ParamsDictType = Dict[str, Union[int, float, List[Union[int, float]]]]
ParamsGridType = List[ParamsDictType]

logger = logging.getLogger("vader_hyp_opt_log")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)
