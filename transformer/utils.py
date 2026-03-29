import math
import os
import random
import sys
from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_type(x: Union[Type[nn.Module], str]) -> int:
    if isinstance(x, str):
        return 0
    elif isinstance(x, type) and issubclass(x, nn.Module):
        return 1
    else:
        raise TypeError(f"Type not valid: {x}")
