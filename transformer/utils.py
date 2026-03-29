import math
import os
import random
import sys
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


def check_type(x: Union[Type[nn.Module], str]) -> int:
    """
    Check the type of x and return an integer code.
    
    :param x: Object to check (should be a string or nn.Module subclass)
    :type x: Union[Type[nn.Module], str]
    
    :return: 0 if string, 1 if nn.Module subclass
    :rtype: int
    
    :raises TypeError: If x is neither a string nor an nn.Module subclass
    """
    if isinstance(x, str):
        return 0
    elif isinstance(x, type) and issubclass(x, nn.Module):
        return 1
    else:
        raise TypeError(f"Type not valid: {x}")


def resolve_layer_config(config_value: Union[str, Type[nn.Module], List], layer_idx: int, n_layers: int):
    """
    Resolve configuration value for a specific layer index.
    Supports both uniform (single value) and per-layer (list) configurations.
    
    :param config_value: Configuration value (string, type, or list)
    :type config_value: Union[str, Type[nn.Module], List]
    
    :param layer_idx: Index of the current layer
    :type layer_idx: int
    
    :param n_layers: Total number of layers
    :type n_layers: int
    
    :return: Resolved configuration value for this layer
    :raises ValueError: If list length doesn't match n_layers
    """
    if isinstance(config_value, list):
        if len(config_value) != n_layers:
            raise ValueError(
                f"List configuration must have length {n_layers}, got {len(config_value)}"
            )
        return config_value[layer_idx]
    return config_value
