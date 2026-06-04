from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerType(Enum):
    """Type enumeration for layer configuration resolution."""

    STRING = auto()
    NN_MODULE = auto()
    LIST = auto()


def get_layer_type(x: Union[str, Type[nn.Module], List]) -> LayerType:
    """
    Determine the type of a layer configuration value.

    This replaces the fragile integer-based check_type function with a robust Enum-based approach.

    :param x: Object to check (should be a string, nn.Module subclass, or list)
    :type x: Union[Type[nn.Module], str, List]

    :return: LayerType.STRING if string, LayerType.NN_MODULE if nn.Module subclass, LayerType.LIST if list
    :rtype: LayerType

    :raises TypeError: If x is not a valid type
    """
    if isinstance(x, str):
        return LayerType.STRING
    elif isinstance(x, type) and issubclass(x, nn.Module):
        return LayerType.NN_MODULE
    elif isinstance(x, list):
        return LayerType.LIST
    else:
        raise TypeError(
            f"Invalid layer configuration type: {type(x).__name__}. Expected str, nn.Module subclass, or list."
        )


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
            raise ValueError(f"List configuration must have length {n_layers}, got {len(config_value)}")
        return config_value[layer_idx]
    return config_value
