from typing import Dict, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class SwiGLU(nn.Module):
    r"""
    SwiGLU feed-forward module

    :param d_model: Model dimension.
    :type d_model: int

    :param d_ff: Intermediate dimension (should be even, as it's split into two halves).
    :type d_ff: int

    :param bias: Whether to use bias in linear layers. Default: ``True``
    :type bias: bool, optional
    """

    def __init__(self, d_model: int, d_ff: int, bias: Optional[bool] = True):
        super().__init__()
        assert d_ff % 2 == 0, "d_ff must be even"
        self.W1 = nn.Linear(d_model, d_ff, bias=bias)
        self.silu = nn.SiLU()
        self.W2 = nn.Linear(d_ff // 2, d_model, bias=bias)

    def forward(self, x: torch.Tensor, return_states: Optional[bool] = False) -> Union[torch.Tensor, Dict]:
        r"""
        Forward pass of SwiGLU.

        :param x: Input tensor of shape :math:`(..., D)`
        :type x: torch.Tensor

        :param return_states: If True, return intermediate activations and input. Default: ``False``
        :type return_states: bool, optional

        :return: Output tensor :math:`(..., D)` or dict with intermediates states
            containing the keys: "output", "y1", "y2" and "input".
        :rtype: Union[torch.Tensor, Dict]
        """
        y1, y2 = self.W1(x).chunk(2, dim=-1)
        if return_states:
            return {"output": self.W2(y1 * self.silu(y2)), "y1": y1, "y2": y2, "input": x}
        else:
            return self.W2(y1 * self.silu(y2))


class MLP(nn.Module):
    r"""
    Classic MLP with GELU activation (as used in the original Transformer).

    :param d_model: Model dimension.
    :type d_model: int

    :param d_ff: Intermediate dimension.
    :type d_ff: int

    :param bias: Whether to use bias in linear layers. Default: ``True``
    :type bias: bool, optional
    """

    def __init__(self, d_model: int, d_ff: int, bias: Optional[bool] = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=bias),
        )

    def forward(self, x: torch.Tensor, return_states: Optional[bool] = False) -> Union[torch.Tensor, Dict]:
        r"""
        Forward pass of MLP.

        :param x: Input tensor of shape :math:`(..., D)`
        :type x: torch.Tensor

        :param return_states: If True, return intermediate activations. Default: ``False``
        :type return_states: bool, optional

        :return: Output tensor :math:`(..., D)` or dict with intermediates states
            containing the keys: "output", "h1", "h2" and "input".
        :rtype: Union[torch.Tensor, Dict]
        """
        if return_states:
            h1 = self.net[0](x)
            h2 = self.net[1](h1)
            out = self.net[2](h2)
            return {"output": out, "input": x, "h1": h1, "h2": h2}
        else:
            return self.net(x)
