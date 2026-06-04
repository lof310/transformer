"""LoRA (Low-Rank Adaptation) modules for parameter-efficient fine-tuning."""

import math
from typing import List, Optional

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    A thin wrapper around nn.Linear that adds LoRA (Low-Rank Adaptation) adapters.

    The original linear layer is kept frozen, and two low-rank trainable matrices
    (lora_A and lora_B) are added. The output is computed as:
        output = W_original @ x + (lora_B @ lora_A) @ x * scaling

    :param original_layer: The original nn.Linear layer to wrap.
    :type original_layer: nn.Linear

    :param lora_rank: Rank of the LoRA decomposition. Default: 8
    :type lora_rank: int, optional

    :param lora_alpha: Scaling factor for LoRA weights. Default: 16
    :type lora_alpha: int, optional

    :param lora_dropout: Dropout probability for LoRA layers. Default: 0.0
    :type lora_dropout: float, optional
    """

    def __init__(
        self,
        original_layer: nn.Linear,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
    ):
        super().__init__()
        self.original_layer = original_layer
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / lora_rank

        # Freeze original layer parameters
        for param in self.original_layer.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(original_layer.out_features, lora_rank))
        self.lora_B = nn.Parameter(torch.zeros(lora_rank, original_layer.in_features))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        self.lora_dropout = nn.Dropout(lora_dropout) if lora_dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing original output plus LoRA adaptation.

        :param x: Input tensor of shape (..., in_features).
        :type x: torch.Tensor

        :return: Output tensor of shape (..., out_features).
        :rtype: torch.Tensor
        """
        orig_out = self.original_layer(x)
        lora_out = self.lora_dropout(x) @ self.lora_B.T @ self.lora_A.T * self.scaling
        return orig_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    target_modules: List[str],
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.0,
) -> nn.Module:
    """
    Recursively walks the model's module tree and replaces every nn.Linear
    whose attribute name matches any string in target_modules with a LoRALinear wrapper.

    After replacement, freeze all parameters except those with 'lora_A' or 'lora_B'
    in their name to make the model ready for parameter-efficient fine-tuning.

    :param model: The PyTorch model to modify.
    :type model: nn.Module

    :param target_modules: List of attribute name patterns to target (e.g., ['qkv_proj', 'W1']).
        Modules whose names contain any of these strings will be wrapped.
    :type target_modules: List[str]

    :param lora_rank: Rank of the LoRA decomposition. Default: 8
    :type lora_rank: int, optional

    :param lora_alpha: Scaling factor for LoRA weights. Default: 16
    :type lora_alpha: int, optional

    :param lora_dropout: Dropout probability for LoRA layers. Default: 0.0
    :type lora_dropout: float, optional

    :return: The modified model with LoRA adapters applied.
    :rtype: nn.Module

    Example::

        # Apply LoRA to query/key/value projections
        model = Transformer(config)
        apply_lora_to_model(model, target_modules=['qkv_proj'], lora_rank=8, lora_alpha=16)

        # Freeze base model, only train LoRA parameters
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if 'lora_' in name:
                param.requires_grad = True
    """

    def _replace_linear(module: nn.Module, name: str, parent: Optional[nn.Module] = None):
        """Recursively replace matching Linear modules with LoRALinear."""
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child_module, nn.Linear):
                should_wrap = any(pattern in child_name for pattern in target_modules)

                if should_wrap:
                    lora_layer = LoRALinear(
                        child_module,
                        lora_rank=lora_rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )
                    setattr(module, child_name, lora_layer)
            else:
                _replace_linear(child_module, full_name, module)

    _replace_linear(model, "")
    return model
