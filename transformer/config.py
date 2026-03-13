import math
from typing import Dict, Optional, Type, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class TransformerConfig(PretrainedConfig):
    r"""
    Configuration class for Transformer models.
    Inherits from `PretrainedConfig` for HuggingFace compatibility.

    Args:

        n_layers (int): Number of Transformer Blocks (layers).

        n_heads (int): Number of Attention Heads.

        vocab_size (int): Vocabulary size of the model. Defines the number of different tokens.

        d_model (int): Model Dimension.

        d_ff (int, optional): Dimension of the Feed-Forward Hidden Layer.

        seq_len (int): Sequence Length.

        max_seq_len (int): Maximum sequence length for positional embeddings.

        ffn_class (Union[Type[nn.Module], str], optional): Feed-Forward Network class or type.
            - If ``str``, one of ``SwiGLU``, ``MLP``.
            - If ``Type[nn.Module]`` then will beinstantiated inside the model.
              Should have the same API as ``SwiGLU`` and ``MLP``.
              Default ``SwiGLU``

        attn_bias (bool, optional): Whether to use bias in attention Linear Projections. Default: ``False``

        attn_qk_norm (bool, optional): Whether to apply Normalization to Queries and Keys before the Attention Computation. Default: ``True``

        norm_class (Union[Type[nn.Module], str], optional): Normalization class or type.
            - If ``str``, one of ``rms_norm`` or ``layer_norm``.
            - If ``Type[nn.Module]`` then will be instantiated inside the model.
              Should have the same API as a torch Normalization Layer.
              Default: ``rms_norm``

        norm_design (str, optional): Normalization Design, one of ``pre-norm``, ``post-norm`` or ``both``. Default: ``pre-norm``

        attn_dropout (float, optional): Dropout probability for the Attention Layer. Default: ``0.0``

        ffn_bias (bool, optional): Whether to use bias in Feed-Forward Linear layers. Default: ``True``

        lm_head_bias (bool, optional): Whether to use bias in the Language Modeling Head. Default: ``False``

        tied_weights (bool, optional): If True, tie the input embedding and output projection weights. Default: ``False``

        attn_class (Union[Type[nn.Module], str], optional): Attention class or type.
            - If ``str``, one of ``MHA``, ``GQA``, ``CrossAttention``. For ``GQA``, also specify `n_kv_heads`.
            - If ``Type[nn.Module]`` then will beinstantiated inside the model.
              Should have the same API as ``transformer.attn.MHA``.
              Default ``MHA``

        n_kv_heads (int, optional): Number of key/value heads for Grouped-Query Attention(GQA). Default: ``n_heads``

        pos_encoding (str, optional): Positional Encoding for attention, one of ``RoPE``, ``AliBI``, ``PartialRoPE``. Default: ``RoPE``
            Note: Is recommended to change the default to ``PartialRoPE`` which is used in SOTA models like Qwen3-Next-80B-A3B

        rope_base (float, optinal): Base for the Exponential Frequency Calculation in RoPE. Default: ``10000.0``

        ``**kwargs`` (dict, optional): Additional keyword arguments passed to `PretrainedConfig`

    """

    model_type = "transformer"

    def __init__(
        self,
        n_layers: int = 12,
        d_model: int = 1536,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        vocab_size: int = 50000,
        d_ff: Optional[int] = None,
        norm_design: str = "pre_norm",
        norm_class: Union[Type[nn.Module], str] = "rms_norm",
        ffn_class: Union[Type[nn.Module], str] = "SwiGLU",
        attn_class: Union[Type[nn.Module], str] = "MHA",
        attn_bias: Optional[bool] = False,
        ffn_bias: bool = True,
        lm_head_bias: bool = False,
        attn_qk_norm: bool = True,
        attn_dropout: Optional[float] = 0.0,
        tied_weights: bool = False,
        seq_len: int = 1024,
        pos_encoding: str = "RoPE",
        rope_base: float = 10000.0,
        max_seq_len: int = 4096,
        **kwargs: Dict,
    ):
        super().__init__(**kwargs)

        self.n_layer = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if attn_class == "GQA" else n_heads
        self.vocab_size = vocab_size

        self.attn_class = attn_class
        self.ffn_class = ffn_class
        self.norm_class = norm_class

        self.norm_design = norm_design

        self.d_ff = d_ff if d_ff is not None else ((math.ceil(d_model * 8 / 3) + 1) // 2) * 2

        self.attn_dropout = attn_dropout
        self.attn_qk_norm = attn_qk_norm

        self.attn_bias = attn_bias
        self.ffn_bias = ffn_bias
        self.lm_head_bias = lm_head_bias

        self.tied_weights = tied_weights

        self.seq_len = seq_len
        self.pos_encoding = pos_encoding
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len
