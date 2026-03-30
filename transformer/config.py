import math
from typing import Dict, List, Optional, Type, Union

import torch
import torch.nn as nn
from transformers import PretrainedConfig


class TransformerConfig(PretrainedConfig):
    r"""
    Configuration class for Transformer models.
    Inherits from `PretrainedConfig` for HuggingFace compatibility.

    :param n_layers: Number of Transformer Blocks (layers).
    :type n_layers: int

    :param d_model: Model Dimension.
    :type d_model: int

    :param n_heads: Number of Attention Heads.
    :type n_heads: int

    :param n_kv_heads: Number of key/value heads for Grouped-Query Attention(GQA). Default: ``n_heads``
    :type n_kv_heads: int, optional

    :param vocab_size: Vocabulary size of the model. Defines the number of different tokens.
    :type vocab_size: int

    :param d_ff: Dimension of the Feed-Forward Hidden Layer.
    :type d_ff: int, optional

    :param norm_design: Normalization Design, one of ``pre-norm``, ``post-norm`` or ``both``. Default: ``pre-norm``
    :type norm_design: str

    :param norm_class: Normalization class or type.
        - If ``str``, one of ``rms_norm`` or ``layer_norm``.
        - If ``Type[nn.Module]`` then will be instantiated inside the model.
          Should have the same API as a torch Normalization Layer.
        - If ``List[Union[Type[nn.Module], str]]`` and len(norm_class) == n_layers
          then will be instantiated inside the model for the corresponding layers.
    :type norm_class: Union[List[Union[Type[nn.Module], str]], Type[nn.Module], str]

    :param ffn_class: Feed-Forward Network class or type.
        - If ``str``, one of ``SwiGLU``, ``MLP``.
        - If ``Type[nn.Module]`` then will be instantiated inside the model.
          Should have the same API as ``SwiGLU`` and ``MLP``.
          Default ``SwiGLU``
        - If ``List[Union[Type[nn.Module], str]]`` and len(ffn_class) == n_layers
          then will be instantiated inside the model for the corresponding layers.
          Default ``SwiGLU`` for every layer.
    :type ffn_class: Union[List[Union[Type[nn.Module], str]], Type[nn.Module], str]

    :param attn_class: Attention class or type.
        - If ``str``, one of ``MHA``, ``GQA``, ``CrossAttention``. For ``GQA``, also specify `n_kv_heads`.
        - If ``Type[nn.Module]`` then will be instantiated inside the model.
          Should have the same API as ``transformer.attn.MHA``.
          Default ``MHA``
        - If ``List[Union[Type[nn.Module], str]]`` and len(attn_class) == n_layers
          then will be instantiated inside the model for the corresponding layers.
          Default ``MHA`` for every layer.
    :type attn_class: Union[List[Union[Type[nn.Module], str]], Type[nn.Module], str]

    :param block_class: Transformer Block class for every layer. Default: ``None``
        - If ``Type[nn.Module]`` then will be instantiated for every layer inside the model.
        - If ``None`` then the default ``transformer.TransformerBlock`` will be used
    :type block_class: Optional[Type[nn.Module]]

    :param attn_bias: Whether to use bias in attention Linear Projections. Default: ``False``
    :type attn_bias: bool, optional

    :param ffn_bias: Whether to use bias in Feed-Forward Linear layers. Default: ``True``
    :type ffn_bias: bool, optional

    :param lm_head_bias: Whether to use bias in the Language Modeling Head. Default: ``False``
    :type lm_head_bias: bool, optional

    :param attn_qk_norm: Whether to apply Normalization to Queries and Keys before the Attention Computation. Default: ``True``
    :type attn_qk_norm: bool, optional

    :param attn_dropout: Dropout probability for the Attention Layer. Default: ``0.0``
    :type attn_dropout: float, optional

    :param tied_weights: If True, tie the input embedding and output projection weights. Default: ``False``
    :type tied_weights: bool, optional

    :param seq_len: Sequence Length.
    :type seq_len: int

    :param pos_encoding: Positional Encoding for attention.
        - If ``str`` one of ``RoPE``, ``AliBI``, ``PartialRoPE``. Default: ``RoPE``
        Note: Is recommended to change the default to ``PartialRoPE`` which is used in SOTA models like Qwen3-Next-80B-A3B
        - If ``List[str]`` and len(pos_encoding) == n_layers, applies different positional encodings per layer.
    :type pos_encoding: Union[List[str], str]

    :param rope_base: Base for the Exponential Frequency Calculation in RoPE. Default: ``10000.0``
    :type rope_base: float, optional

    :param max_seq_len: Maximum sequence length for positional embeddings.
    :type max_seq_len: int

    :param use_cache: Whether to use KV cache during generation. Default: ``True``
    :type use_cache: bool, optional

    :param is_decoder: Whether this is a decoder model. Default: ``True``
    :type is_decoder: bool, optional

    :param kwargs: Additional keyword arguments passed to `PretrainedConfig`
    :type kwargs: dict, optional

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
        norm_class: Union[List[Union[Type[nn.Module], str]], Type[nn.Module], str] = "rms_norm",
        ffn_class: Union[List[Union[Type[nn.Module], str]], Type[nn.Module], str] = "SwiGLU",
        attn_class: Union[List[Union[Type[nn.Module], str]], Type[nn.Module], str] = "MHA",
        block_class: Optional[Type[nn.Module]] = None,
        attn_bias: bool = False,
        ffn_bias: bool = True,
        lm_head_bias: bool = False,
        attn_qk_norm: bool = True,
        attn_dropout: Optional[float] = 0.0,
        tied_weights: bool = False,
        seq_len: int = 1024,
        pos_encoding: Union[List[str], str] = "RoPE",
        rope_base: float = 10000.0,
        max_seq_len: int = 4096,
        use_cache: bool = True,
        is_decoder: bool = True,
        **kwargs: Dict,
    ):
        super().__init__(**kwargs)

        self.n_layer = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads if n_kv_heads is not None else n_heads
        self.vocab_size = vocab_size

        self.attn_class = attn_class
        self.ffn_class = ffn_class
        self.norm_class = norm_class
        self.block_class = block_class

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

        self.use_cache = use_cache
        self.is_decoder = is_decoder
