import math
from typing import Dict, Optional

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

        attn_bias (bool, optional): Whether to use bias in attention Linear Projections. Default: ``False``

        attn_qk_norm (bool, optional): Whether to apply Normalization to Queries and Keys before the Attention Computation. Default: ``True``

        attn_dropout (float, optional): Dropout probability for the Attention Layer. Default: ``0.0``

        ffn_bias (bool, optional): Whether to use bias in Feed-Forward Linear layers. Default: ``True``

        lm_head_bias (bool, optional): Whether to use bias in the Language Modeling Head. Default: ``False``

        tied_weights (bool, optional): If True, tie the input embedding and output projection weights. Default: ``False``

        rope_base (float, optional): Base for the RoPE frequency computation. Default: ``10000.0``

        attn_type (str, optional): Attention type, one of ``MHA``, ``GQA``, ``CrossAttention``. For ``GQA``, also specify `n_kv_heads`. Default: ``MHA``

        n_kv_heads (int, optional): Number of key/value heads for Grouped-Query Attention(GQA). Default: ``n_heads``

        rope_base (float, optinal): Base for the Exponential Frequency Calculation in RoPE. Default: ``10000.0``

        ``**kwargs`` (dict, optional): Additional keyword arguments passed to `PretrainedConfig`

    """

    model_type = "transformer"

    def __init__(
        self,
        n_layers: int = 12,
        n_heads: int = 32,
        n_kv_heads: Optional[int] = None,
        vocab_size: int = 50000,
        d_model: int = 1536,
        d_ff: Optional[int] = None,
        attn_type: str = "MHA",
        attn_bias: Optional[bool] = False,
        attn_dropout: Optional[float] = 0.0,
        ffn_bias: bool = True,
        attn_qk_norm: bool = True,
        lm_head_bias: bool = False,
        tied_weights: bool = False,
        seq_len: int = 1024,
        rope_base: float = 10000.0,
        max_seq_len: int = 4096,
        **kwargs: Dict,
    ):
        super().__init__(**kwargs)
        self.n_layer = n_layers
        self.n_heads = n_heads
        self.attn_type = attn_type
        self.n_kv_heads = n_kv_heads if attn_type == "GQA" else n_heads
        self.vocab_size = vocab_size
        self.attn_bias = attn_bias
        self.attn_dropout = attn_dropout
        self.ffn_bias = ffn_bias
        self.attn_qk_norm = attn_qk_norm
        self.lm_head_bias = lm_head_bias
        self.tied_weights = tied_weights

        self.seq_len = seq_len
        self.rope_base = rope_base
        self.max_seq_len = max_seq_len

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else math.ceil(d_model * 2.666)
