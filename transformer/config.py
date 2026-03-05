from transformers import PretrainedConfig

class TransformerConfig(PretrainedConfig):
    """
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
        attn_bias (bool): Whether to use bias in attention Linear Projections.
        attn_qk_norm (bool): Whether to apply Normalization to Queries and Keys before the Attention Computation.
        ffn_bias (bool): Whether to use bias in Feed-Forward Linear layers.
        lm_head_bias (bool): Whether to use bias in the Language Modeling Head.
        tied_weights (bool): If True, tie the input embedding and output projection weights.
        rope_base (float, optional): Base for the RoPE frequency computation.
        attn_type (str): Attention type, one of "MHA", "GQA". For GQA, also specify `n_kv_heads`. Default: "MHA"
        n_kv_heads (int, optional): Number of key/value heads for Grouped-Query Attention(GQA).
        **kwargs: Additional keyword arguments passed to `PretrainedConfig`.
    """
    model_type = "transformer"

    def __init__(
        self,
        n_layers: int = 12,
        n_heads: int = 32,
        n_kv_heads: int = None,
        vocab_size: int = 50000,
        d_model: int = 1536,
        d_ff: int = None,
        attn: str = "MHA",
        attn_bias: bool = False,
        ffn_bias: bool = True,
        attn_qk_norm: bool = False,
        lm_head_bias: bool = False,
        tied_weights: bool = False,
        seq_len: int = 1024,
        max_seq_len: int = 4096,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.n_layer = n_layers
        self.n_heads = n_heads
        self.attn = attn
        self.n_kv_heads = n_kv_heads if attn == "GQA" else n_heads
        self.vocab_size = vocab_size
        self.attn_bias = attn_bias
        self.ffn_bias = ffn_bias
        self.attn_qk_norm = qk_norm
        self.lm_head_bias = lm_head_bias
        self.tied_weights = tied_weights

        self.seq_len = seq_len
        self.max_seq_len = max_seq_len

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else math.ceil(d_model * 2.666)
