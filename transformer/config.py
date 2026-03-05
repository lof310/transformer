from transformers import PretrainedConfig

class TransformerConfig(PretrainedConfig):
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
