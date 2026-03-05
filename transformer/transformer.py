import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import PreTrainedModel, PretrainedConfig, GenerationMixin
from transformers.modeling_outputs import CausalLMOutput
import math

from .config import TransformerConfig
from .attns import MHA
from .pos import RoPE
from .ffn import SwiGLU

class TransformerBlock(nn.Module):
    """
    A Single Decoder Transformer Block consisting of Multi-Head Attention and Feed-Forward layers,
    each with Pre-Normalization (RMSNorm) and Standard Residual Connections.

    Args:
        config (TransformerConfig): Configuration object.
        layer_idx (int, optional): Index of this block (used for debugging/logging).
    """
    def __init__(self, config, layer_idx: int = 0):
        super().__init__()
        self.d_model, self.d_ff, self.n_heads, self.layer_idx = config.d_model, config.d_ff, config.n_heads, layer_idx

        if config.attn_type == "MHA":
            self.attn = MHA(self.d_model, self.n_heads, attn_bias=config.attn_bias, qk_norm=config.attn_qk_norm, layer_idx=layer_idx, max_seq_len=config.max_seq_len)
        elif config.attn_type == "GQA":
            self.attn = GQA(self.d_model, self.n_heads, config.n_kv_heads, attn_bias=config.attn_bias, qk_norm=config.attn_qk_norm, layer_idx=layer_idx, max_seq_len=config.max_seq_len)
        elif config.attn_type == "CrossAttention":
            raise ValueError(f"Under Development: {config.attn_type}")
        elif config.attn_type == "MQA":
            raise ValueError(f"Under Development: {config.attn_type}")
        else:
            raise ValueError(f"Unknown attention type: {config.attn_type}")
        self.ffn = SwiGLU(self.d_model, self.d_ff, bias=config.ffn_bias)
        self.norm_attn, self.norm_ffn = nn.RMSNorm(self.d_model), nn.RMSNorm(self.d_model)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None, pos: torch.Tensor = None, return_states: bool = False):
        """
        Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, d_model)
            attn_mask (torch.Tensor, optional): Attention mask for the Attention block.
            pos (torch.Tensor, optional): Position indices for Positional Encoding.
            return_states (bool, optional): If True, return a dictionary of intermediate outputs. Default: False

        Returns:
            Union[torch.Tensor, Dict]: Output tensor (batch_size, seq_len, d_model) if not return_states,
                                        else a dict containing Final Output with Attention and FFN Outputs.
        """
        attn = self.attn(self.norm_attn(x), attn_mask, pos, return_states)
        x = x + attn["output"] if return_states else x + attn
        ffn = self.ffn(self.norm_ffn(x), return_states)
        x = x + ffn["output"] if return_states else x + ffn
        if return_states:
            return {"output": x, "attn_output": attn, "ffn_output": ffn}
        else:
            return x

class Transformer(PreTrainedModel):
    """
    Transformer language model, compatible with the HuggingFace interface.

    Args:
        config (TransformerConfig): Model configuration.
    """
    config_class = TransformerConfig
    base_model_prefix = "transformer"

    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model

        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config, i) for i in range(config.n_layer)])
        self.norm_out = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.lm_head_bias)

        self.emb.weight.data.normal_(mean=0.0, std=0.025)

        if config.tied_weights:
            self.lm_head.weight = self.emb.weight
        else:
            self.lm_head.weight.data.normal_(mean=0.0, std=0.025)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor = None, is_causal: bool = True, attn_mask: torch.Tensor = None, pos: torch.Tensor = None, return_states=False, **kwargs):
        """
        Forward pass of the Transformer model.

        Args:
            input_ids (torch.Tensor): Token indices of shape (batch, seq_len)
            labels (torch.Tensor, optional): Target token indices for loss computation, same shape as input_ids.
            is_causal (bool, optional): If True, create a causal attention mask. Default: True
            attn_mask (torch.Tensor, optional): Custom attention mask. If None and is_causal, a upper triangular causal mask is generated.
            pos (torch.Tensor, optional): Position indices. If None, uses torch.arange(seq_len).
            return_states (bool, optional): If True, return hidden states of all layers. Default: False

        Returns:
            CausalLMOutput: Contains loss (if labels given), logits, and optionally hidden states.
        """
        B, N = input_ids.shape

        input_embs = self.emb(input_ids)
        out = input_embs
        attn_mask, pos = (
            torch.triu(torch.ones(N, N, device=out.device), diagonal=1).bool() if attn_mask is None and is_causal else attn_mask,
            torch.arange(N, device=out.device) if pos is None else pos
        )
        hidden_states = [] if return_states else None

        for block in self.blocks:
            output_dict = block(out, attn_mask, pos, return_states)
            out = output_dict["output"] if return_states else output_dict
            if return_states:
                hidden_states.append(out)

        logits = self.lm_head(self.norm_out(out))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-1
            )
        return CausalLMOutput(loss=loss, logits=logits, hidden_states=(input_embs, hidden_states) if return_states else None)

    def get_num_params(self):
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
