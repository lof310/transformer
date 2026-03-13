import math
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutput

from .attns import MHA
from .config import TransformerConfig
from .ffn import SwiGLU
from .pos import RoPE
from .utils import check_type


class TransformerBlock(GradientCheckpointingLayer):
    """
    A Single Transformer Decoder Block consisting of Multi-Head Attention and Feed-Forward layers,
    each with Pre-Normalization (RMSNorm) and Standard Residual Connections.

    Args:
        config (TransformerConfig): Configuration object.
        attn_kwargs: (Dict, optional): Additional Arguments for the attention class passed from ``TransformerConfig.attn_class``.
            It is only used if ``TransformerConfig.attn_class`` is ``Type[nn.Module]``
        ffn_kwargs: (Dict, optional): Additional Arguments for the ffn class passed from ``TransformerConfig.ffn_class``.
            It is only used if ``TransformerConfig.ffn_class`` is ``Type[nn.Module]``
        norm_kwargs: (Dict, optional): Additional Arguments for the normalization class passed from ``TransformerConfig.norm_class``. It is always passed.
        layer_idx (int, optional): Index of this block (used for debugging/logging).

    """

    def __init__(
        self,
        config,
        attn_kwargs: Optional[Dict] = {},
        ffn_kwargs: Optional[Dict] = {},
        norm_kwargs: Optional[Dict] = {},
        layer_idx: int = 0,
    ):
        super().__init__()
        self.d_model, self.d_ff, self.n_heads, self.layer_idx = config.d_model, config.d_ff, config.n_heads, layer_idx
        self.norm_design = config.norm_design

        if config.attn_class == "MHA":
            self.attn = MHA(
                self.d_model,
                self.n_heads,
                dropout=config.attn_dropout,
                attn_bias=config.attn_bias,
                qk_norm=config.attn_qk_norm,
                layer_idx=layer_idx,
                pos_encoding=config.pos_encoding,
                max_seq_len=config.max_seq_len,
                **attn_kwargs,
            )
        elif config.attn_class == "GQA":
            self.attn = GQA(
                self.d_model,
                self.n_heads,
                n_kv_heads=config.n_kv_heads,
                dropout=config.attn_dropout,
                attn_bias=config.attn_bias,
                qk_norm=config.attn_qk_norm,
                layer_idx=layer_idx,
                pos_encoding=config.pos_encoding,
                max_seq_len=config.max_seq_len,
                **attn_kwargs,
            )
        elif config.attn_class == "CrossAttention":
            raise ValueError(f"Under Development: {config.attn_class}")
        elif check_type(config.attn_class) == 0:
            raise ValueError(f"Unknown attention type: {config.attn_class}")
        elif check_type(config.attn_class) == 1:
            self.attn = config.attn_class(
                self.d_model,
                self.n_heads,
                config.attn_bias,
                **attn_kwargs,
            )
        else:
            raise RuntimeError(
                "TransformerConfig.attn_class Should be str or Type[nn.Module] but found: {config.attn_class}"
            )

        if config.ffn_class == "SwiGLU":
            self.ffn = SwiGLU(self.d_model, self.d_ff, bias=config.ffn_bias, **ffn_kwargs)
        elif config.ffn_class == "MLP":
            self.ffn = MLP(self.d_model, self.d_ff, bias=config.ffn_bias, **ffn_kwargs)
        elif config.ffn_class == "MoE":
            raise ValueError(f"Under Development: {config.ffn_class}")
        elif check_type(config.ffn_class) == 0:
            raise ValueError(f"Unknown ffn class: {config.ffn_class}")
        elif check_type(config.ffn_class) == 1:
            self.ffn = config.ffn_class(self.d_model, self.d_ff, bias=config.ffn_bias, **ffn_kwargs)
        else:
            raise RuntimeError(
                "TransformerConfig.ffn_class Should be str or Type[nn.Module] but found: {config.ffn_class}"
            )

        if config.norm_class == "rms_norm":
            if config.norm_design == "pre_norm" or config.norm_design == "post_norm":
                self.norm_attn, self.norm_ffn = (
                    nn.RMSNorm(self.d_model, **norm_kwargs),
                    nn.RMSNorm(self.d_model, **norm_kwargs),
                )
            elif config.norm_design == "both":
                self.pre_norm_attn, self.pre_norm_ffn, self.post_norm_attn, self.post_norm_ffn = (
                    nn.RMSNorm(self.d_model, **norm_kwargs),
                    nn.RMSNorm(self.d_model, **norm_kwargs),
                    nn.RMSNorm(self.d_model, **norm_kwargs),
                    nn.RMSNorm(self.d_model, **norm_kwargs),
                )
            else:
                raise ValueError(f"Invalid norm_design: {config.norm_design}")
        elif config.norm_class == "layer_norm":
            if config.norm_design == "pre_norm" or config.norm_design == "post_norm":
                self.norm_attn, self.norm_ffn = (
                    nn.LayerNorm(self.d_model, **norm_kwargs),
                    nn.LayerNorm(self.d_model, **norm_kwargs),
                )
            elif config.norm_design == "both":
                self.pre_norm_attn, self.pre_norm_ffn, self.post_norm_attn, self.post_norm_ffn = (
                    nn.LayerNorm(self.d_model, **norm_kwargs),
                    nn.LayerNorm(self.d_model, **norm_kwargs),
                    nn.LayerNorm(self.d_model, **norm_kwargs),
                    nn.LayerNorm(self.d_model, **norm_kwargs),
                )
            else:
                raise ValueError(f"Invalid norm_design: {config.norm_design}")
        elif check_type(config.norm_class) == 0:
            raise ValueError(f"Unknown normalization class: {config.norm_class}")
        elif check_type(config.norm_class) == 1:
            if config.norm_design == "pre_norm" or config.norm_design == "post_norm":
                self.norm_attn, self.norm_ffn = (
                    config.norm_class(self.d_model, **norm_kwargs),
                    config.norm_class(self.d_model, **norm_kwargs),
                )
            elif config.norm_design == "both":
                self.pre_norm_attn, self.pre_norm_ffn, self.post_norm_attn, self.post_norm_ffn = (
                    config.norm_class(self.d_model, **norm_kwargs),
                    config.norm_class(self.d_model, **norm_kwargs),
                    config.norm_class(self.d_model, **norm_kwargs),
                    config.norm_class(self.d_model, **norm_kwargs),
                )
            else:
                raise ValueError(f"Invalid norm_design: {config.norm_design}")
        else:
            raise RuntimeError(
                "TransformerConfig.norm_class Should be str or Type[nn.Module] but found: {config.norm_class}"
            )

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool] = (
            False,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            False,
        ),
        return_states: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict]:
        r"""
        Forward pass of the transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape :math:`(B, N, D)`.
            attn_mask (torch.Tensor, optional): Attention mask for the Attention block.
            pos (torch.Tensor, optional): Position indices for Positional Encoding.
            flash_attn (Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional): Tuple of Arguments for Flash Attention.
            return_states (bool, optional): If True, return a dictionary of intermediate outputs. Default: False

        Returns:
            Union[torch.Tensor, Dict]: Output tensor (batch_size, seq_len, d_model) if not return_states,
                else a dict containing the keys: "output", "attn_output" and "ffn_output".

        """

        def extract(out):
            """Helper to extract output tensor from module return (handles dict vs tensor)"""
            return out["output"] if return_states else out

        attn, ffn = None, None
        if self.norm_design == "pre_norm":
            attn = self.attn(
                self.norm_attn(x),
                return_states=return_states,
                **{"mask": attn_mask, "pos": pos, "flash_attn": flash_attn},
            )
            x = x + extract(attn)

            ffn = self.ffn(self.norm_ffn(x), return_states=return_states)
            x = x + extract(ffn)
        elif self.norm_design == "post_norm":
            attn = self.attn(
                x, return_states=return_states, **{"mask": attn_mask, "pos": pos, "flash_attn": flash_attn}
            )
            x = self.norm_attn(x + extract(attn))

            ffn = self.ffn(x, return_states=return_states)
            x = self.norm_ffn(x + extract(ffn))
        elif self.norm_design == "both":
            attn = self.attn(
                self.pre_norm_attn(x),
                return_states=return_states,
                **{"mask": attn_mask, "pos": pos, "flash_attn": flash_attn},
            )
            x = self.post_norm_attn(x + extract(attn))

            ffn = self.ffn(self.pre_norm_ffn(x), return_states=return_states)
            x = self.post_norm_ffn(x + extract(ffn))
        else:
            raise ValueError(f"Invalid norm_design: {self.norm_design}")

        if return_states:
            return {"output": x, "attn_output": attn, "ffn_output": ffn}
        else:
            return x


class Transformer(PreTrainedModel, GenerationMixin):
    r"""
    Transformer language model, compatible with the HuggingFace interface.

    Args:
        config (TransformerConfig): Model configuration.

        attn_kwargs (Dict, optional): Additional Keyword Arguments passed to the Attention Module. Default: ``{"pos_encoding_kwargs": **pos_encoding_kwargs}``

        pos_encoding_kwargs (Dict, optional): Additional Arguments for Positional Encoding. Default: ``{}``
            Example: ``{"rope_base": 12000, "persistent": False}``

        ffn_kwargs (Dict, optional): Additional Keyword Arguments passed to the Feed-Forward Module. Default: ``{}``

        norm_kwargs (Dict, optional): Additional Keyword Arguments passed to the Normalization Layer. Default: ``{}``

    """

    config_class = TransformerConfig
    base_model_prefix = "transformer"

    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    input_modalities = "text"  # Will add "image" for v0.4.0

    def __init__(
        self,
        config,
        attn_kwargs: Dict = {},
        pos_encoding_kwargs: Dict = {},
        ffn_kwargs: Dict = {},
        norm_kwargs: Dict = {},
    ):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model

        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    config,
                    (
                        attn_kwargs
                        if attn_kwargs != {}
                        else {
                            "pos_encoding_kwargs": (
                                pos_encoding_kwargs if pos_encoding_kwargs != {} else {"rope_base": config.rope_base}
                            )
                        }
                    ),
                    ffn_kwargs,
                    norm_kwargs,
                    i,
                )
                for i in range(config.n_layer)
            ]
        )
        self.norm_out = nn.RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.lm_head_bias)

        if config.tied_weights:
            self.lm_head.weight = self.emb.weight
        else:
            self.lm_head.weight.data.normal_(mean=0.0, std=0.025)

        self.post_init()

    @torch.no_grad()
    def _init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.025)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.025)

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: torch.LongTensor = None,
        is_causal: Optional[bool] = True,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool] = (
            False,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            False,
        ),
        return_states: Optional[bool] = False,
        loss_kwargs: Dict = {},
        **kwargs: Dict,
    ) -> CausalLMOutput:
        """
        Forward pass of the Transformer model.

        Args:
            input_ids (torch.Tensor): Token indices of shape :math:`(B, N)`
            labels (torch.Tensor, optional): Target token indices for loss computation, same shape as input_ids.
            is_causal (bool, optional): If True, create a causal attention mask. Default: True
            attn_mask (torch.Tensor, optional): Custom attention mask. If None and is_causal, a upper triangular causal mask is generated.
            pos (torch.Tensor, optional): Position indices. If None, uses ``torch.arange(N)``.
            flash_attn (Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional): Tuple of Arguments for Flash Attention.
            return_states (bool, optional): If True, return hidden states of all layers. Default: False
            ``**loss_kwargs`` (dict, optional): Additional keyword arguments passed to `F.cross_entropy` for loss computation.
            ``**kwargs`` (dict, optional): Additional keyword arguments

        Returns:
            CausalLMOutput: Contains loss (if labels given else None), logits, and optionally hidden states being a tuple of (input_embs, hidden_states)
                where `hidden_states` is a list of dictionaries for the output of each layer.
        """
        B, N = input_ids.shape

        input_embs = self.emb(input_ids)
        out = input_embs
        attn_mask, pos = (
            (
                torch.triu(torch.ones(N, N, device=out.device), diagonal=1).bool()
                if attn_mask is None and is_causal
                else attn_mask
            ),
            torch.arange(N, device=out.device) if pos is None else pos,
        )
        hidden_states = [] if return_states else None

        for block in self.blocks:
            output_dict = block(out, attn_mask, pos, flash_attn=flash_attn, return_states=return_states)
            out = output_dict["output"] if return_states else output_dict
            if return_states:
                hidden_states.append(output_dict)

        logits = self.lm_head(self.norm_out(out))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), **loss_kwargs)

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=(input_embs, hidden_states) if return_states else None
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.emb

    def set_input_embeddings(self, embeddings: nn.Embedding):
        self.emb = embeddings

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
