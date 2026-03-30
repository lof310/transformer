import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutput

from .attns import GQA, MHA, CrossAttention
from .config import TransformerConfig
from .ffn import MLP, SwiGLU
from .pos import ALiBi, PartialRoPE, RoPE
from .utils import check_type, resolve_layer_config


class TransformerBlock(GradientCheckpointingLayer):
    """
    A Single Transformer Decoder Block with support for Gradient Checkpointing consisting of Multi-Head Attention and Feed-Forward layers,
    each with Pre-Normalization (RMSNorm) and Standard Residual Connections.

    :param config: Configuration object.
    :type config: TransformerConfig

    :param attn_kwargs: Additional Arguments for the attention class passed from ``TransformerConfig.attn_class``.
        It is only used if ``TransformerConfig.attn_class`` is ``Type[nn.Module]``
    :type attn_kwargs: Dict, optional

    :param ffn_kwargs: Additional Arguments for the ffn class passed from ``TransformerConfig.ffn_class``.
        It is only used if ``TransformerConfig.ffn_class`` is ``Type[nn.Module]``
    :type ffn_kwargs: Dict, optional

    :param norm_kwargs: Additional Arguments for the normalization class passed from ``TransformerConfig.norm_class``. It is always passed.
    :type norm_kwargs: Dict, optional

    :param layer_idx: Index of this block (used for debugging/logging).
    :type layer_idx: int, optional
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
        self.n_layers = config.n_layer

        # Resolve per-layer configurations
        attn_class = resolve_layer_config(config.attn_class, layer_idx, self.n_layers)
        ffn_class = resolve_layer_config(config.ffn_class, layer_idx, self.n_layers)
        norm_class = resolve_layer_config(config.norm_class, layer_idx, self.n_layers)
        pos_encoding = (
            resolve_layer_config(config.pos_encoding, layer_idx, self.n_layers)
            if isinstance(config.pos_encoding, list)
            else config.pos_encoding
        )

        # Create attention module
        if attn_class == "MHA":
            self.attn = MHA(
                self.d_model,
                self.n_heads,
                dropout=config.attn_dropout,
                attn_bias=config.attn_bias,
                qk_norm=config.attn_qk_norm,
                layer_idx=layer_idx,
                pos_encoding=pos_encoding,
                max_seq_len=config.max_seq_len,
                **attn_kwargs,
            )
        elif attn_class == "GQA":
            self.attn = GQA(
                self.d_model,
                self.n_heads,
                n_kv_heads=config.n_kv_heads,
                dropout=config.attn_dropout,
                attn_bias=config.attn_bias,
                qk_norm=config.attn_qk_norm,
                layer_idx=layer_idx,
                pos_encoding=pos_encoding,
                max_seq_len=config.max_seq_len,
                **attn_kwargs,
            )
        elif attn_class == "CrossAttention":
            self.attn = CrossAttention(
                self.d_model,
                self.n_heads,
                dropout=config.attn_dropout,
                attn_bias=config.attn_bias,
                qk_norm=config.attn_qk_norm,
                layer_idx=layer_idx,
                rope_base=config.rope_base,
                max_seq_len=config.max_seq_len,
            )
        elif check_type(attn_class) == 1:
            self.attn = attn_class(
                self.d_model,
                self.n_heads,
                dropout=config.attn_dropout,
                attn_bias=config.attn_bias,
                qk_norm=config.attn_qk_norm,
                layer_idx=layer_idx,
                max_seq_len=config.max_seq_len,
                pos_encoding=pos_encoding,
                **attn_kwargs,
            )
        else:
            raise ValueError(f"Unknown attention type: {attn_class}")

        # Create feed-forward module
        if ffn_class == "SwiGLU":
            self.ffn = SwiGLU(self.d_model, self.d_ff, bias=config.ffn_bias, **ffn_kwargs)
        elif ffn_class == "MLP":
            self.ffn = MLP(self.d_model, self.d_ff, bias=config.ffn_bias, **ffn_kwargs)
        elif check_type(ffn_class) == 1:
            self.ffn = ffn_class(self.d_model, self.d_ff, bias=config.ffn_bias, **ffn_kwargs)
        else:
            raise ValueError(f"Unknown ffn class: {ffn_class}")

        # Create normalization modules
        if norm_class == "rms_norm":
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
        elif norm_class == "layer_norm":
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
        elif check_type(norm_class) == 1:
            if config.norm_design == "pre_norm" or config.norm_design == "post_norm":
                self.norm_attn, self.norm_ffn = (
                    norm_class(self.d_model, **norm_kwargs),
                    norm_class(self.d_model, **norm_kwargs),
                )
            elif config.norm_design == "both":
                self.pre_norm_attn, self.pre_norm_ffn, self.post_norm_attn, self.post_norm_ffn = (
                    norm_class(self.d_model, **norm_kwargs),
                    norm_class(self.d_model, **norm_kwargs),
                    norm_class(self.d_model, **norm_kwargs),
                    norm_class(self.d_model, **norm_kwargs),
                )
            else:
                raise ValueError(f"Invalid norm_design: {config.norm_design}")
        else:
            raise ValueError(f"Unknown normalization class: {norm_class}")

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

        :param x: Input tensor of shape :math:`(B, N, D)`.
        :type x: torch.Tensor

        :param attn_mask: Attention mask for the Attention block.
        :type attn_mask: torch.Tensor, optional

        :param pos: Position indices for Positional Encoding.
        :type pos: torch.Tensor, optional

        :param flash_attn: Tuple of Arguments for Flash Attention.
        :type flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional

        :param return_states: If True, return a dictionary of intermediate outputs. Default: False
        :type return_states: bool, optional

        :return: Output tensor (batch_size, seq_len, d_model) if not return_states,
            else a dict containing the keys: "output", "attn_output" and "ffn_output".
        :rtype: Union[torch.Tensor, Dict]
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

    :param config: Model configuration.
    :type config: TransformerConfig

    :param attn_kwargs: Additional Keyword Arguments passed to the Attention Module. Default: ``{"pos_encoding_kwargs": **pos_encoding_kwargs}``
    :type attn_kwargs: Dict, optional

    :param pos_encoding_kwargs: Additional Arguments for Positional Encoding. Default: ``{}``
        Example: ``{"rope_base": 12000, "persistent": False}``
    :type pos_encoding_kwargs: Dict, optional

    :param ffn_kwargs: Additional Keyword Arguments passed to the Feed-Forward Module. Default: ``{}``
    :type ffn_kwargs: Dict, optional

    :param norm_kwargs: Additional Keyword Arguments passed to the Normalization Layer. Default: ``{}``
    :type norm_kwargs: Dict, optional

    :param patch_size: Patch size for Vision Transformer (ViT) compatibility. If specified, adds a patch embedding layer.
    :type patch_size: Optional[int], optional

    :param img_size: Image size for ViT compatibility. Used with patch_size to compute number of patches.
    :type img_size: Optional[Union[int, Tuple[int, int]]], optional

    :param num_channels: Number of input channels for ViT. Default: 3 (RGB).
    :type num_channels: int, optional
    """

    config_class = TransformerConfig
    base_model_prefix = "transformer"

    supports_gradient_checkpointing = True
    _supports_flash_attn = True
    _supports_sdpa = True

    input_modalities = ["text", "image"]

    def __init__(
        self,
        config,
        attn_kwargs: Dict = {},
        pos_encoding_kwargs: Dict = {},
        ffn_kwargs: Dict = {},
        norm_kwargs: Dict = {},
        patch_size: Optional[int] = None,
        img_size: Optional[Union[int, Tuple[int, int]]] = None,
        num_channels: int = 3,
    ):
        super().__init__(config)
        self.config = config
        self.d_model = config.d_model
        self.patch_size = patch_size
        self.img_size = img_size

        # Vision Transformer (ViT) support
        if patch_size is not None and img_size is not None:
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            self.num_patches = (img_size[0] // patch_size) * (img_size[1] // patch_size)
            self.patch_embed = nn.Conv2d(num_channels, config.d_model, kernel_size=patch_size, stride=patch_size)
            self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, config.d_model))
        else:
            self.patch_embed = None
            self.cls_token = None
            self.pos_embed = None
            self.num_patches = None

        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        block_class = config.block_class if config.block_class is not None else TransformerBlock

        self.blocks = nn.ModuleList(
            [
                block_class(
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

        # Initialize ViT-specific parameters
        if self.patch_embed is not None:
            nn.init.normal_(self.cls_token, std=0.02)
            nn.init.normal_(self.pos_embed, std=0.02)

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

        :param input_ids: Token indices of shape :math:`(B, N)`
        :type input_ids: torch.LongTensor

        :param labels: Target token indices for loss computation, same shape as input_ids.
        :type labels: torch.LongTensor, optional

        :param is_causal: If True, create a causal attention mask. Default: True
        :type is_causal: bool, optional

        :param attn_mask: Custom attention mask. If None and is_causal, a upper triangular causal mask is generated.
        :type attn_mask: torch.Tensor, optional

        :param pos: Position indices. If None, uses ``torch.arange(N)``.
        :type pos: torch.Tensor, optional

        :param flash_attn: Tuple of Arguments for Flash Attention.
        :type flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional

        :param return_states: If True, return hidden states of all layers. Default: False
        :type return_states: bool, optional

        :param loss_kwargs: Additional keyword arguments passed to `F.cross_entropy` for loss computation.
        :type loss_kwargs: Dict, optional

        :param kwargs: Additional keyword arguments
        :type kwargs: Dict, optional

        :return: Contains loss (if labels given else None), logits, and optionally hidden states being a tuple of (input_embs, hidden_states)
            where `hidden_states` is a list of dictionaries for the output of each layer.
        :rtype: CausalLMOutput
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
