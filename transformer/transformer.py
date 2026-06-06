"""Core Transformer model implementation."""

import math
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import \
    CausalLMOutputWithPast as CausalLMOutput

from .attns import GQA, MHA, CrossAttention
from .config import TransformerConfig
from .ffn import MLP, SwiGLU
from .pos import ALiBi, PartialRoPE, RoPE
from .utils import LayerType, get_layer_type, resolve_layer_config


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

        # Resolve per-layer configurations using resolve_layer_config which handles both uniform and per-layer configs
        attn_class = resolve_layer_config(config.attn_class, layer_idx, self.n_layers)
        ffn_class = resolve_layer_config(config.ffn_class, layer_idx, self.n_layers)
        norm_class = resolve_layer_config(config.norm_class, layer_idx, self.n_layers)
        pos_encoding = (
            resolve_layer_config(config.pos_encoding, layer_idx, self.n_layers)
            if isinstance(config.pos_encoding, list)
            else config.pos_encoding
        )

        # Check for CrossAttention incompatibility with norm_design='both'
        if attn_class == "CrossAttention" and config.norm_design == "both":
            raise ValueError(
                "norm_design='both' is not compatible with CrossAttention because CrossAttention expects (queries, kv) inputs while 'both' assumes x input for residual connections."
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
                max_seq_len=config.max_seq_len,
                pos_encoding=pos_encoding,
                pos_encoding_kwargs=attn_kwargs.get("pos_encoding_kwargs", {}),
            )
        elif get_layer_type(attn_class) == LayerType.NN_MODULE:
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
        elif get_layer_type(ffn_class) == LayerType.NN_MODULE:
            self.ffn = ffn_class(self.d_model, self.d_ff, bias=config.ffn_bias, **ffn_kwargs)
        else:
            raise ValueError(f"Unknown ffn class: {ffn_class}")

        # Determine if this block has cross-attention (only relevant for encoder-decoder models)
        self.has_cross_attention = isinstance(self.attn, CrossAttention)

        # Validate norm_design compatibility with CrossAttention
        if self.has_cross_attention and config.norm_design == "both":
            raise ValueError(
                f"norm_design='both' is not compatible with CrossAttention in layer {layer_idx}. "
                f"CrossAttention expects (queries, kv) inputs, not residual connections with x. "
                f"Use norm_design='pre_norm' or 'post_norm' instead."
            )

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
        elif get_layer_type(norm_class) == LayerType.NN_MODULE:
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
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attn_mask: Optional[torch.Tensor] = None,
        encoder_pos: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Dict]:
        r"""
        Forward pass of the transformer block.

        :param x: Input tensor of shape :math:`(B, N, D)`.
        :type x: torch.Tensor

        :param attn_mask: Attention mask for the Attention block.
        :type attn_mask: torch.Tensor, optional

        :param pos: Position indices for Positional Encoding (RoPE, PartialRoPE, etc.).
            Shape :math:`(N,)` or :math:`(B, N)`.
        :type pos: torch.Tensor, optional

        :param flash_attn: Tuple of Arguments for Flash Attention.
        :type flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional

        :param return_states: If True, return a dictionary of intermediate outputs. Default: False
        :type return_states: bool, optional

        :param cache: Optional KV cache tuple `(k_prev, v_prev)` for incremental decoding.
            Only used by MHA and GQA attention types.
        :type cache: Tuple[torch.Tensor, torch.Tensor], optional

        :param encoder_hidden_states: Encoder output tensor of shape :math:`(B, L_{enc}, D)`.
            Only used when this block has CrossAttention.
        :type encoder_hidden_states: torch.Tensor, optional

        :param encoder_attn_mask: Attention mask for cross-attention to encoder.
        :type encoder_attn_mask: torch.Tensor, optional

        :param encoder_pos: Position indices for encoder hidden states (used by CrossAttention).
        :type encoder_pos: torch.Tensor, optional

        :param use_cache: Whether to return KV cache. Defaults to None which uses config setting.
        :type use_cache: bool, optional

        :return: Output tensor of shape :math:`(B, N, D)` if not return_states and no cache,
            else a dict containing the keys: "output", "attn_output" and "ffn_output".
            If use_cache is True or cache is provided, returns a tuple `(output, new_cache)`.
        :rtype: Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], Dict]
        """

        def extract(out):
            """Helper to extract output tensor from module return (handles dict vs tensor)"""
            return out["output"] if return_states else out

        attn, ffn = None, None
        attn_new_cache = None

        # Determine if we need to return cache (either provided or building new cache)
        # Cache is returned when: cache was provided OR we're in a use_cache context
        # The Transformer.forward passes cache=None on first pass but still expects cache back if use_cache=True
        # We detect this by checking if attention returns a tuple

        # Determine if we should request cache from attention (for use_cache mode)
        # When building cache on first pass, we need return_cache=True to get the cache back
        should_return_cache = use_cache or cache is not None

        if self.norm_design == "pre_norm":
            # Self-attention with optional KV cache
            attn_kwargs = {"mask": attn_mask, "pos": pos, "flash_attn": flash_attn, "cache": cache}
            if hasattr(self.attn, "forward") and "return_cache" in str(self.attn.forward.__code__.co_varnames):
                attn_kwargs["return_cache"] = should_return_cache
            attn = self.attn(
                self.norm_attn(x),
                return_states=return_states,
                **attn_kwargs,
            )
            if isinstance(attn, tuple):
                attn_output, attn_new_cache = attn
            else:
                attn_output = attn
                attn_new_cache = None
            x = x + extract(attn_output)

            ffn = self.ffn(self.norm_ffn(x), return_states=return_states)
            x = x + extract(ffn)

        elif self.norm_design == "post_norm":
            attn_kwargs = {"mask": attn_mask, "pos": pos, "flash_attn": flash_attn, "cache": cache}
            if hasattr(self.attn, "forward") and "return_cache" in str(self.attn.forward.__code__.co_varnames):
                attn_kwargs["return_cache"] = should_return_cache
            attn = self.attn(x, return_states=return_states, **attn_kwargs)
            if isinstance(attn, tuple):
                attn_output, attn_new_cache = attn
            else:
                attn_output = attn
                attn_new_cache = None
            x = self.norm_attn(x + extract(attn_output))

            ffn = self.ffn(x, return_states=return_states)
            x = self.norm_ffn(x + extract(ffn))

        elif self.norm_design == "both":
            attn_kwargs = {"mask": attn_mask, "pos": pos, "flash_attn": flash_attn, "cache": cache}
            if hasattr(self.attn, "forward") and "return_cache" in str(self.attn.forward.__code__.co_varnames):
                attn_kwargs["return_cache"] = should_return_cache
            attn = self.attn(
                self.pre_norm_attn(x),
                return_states=return_states,
                **attn_kwargs,
            )
            if isinstance(attn, tuple):
                attn_output, attn_new_cache = attn
            else:
                attn_output = attn
                attn_new_cache = None
            x = self.post_norm_attn(x + extract(attn_output))

            ffn = self.ffn(self.pre_norm_ffn(x), return_states=return_states)
            x = self.post_norm_ffn(x + extract(ffn))
        else:
            raise ValueError(f"Invalid norm_design: {self.norm_design}")

        # Return cache if attention provided one (either from existing cache or newly built)
        if attn_new_cache is not None:
            if return_states:
                result = {"output": x, "attn_output": attn, "ffn_output": ffn}
                return result, attn_new_cache
            return x, attn_new_cache

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
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: Optional[bool] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attn_mask: Optional[torch.Tensor] = None,
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

        :param pos: Position indices. If None, computed from sequence length or cache length.
        :type pos: torch.Tensor, optional

        :param flash_attn: Tuple of Arguments for Flash Attention.
        :type flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional

        :param return_states: If True, return hidden states of all layers. Default: False
        :type return_states: bool, optional

        :param loss_kwargs: Additional keyword arguments passed to `F.cross_entropy` for loss computation.
        :type loss_kwargs: Dict, optional

        :param past_key_values: Pre-computed KV cache from previous generation steps.
            Tuple of tuples, one per layer, each containing (key, value) tensors of shape (B, H, L, d).
        :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor], ...], optional

        :param use_cache: Whether to use KV cache. Defaults to config.use_cache if not specified.
        :type use_cache: bool, optional

        :param encoder_hidden_states: Encoder output for encoder-decoder models.
        :type encoder_hidden_states: torch.Tensor, optional

        :param encoder_attn_mask: Mask for encoder hidden states.
        :type encoder_attn_mask: torch.Tensor, optional

        :param kwargs: Additional keyword arguments
        :type kwargs: Dict, optional

        :return: CausalLMOutput containing loss (if labels given), logits, and optionally past_key_values and hidden_states.
        :rtype: CausalLMOutput
        """
        B, N = input_ids.shape
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        input_embs = self.emb(input_ids)
        out = input_embs

        # Compute position indices considering cache length
        if past_key_values is not None and len(past_key_values) > 0:
            cache_len = past_key_values[0][0].shape[2]  # Get cached sequence length from first layer's key
            total_len = cache_len + N
            if pos is None:
                pos = torch.arange(cache_len, total_len, device=out.device)
        else:
            cache_len = 0
            if pos is None:
                pos = torch.arange(N, device=out.device)

        # Generate causal mask covering full sequence length (cached + new)
        # For incremental decoding with N=1 query token and L_total cached tokens,
        # the mask should be (N, L_total) to mask out future positions from each query
        full_seq_len = cache_len + N
        if attn_mask is None and is_causal:
            # Create mask of shape (N, full_seq_len) where True means "mask this position"
            # For causal masking, query at position i can only attend to positions <= i
            attn_mask = torch.zeros(N, full_seq_len, device=out.device, dtype=torch.bool)
            for i in range(N):
                # Query at position (cache_len + i) can attend to positions 0 through (cache_len + i)
                # So we mask positions (cache_len + i + 1) onwards
                attn_mask[i, cache_len + i + 1 :] = True

        hidden_states = [] if return_states else None
        # Initialize new_past_key_values list if use_cache is enabled
        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            layer_cache = past_key_values[i] if past_key_values is not None else None

            # When building cache (first pass), we still need cache returned
            # Pass a special flag or handle differently
            build_cache = use_cache and layer_cache is None

            block_out = block(
                out,
                attn_mask=attn_mask,
                pos=pos,
                flash_attn=flash_attn,
                return_states=return_states,
                cache=layer_cache,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attn_mask=encoder_attn_mask,
                use_cache=use_cache,
            )

            if use_cache:
                # Block should return (output, new_cache) when use_cache is True
                if isinstance(block_out, tuple) and len(block_out) == 2:
                    out, new_cache = block_out
                    new_past_key_values.append(new_cache)
                else:
                    out = block_out

            if return_states:
                if use_cache and layer_cache is not None:
                    hidden_states.append(block_out[0])
                else:
                    hidden_states.append(out if isinstance(out, dict) else {"output": out})

        logits = self.lm_head(self.norm_out(out))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                **loss_kwargs
            )

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=tuple(new_past_key_values) if new_past_key_values else None,
            hidden_states=(input_embs, hidden_states) if return_states else None,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.emb

    def set_input_embeddings(self, embeddings: nn.Embedding):
        self.emb = embeddings

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""
        Load a pretrained model from a local directory or HuggingFace Hub.

        :param pretrained_model_name_or_path: Path to local model directory or HF Hub model ID.
        :type pretrained_model_name_or_path: str

        :param model_args: Additional positional arguments passed to PreTrainedModel.from_pretrained.
        :type model_args: tuple

        :param kwargs: Additional keyword arguments passed to PreTrainedModel.from_pretrained.
        :type kwargs: dict

        :return: Loaded Transformer model.
        :rtype: Transformer
        """
        # Use parent class method from PreTrainedModel
        return super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

    def push_to_hub(
        self,
        repo_id: str,
        config=None,
        commit_message: str = "Push model to hub",
        private: bool = False,
        token=None,
        max_shard_size: str = "5GB",
        **kwargs,
    ):
        r"""
        Push the model and its configuration to the HuggingFace Hub.

        :param repo_id: Repository ID on the Hub (e.g., \"username/my-model\").
        :type repo_id: str

        :param config: Optional config object. If None, uses self.config.
        :type config: PretrainedConfig, optional

        :param commit_message: Commit message for the upload. Default: \"Push model to hub\"
        :type commit_message: str, optional

        :param private: Whether to create a private repository. Default: False
        :type private: bool, optional

        :param token: HuggingFace API token. If None, uses cached token.
        :type token: str, optional

        :param max_shard_size: Maximum size per shard when sharding. Default: \"5GB\"
        :type max_shard_size: str, optional

        :param kwargs: Additional keyword arguments passed to PreTrainedModel.push_to_hub.
        :type kwargs: dict

        :return: URL to the uploaded model repository.
        :rtype: str
        """
        return super().push_to_hub(
            repo_id=repo_id,
            config=config or self.config,
            commit_message=commit_message,
            private=private,
            token=token,
            max_shard_size=max_shard_size,
            **kwargs,
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.emb

    def set_input_embeddings(self, embeddings: nn.Embedding):
        self.emb = embeddings

    def get_num_params(self) -> int:
        """Return the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
