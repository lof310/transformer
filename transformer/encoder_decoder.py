"""Encoder-Decoder Transformer model components."""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_outputs import \
    CausalLMOutputWithPast as CausalLMOutput

from .config import TransformerConfig
from .transformer import TransformerBlock


class Encoder(nn.Module):
    """
    Transformer Encoder for processing source sequences.

    :param config: Encoder configuration.
    :type config: TransformerConfig
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layer

        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_layer)])
        self.norm = nn.RMSNorm(config.d_model)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Dict]:
        """
        Encode source sequences.

        :param input_ids: Source token IDs of shape (B, L_enc).
        :type input_ids: torch.Tensor

        :param attention_mask: Attention mask for encoder (1 for valid, 0 for padding).
        :type attention_mask: torch.Tensor, optional

        :param return_dict: If True, return dict; else return tuple. Default: True
        :type return_dict: bool, optional

        :return: Encoder hidden states of shape (B, L_enc, D).
        :rtype: Union[Tuple[torch.Tensor], Dict]
        """
        B, L_enc = input_ids.shape
        hidden_states = self.emb(input_ids)

        encoder_mask = None
        if attention_mask is not None:
            encoder_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)) * torch.finfo(hidden_states.dtype).min
            encoder_mask = encoder_mask.expand(B, 1, 1, L_enc)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                attn_mask=encoder_mask,
                pos=None,
                return_states=False,
            )

        hidden_states = self.norm(hidden_states)

        if return_dict:
            return {"last_hidden_state": hidden_states}
        return (hidden_states,)


class Decoder(nn.Module):
    """
    Transformer Decoder with cross-attention support.

    :param config: Decoder configuration. Must have add_cross_attention=True for seq2seq.
    :type config: TransformerConfig
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_layers = config.n_layer
        self.vocab_size = config.vocab_size
        self.tied_weights = config.tied_weights
        self.lm_head_bias = config.lm_head_bias

        self.emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([TransformerBlock(config, layer_idx=i) for i in range(config.n_layer)])
        self.norm = nn.RMSNorm(config.d_model)

        if config.tied_weights:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.lm_head_bias)
            self.lm_head.weight = self.emb.weight
        else:
            self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=config.lm_head_bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        is_causal: bool = True,
        attn_mask: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None,
        flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool] = (
            False,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            False,
        ),
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutput]:
        """
        Decode target sequences with optional cross-attention.

        :param input_ids: Target token IDs of shape (B, L_dec).
        :type input_ids: torch.Tensor

        :param encoder_hidden_states: Encoder output of shape (B, L_enc, D).
        :type encoder_hidden_states: torch.Tensor, optional

        :param encoder_attention_mask: Encoder attention mask.
        :type encoder_attention_mask: torch.Tensor, optional

        :param past_key_values: KV cache from previous decoding steps.
        :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]], optional

        :param use_cache: Whether to use KV cache. Default: True
        :type use_cache: bool, optional

        :param is_causal: If True, apply causal masking. Default: True
        :type is_causal: bool, optional

        :param attn_mask: Custom attention mask.
        :type attn_mask: torch.Tensor, optional

        :param pos: Position indices.
        :type pos: torch.Tensor, optional

        :param flash_attn: Flash attention arguments.
        :type flash_attn: Tuple, optional

        :param return_dict: If True, return dict; else return tuple. Default: True
        :type return_dict: bool, optional

        :return: Decoder logits and optionally past_key_values.
        :rtype: Union[Tuple[torch.Tensor], CausalLMOutput]
        """
        B, L_dec = input_ids.shape
        device = input_ids.device

        cache_len = past_key_values[0][0].shape[2] if past_key_values else 0
        total_len = cache_len + L_dec

        if pos is None:
            pos = torch.arange(cache_len, total_len, device=device)

        hidden_states = self.emb(input_ids)

        if attn_mask is None and is_causal:
            attn_mask = torch.zeros(L_dec, total_len, device=device, dtype=torch.bool)
            for i in range(L_dec):
                attn_mask[i, cache_len + i + 1 :] = True

        new_past_key_values = [] if use_cache else None

        for i, block in enumerate(self.blocks):
            layer_cache = past_key_values[i] if past_key_values else None

            block_output = block(
                hidden_states,
                attn_mask=attn_mask,
                pos=pos,
                flash_attn=flash_attn,
                cache=layer_cache,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attn_mask=encoder_attention_mask,
                use_cache=use_cache,
                return_states=False,
            )

            if use_cache:
                hidden_states, new_cache = block_output
                new_past_key_values.append(new_cache)
            else:
                hidden_states = block_output

        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        if return_dict:
            return CausalLMOutput(
                logits=logits,
                past_key_values=tuple(new_past_key_values) if new_past_key_values else None,
            )
        return (logits, tuple(new_past_key_values)) if use_cache else (logits,)


class EncoderDecoderModel(nn.Module):
    """
    Encoder-Decoder Transformer model combining an encoder and decoder.

    The encoder processes source sequences through embedding and Transformer blocks
    (without causal masking). The decoder processes target sequences with causal
    masking and cross-attention to encoder outputs.

    :param encoder_config: Configuration for the encoder.
    :type encoder_config: TransformerConfig

    :param decoder_config: Configuration for the decoder. Must have add_cross_attention=True
        for cross-attention to work.
    :type decoder_config: TransformerConfig
    """

    def __init__(self, encoder_config: TransformerConfig, decoder_config: TransformerConfig):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.encoder = Encoder(encoder_config)
        self.decoder = Decoder(decoder_config)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], Dict]:
        """
        Encode source sequences.

        :param input_ids: Source token IDs of shape (B, L_enc).
        :type input_ids: torch.Tensor

        :param attention_mask: Attention mask for encoder (1 for valid, 0 for padding).
        :type attention_mask: torch.Tensor, optional

        :param return_dict: If True, return dict; else return tuple. Default: True
        :type return_dict: bool, optional

        :return: Encoder hidden states of shape (B, L_enc, D).
        :rtype: Union[Tuple[torch.Tensor], Dict]
        """
        return self.encoder(input_ids, attention_mask, return_dict)

    def decode(
        self,
        input_ids: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> Union[Tuple[torch.Tensor], CausalLMOutput]:
        """
        Decode target sequences with cross-attention to encoder outputs.

        :param input_ids: Target token IDs of shape (B, L_dec).
        :type input_ids: torch.Tensor

        :param encoder_hidden_states: Encoder output of shape (B, L_enc, D).
        :type encoder_hidden_states: torch.Tensor

        :param encoder_attention_mask: Encoder attention mask.
        :type encoder_attention_mask: torch.Tensor, optional

        :param past_key_values: KV cache from previous decoding steps.
        :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]], optional

        :param use_cache: Whether to use KV cache. Default: True
        :type use_cache: bool, optional

        :param return_dict: If True, return dict; else return tuple. Default: True
        :type return_dict: bool, optional

        :return: Decoder logits and optionally past_key_values.
        :rtype: Union[Tuple[torch.Tensor], CausalLMOutput]
        """
        return self.decoder(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=return_dict,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        return_dict: bool = True,
    ) -> CausalLMOutput:
        """
        Full forward pass through encoder-decoder model.

        :param input_ids: Target token IDs of shape (B, L_dec).
        :type input_ids: torch.Tensor

        :param encoder_input_ids: Source token IDs of shape (B, L_enc).
        :type encoder_input_ids: torch.Tensor

        :param encoder_attention_mask: Encoder attention mask (1 for valid, 0 for padding).
        :type encoder_attention_mask: torch.Tensor, optional

        :param labels: Target labels for loss computation.
        :type labels: torch.Tensor, optional

        :param past_key_values: KV cache for incremental decoding.
        :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]], optional

        :param use_cache: Whether to use KV cache. Default: True
        :type use_cache: bool, optional

        :param return_dict: If True, return CausalLMOutput; else return tuple. Default: True
        :type return_dict: bool, optional

        :return: Model output with logits and optionally loss.
        :rtype: CausalLMOutput
        """
        encoder_outputs = self.encode(
            encoder_input_ids,
            attention_mask=encoder_attention_mask,
            return_dict=True,
        )
        encoder_hidden_states = encoder_outputs["last_hidden_state"]

        decoder_outputs = self.decode(
            input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            return_dict=True,
        )

        logits = decoder_outputs.logits
        past_key_values = decoder_outputs.past_key_values

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
            past_key_values=past_key_values,
        )
