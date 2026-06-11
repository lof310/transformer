"""Encoder-Decoder Transformer model components."""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin
from transformers.modeling_outputs import \
    CausalLMOutputWithPast as CausalLMOutput

from .config import TransformerConfig
from .transformer import Transformer


class EncoderDecoderModel(nn.Module, GenerationMixin):
    """
    Encoder-Decoder Transformer model for seq2seq tasks.

    Uses a Transformer encoder and decoder with cross-attention support.

    :param encoder_config: Configuration for the encoder.
    :type encoder_config: TransformerConfig

    :param decoder_config: Configuration for the decoder.
    :type decoder_config: TransformerConfig
    """

    def __init__(self, encoder_config: TransformerConfig, decoder_config: TransformerConfig):
        super().__init__()
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config

        self.encoder = Transformer(encoder_config)
        self.decoder = Transformer(decoder_config)

        self.main_input_name = "input_ids"

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
        # Run encoder forward manually to get hidden states (before lm_head projection)
        input_embs = self.encoder.emb(input_ids)
        out = input_embs

        B, N = input_ids.shape
        pos = torch.arange(N, device=out.device)

        # Generate causal mask
        attn_mask = None
        if attention_mask is not None:
            attn_mask = (1.0 - attention_mask.unsqueeze(1).unsqueeze(2)).bool()

        for block in self.encoder.blocks:
            block_out = block(
                out,
                attn_mask=attn_mask,
                pos=pos,
                return_states=False,
            )
            out = block_out

        hidden_states = self.encoder.norm_out(out)

        if return_dict:
            return {"last_hidden_state": hidden_states}
        return (hidden_states,)

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
        # CrossAttention doesn't support causal masking in the same way as self-attention
        # The decoder uses cross-attention to encoder, so is_causal should be False
        decoder_outputs = self.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attn_mask=encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            is_causal=False,  # Cross-attention doesn't use causal masking
            return_states=False,
        )
        return decoder_outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        encoder_input_ids: torch.Tensor,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = True,
        return_dict: bool = True,
        **kwargs,
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
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1), **kwargs.get("loss_kwargs", {}))

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=past_key_values,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Dict:
        """
        Prepare inputs for generation using HuggingFace's GenerationMixin.

        :param input_ids: Current token IDs.
        :type input_ids: torch.LongTensor

        :param past_key_values: Cached key/value states from previous steps.
        :type past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor]]]

        :param attention_mask: Decoder attention mask.
        :type attention_mask: Optional[torch.Tensor]

        :param encoder_attention_mask: Encoder attention mask.
        :type encoder_attention_mask: Optional[torch.Tensor]

        :param encoder_hidden_states: Pre-computed encoder outputs.
        :type encoder_hidden_states: Optional[torch.Tensor]

        :param kwargs: Additional keyword arguments.

        :return: Dictionary of inputs for the forward method.
        :rtype: Dict
        """
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        return {
            "input_ids": input_ids,
            "encoder_input_ids": kwargs.get("encoder_input_ids", None),
            "encoder_attention_mask": encoder_attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def _reorder_cache(
        self, past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]], beam_idx: torch.LongTensor
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Reorder cache for beam search.

        :param past_key_values: KV cache tuple.
        :type past_key_values: Tuple[Tuple[torch.Tensor, torch.Tensor]]

        :param beam_idx: Beam indices for reordering.
        :type beam_idx: torch.LongTensor

        :return: Reordered cache.
        :rtype: Tuple[Tuple[torch.Tensor, torch.Tensor]]
        """
        reordered_past = []
        for layer_past in past_key_values:
            reordered_layer = tuple(
                past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past
            )
            reordered_past.append(reordered_layer)
        return tuple(reordered_past)
