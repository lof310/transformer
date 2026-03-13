from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos import RoPE


class MHA(nn.Module):
    r"""
    **Multi-Head Attention** ``MHA`` module using the optimized implementation of
    ``torch.nn.functional.scaled_dot_product_attention()`` when possible.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads. Note that ``d_model`` will be split
        across ``n_heads`` (i.e. each head will have dimension ``d_head//n_heads``).

        dropout (float): Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
            **Note: Latest SOTA Architectures do not use Dropout at all and for Research Purposes
            it is recommended to never use it.**

        attn_bias (bool, optional): Whether to use bias in linear projections. Default: ``False``

        qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. Default: ``True``

        layer_idx (int, optional): Index of the layer (used for debugging/logging).

        rope_base (float, optional): Base for the Exponential Frequency Calculation in RoPE.
        Default: ``10000.0``

        max_seq_len (int): Maximum sequence length for RoPE.

    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        attn_bias: Optional[bool] = False,
        qk_norm: Optional[bool] = True,
        layer_idx: int = 0,
        rope_base: float = 10000.0,
        pos_encoding: str = "RoPE",
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head, self.layer_idx = d_model, n_heads, d_model // n_heads, layer_idx
        self.qk_norm = qk_norm

        self.qkv_proj = nn.Linear(self.d_model, self.d_model * 3, bias=attn_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=attn_bias)

        if pos_encoding == "RoPE":
            self.rope = RoPE(max_seq_len, self.d_head, rope_base=rope_base)
        elif pos_encoding == "AliBI":
            raise ValueError("Under Development")
        else:
            raise ValueError("Not implemented")
        self.scale = self.d_head**-0.5

        self.dropout = dropout if dropout is not None or dropout != 0.0 else None

        if qk_norm:
            self.q_norm, self.k_norm = nn.RMSNorm(self.d_head), nn.RMSNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.LongTensor] = None,
        flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool] = (
            False,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            False,
        ),
        return_states: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict]:
        r"""
        Forward pass of MHA.

        Args:
            x (torch.Tensor): Input tensor of shape :math:`(B, N, D)` where :math:`N` is the Sequence Length,
                :math:`B` is the batch size, and :math:`D` is the embedding dimension ``d_model``.

            mask (torch.BoolTensor, optional): If specified, a 2D or 4D mask preventing attention to certain positions. Must be of shape
                :math:`(N, N)` or :math:`(B, H, N, N)`, where :math:`B` is the batch size, :math:`H` is the number of heads and
                :math:`N` is the Sequence Length. A 2D mask will be broadcasted across the batch while a 4D mask allows
                for a different mask for each entry in the batch and/or heads dimensions.
                **Note: Should be a boolean mask where True indicates masked positions.**

            pos (torch.LongTensor, optional): Position indices for RoPE, shape :math:`(N)` or :math:`(B, N)`

            flash_attn (Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional): Tuple of Arguments for Flash Attention and the Context manager to select which backend to use for scaled dot product attention.
                bool: Whether to use or not Flash Attention. Default: ``False``
                Union[List[SDPBackend], SDPBackend]: A backend or list of backends for scaled dot product attention. Default: ``torch.nn.attention.SPDBackend.FLASH_ATTENTION``
                bool: Whether the ordering of the backends is interpreted as their priority order. Default: ``False``

            return_states (bool, optional): If ``True``, return a dictionary of intermediate tensors. Default: ``False``

        Returns:
            Union[torch.Tensor, Dict]: Output tensor :math:`(B, N, D)` if not return_states, else a dict containing
                The keys: {`output`, `queries`, `keys`, `values`, `attn_weights`, `attn_scores`, `output_before_proj` and `input`}

        """
        B, N, D, H, d = *x.shape, self.n_heads, self.d_head

        q, k, v = self.qkv_proj(x).view(B, N, H, d * 3).transpose(1, 2).chunk(3, dim=-1)
        q, k = (self.q_norm(q), self.k_norm(k)) if self.qk_norm else (q, k)
        q, k = self.rope(q, k, pos, pos) if pos is not None else (q, k)

        y, A_weights, A_scores = None, None, None
        if flash_attn[0]:
            with torch.nn.attention.sdpa_kernel(backends=flash_attn[1], set_priority=flash_attn[2]):
                y = (
                    F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=mask,
                        dropout_p=self.dropout if self.dropout is not None else 0.0,
                        is_causal=False,
                        scale=self.scale,
                        enable_gqa=False,
                    )
                    .transpose(1, 2)
                    .contriguous()
                    .view(B, N, D)
                )
        else:
            A_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            A_scores = (
                F.softmax(A_weights.masked_fill_(mask, float("-inf")), dim=-1)
                if mask is not None
                else F.softmax(A_weights, dim=-1)
            )
            if self.dropout is not None:
                A_scores = F.dropout(A_scores, p=self.dropout, training=self.training, inplace=False)
            y = torch.matmul(A_scores, v).transpose(1, 2).contiguous().view(B, N, D)

        if return_states:
            if flash_attn[0]:
                return {
                    "output": self.out_proj(y),
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "output_before_proj": y,
                    "input": x,
                }
            else:
                return {
                    "output": self.out_proj(y),
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "attn_weights": A_weights,
                    "attn_scores": A_scores,
                    "output_before_proj": y,
                    "input": x,
                }
        else:
            return self.out_proj(y)


class GQA(nn.Module):
    """
    **Grouped Query Attention** ``GQA`` module using the optimized implementation of
    ``torch.nn.functional.scaled_dot_product_attention()`` when possible.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads. Note that ``d_model`` will be split
        across ``n_heads`` (i.e. each head will have dimension ``d_head//n_heads``).

        n_kv_heads (int): Number of key/value heads (must divide n_heads).

        dropout (float, optional): Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
            **Note: Latest SOTA Architectures do not use Dropout at all and for Research Purposes
            it is recommended to never use it.**

        attn_bias (bool, optional): Whether to use bias in linear projections. Default: ``False``

        qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. Default: ``True``

        layer_idx (int, optional): Index of the layer (used for debugging/logging).

        rope_base (float, optional): Base for the Exponential Frequency Calculation in RoPE. Default: ``10000.0``

        max_seq_len (int): Maximum sequence length for RoPE.

    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: Optional[float] = 0.0,
        attn_bias: Optional[bool] = False,
        qk_norm: Optional[bool] = True,
        layer_idx: int = 0,
        rope_base: float = 10000.0,
        pos_encoding: str = "RoPE",
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model, self.n_heads, self.n_kv_heads, self.groups, self.d_head, self.layer_idx = (
            d_model,
            n_heads,
            n_kv_heads,
            n_heads // n_kv_heads,
            d_model // n_heads,
            layer_idx,
        )
        self.qk_norm = qk_norm

        self.qkv_proj = nn.Linear(
            self.d_model, (self.d_head * n_heads) + (self.d_head * self.n_kv_heads * 2), bias=attn_bias
        )
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=attn_bias)

        self.rope = RoPE(max_block_size, self.d_head, rope_base=rope_base)
        self.scale = self.d_head**-0.5

        self.dropout = dropout if dropout is not None or dropout != 0.0 else None

        if qk_norm:
            self.q_norm, self.k_norm = nn.RMSNorm(self.d_head), nn.RMSNorm(self.d_head)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pos: Optional[torch.LongTensor] = None,
        flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool] = (
            False,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            False,
        ),
        return_states: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict]:
        """
        Forward pass of GQA.

        Args:
            x (torch.Tensor): Input tensor of shape :math:`(B, N, D)` where :math:`N` is the Sequence Length,
                :math:`B` is the batch size, and :math:`D` is the embedding dimension ``d_model``.

            mask (torch.BoolTensor, optional): If specified, a 2D or 4D mask preventing attention to certain positions. Must be of shape
                :math:`(N, N)` or :math:`(B, H, N, N)`, where :math:`B` is the batch size, :math:`H` is the number of heads and
                :math:`N` is the Sequence Length. A 2D mask will be broadcasted across the batch while a 4D mask allows
                for a different mask for each entry in the batch and/or heads dimensions.
                **Note: Should be a boolean mask where True indicates masked positions.**

            pos (torch.LongTensor, optional): Position indices for RoPE, shape :math:`(N)` or :math:`(B, N)`

            flash_attn (Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional): Tuple of Arguments for Flash Attention and the Context manager to select which backend to use for scaled dot product attention.
                bool: Whether to use or not Flash Attention. Default: ``False``
                Union[List[SDPBackend], SDPBackend]: A backend or list of backends for scaled dot product attention. Default: ``torch.nn.attention.SPDBackend.FLASH_ATTENTION``
                bool: Whether the ordering of the backends is interpreted as their priority order. Default: ``False``

            return_states (bool, optional): If ``True``, return a dictionary of intermediate tensors. Default: ``False``

        Returns:
            Union[torch.Tensor, Dict]: Output tensor of shape :math:`(B, N, D)` if not return_states, else a dict containing
                The keys: {`output`, `queries`, `keys`, `values`, `attn_weights`, `attn_scores`, `output_before_proj` and `input`}

        """
        B, N, D, H_q, H_kv, G, d = *x.shape, self.n_heads, self.n_kv_heads, self.groups, self.d_head

        q, k, v = self.qkv_proj(x).view(B, N, H_q + (H_kv * 2), d).transpose(1, 2).split([H_q, H_kv, H_kv], dim=1)
        q, k = (self.q_norm(q), self.k_norm(k)) if qk_norm else (q, k)
        q, k = self.rope(q, k, pos, pos) if pos is not None else (q, k)

        y, A_weights, A_scores = None, None, None
        if flash_attn[0]:
            with torch.nn.attention.sdpa_kernel(backends=flash_attn[1], set_priority=flash_attn[2]):
                y = (
                    F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=mask,
                        dropout_p=self.dropout if self.dropout is not None else 0.0,
                        is_causal=False,
                        scale=self.scale,
                        enable_gqa=True,
                    )
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, N, D)
                )
        else:
            A_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            A_scores = (
                F.softmax(A_weights.masked_fill_(mask, float("-inf")), dim=-1)
                if mask is not None
                else F.softmax(A_weights, dim=-1)
            )
            if self.dropout is not None:
                A_scores = F.dropout(A_scores, p=self.dropout, training=self.training, inplace=False)
            y = torch.matmul(A_scores, v).transpose(1, 2).contiguous().view(B, N, D)

        if return_states:
            if flash_attn[0]:
                return {
                    "output": self.out_proj(y),
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "output_before_proj": y,
                    "input": x,
                }
            else:
                return {
                    "output": self.out_proj(y),
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "attn_weights": A_weights,
                    "attn_scores": A_scores,
                    "output_before_proj": y,
                    "input": x,
                }
        else:
            return self.out_proj(y)


class CrossAttention(nn.Module):
    """
    **CrossAttention** module using the optimized implementation of
    ``torch.nn.functional.scaled_dot_product_attention()`` when possible.

    Args:
        d_model (int): Model dimension.
        n_heads (int): Number of attention heads. Note that ``d_model`` will be split
        across ``n_heads`` (i.e. each head will have dimension ``d_head//n_heads``).

        n_kv_heads (int): Number of key/value heads (must divide n_heads). Default: ``n_heads``

        dropout (float, optional): Dropout probability on ``attn_output_weights``. Default: ``0.0`` (no dropout).
            **Note: Latest SOTA Architectures do not use Dropout at all and for Research Purposes
            it is recommended to never use it.**

        attn_bias (bool, optional): Whether to use bias in linear projections. Default: ``False``

        qk_norm (bool, optional): Whether to apply RMSNorm to queries and keys. Default: ``True``

        layer_idx (int, optional): Index of the layer (used for debugging/logging).

        rope_base (float, optional): Base for the Exponential Frequency Calculation in RoPE. Default: ``10000.0``

        max_seq_len (int): Maximum sequence length for RoPE.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: Optional[float] = 0.0,
        attn_bias: Optional[bool] = False,
        qk_norm: Optional[bool] = True,
        layer_idx: int = 0,
        rope_base: float = 10000.0,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head, self.layer_idx = d_model, n_heads, d_model // n_heads, layer_idx
        self.qk_norm = qk_norm

        self.q, self.kv_proj, self.out_proj = (
            nn.Linear(self.d_model, self.d_model, bias=attn_bias),
            nn.Linear(self.d_model, self.d_model * 2, bias=attn_bias),
            nn.Linear(self.d_model, self.d_model, bias=attn_bias),
        )

        self.rope = RoPE(max_block_size, self.d_head, rope_base=rope_base)
        self.scale = self.d_head**-0.5

        self.dropout = nn.Dropout(dropout) if dropout is not None or dropout != 0.0 else None

        if qk_norm:
            self.q_norm, self.k_norm = nn.RMSNorm(self.d_head), nn.RMSNorm(self.d_head)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        mask: Optional[torch.BoolTensor] = None,
        pos_q: Optional[torch.LongTensor] = None,
        pos_k: Optional[torch.LongTensor] = None,
        flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool] = (
            False,
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            False,
        ),
        return_states: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict]:
        """
        Forward pass of CrossAttention.

        Args:
            queries (torch.Tensor): Input tensor of shape :math:`(B, Lq, D)` where :math:`Lq` is the Sequence Length for the query sequence,
                :math:`B` is the batch size, and :math:`D` is the embedding dimension ``d_model``.

            kv (torch.Tensor): Input tensor of shape :math:`(B, Lq, D)` where :math:`Lk` is the Sequence Length for the key/value sequence,
                :math:`B` is the batch size, and :math:`D` is the embedding dimension ``d_model``.

            mask (torch.BoolTensor, optional): If specified, a 2D or 4D mask preventing attention to certain positions. Must be of shape
                :math:`(Lq, Lk)` or :math:`(B, H, Lq, Lk)`, where :math:`B` is the batch size, :math:`H` is the number of heads,
                :math:`Lq` is the Sequence Length of the query sequence and :math:`Lk` is the Sequence Length of the key/value sequence.
                A 2D mask will be broadcasted across the batch while a 4D mask allows for a different mask for each entry
                in the batch and/or heads dimensions.
                **Note: Should be a boolean mask where True indicates masked positions.**

            pos_q (torch.LongTensor, optional): Position indices for Queries, shape :math:`(Lq)` or :math:`(B, Lq)`

            pos_k (torch.LongTensor, optional): Position indices for Keys, shape :math:`(Lk)` or :math:`(B, Lk)`

            flash_attn (Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional): Tuple of Arguments for Flash Attention and the Context manager to select which backend to use for scaled dot product attention.
                bool: Whether to use or not Flash Attention. Default: ``False``
                Union[List[SDPBackend], SDPBackend]: A backend or list of backends for scaled dot product attention. Default: ``torch.nn.attention.SPDBackend.FLASH_ATTENTION``
                bool: Whether the ordering of the backends is interpreted as their priority order. Default: ``False``

            return_states (bool, optional): If True, return dictionary of intermediates tensors. Default: False

        Returns:
            Union[torch.Tensor, Dict]: Output tensor of shape :math:`(B, N, D)` if not return_states, else a dict containing
                The keys: {`output`, `queries`, `keys`, `values`, `attn_weights`, `attn_scores`, `output_before_proj` and `input`} where `input` is a tuple (queries, kv)

        """
        B, Lq, D, Lk, H, d = *q.shape, kv.shape[1], self.n_heads, self.d_head

        q, k, v = self.q_proj(queries).view(B, Lq, H, d).transpose(1, 2), *self.kv_proj(kv).view(
            B, Lk, H, d * 2
        ).transpose(1, 2).chunk(2, dim=-1)
        q, k = (self.q_norm(q), self.k_norm(k)) if self.qk_norm else (q, k)
        q, k = self.rope(q, k, pos_q, pos_k) if pos_q is not None and pos_k is not None else (q, k)

        y, A_weights, A_scores = None, None, None
        if flash_attn[0]:
            with torch.nn.attention.sdpa_kernel(backends=flash_attn[1], set_priority=flash_attn[2]):
                y = (
                    F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=mask,
                        dropout_p=self.dropout if self.dropout is not None else 0.0,
                        is_causal=False,
                        scale=self.scale,
                        enable_gqa=False,
                    )
                    .transpose(1, 2)
                    .contiguous()
                    .view(B, Lq, D)
                )
        else:
            A_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
            A_scores = (
                F.softmax(A_weights.masked_fill_(mask, float("-inf")), dim=-1)
                if mask is not None
                else F.softmax(A_weights, dim=-1)
            )
            if self.dropout is not None:
                A_scores = F.dropout(A_scores, p=self.dropout, training=self.training, inplace=False)
            y = torch.matmul(A_scores, v).transpose(1, 2).contiguous().view(B, Lq, D)

        if return_states:
            if flash_attn[0]:
                return {
                    "output": self.out_proj(y),
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "output_before_proj": y,
                    "input": x,
                }
            else:
                return {
                    "output": self.out_proj(y),
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "attn_weights": A_weights,
                    "attn_scores": A_scores,
                    "output_before_proj": y,
                    "input": x,
                }
        else:
            return self.out_proj(y)
