from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .pos import ALiBi, PartialRoPE, RoPE


class MHA(nn.Module):
    r"""
    Multi-Head Attention (MHA) module using optimized scaled_dot_product_attention.

    :param d_model: Model dimension.
    :type d_model: int

    :param n_heads: Number of attention heads. ``d_model`` is split across ``n_heads``.
    :type n_heads: int

    :param dropout: Dropout probability on attention weights. Default: ``0.0``
    :type dropout: float, optional

    :param attn_bias: Whether to use bias in linear projections. Default: ``False``
    :type attn_bias: bool, optional

    :param qk_norm: Whether to apply RMSNorm to queries and keys. Default: ``True``
    :type qk_norm: bool, optional

    :param layer_idx: Index of the layer (used for debugging/logging).
    :type layer_idx: int, optional

    :param pos_encoding: Positional Encoding to use. Default: ``RoPE``
    :type pos_encoding: str, optional

    :param pos_encoding_kwargs: Dictionary of additional arguments for positional encoding.
    :type pos_encoding_kwargs: Dict, optional

    :param max_seq_len: Maximum sequence length for RoPE.
    :type max_seq_len: int
    """

    supports_cache = True

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        attn_bias: Optional[bool] = False,
        qk_norm: Optional[bool] = True,
        layer_idx: int = 0,
        pos_encoding: str = "RoPE",
        pos_encoding_kwargs: Dict = {},
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head, self.layer_idx = d_model, n_heads, d_model // n_heads, layer_idx
        self.qk_norm = qk_norm

        self.qkv_proj = nn.Linear(self.d_model, self.d_model * 3, bias=attn_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=attn_bias)

        if pos_encoding == "RoPE":
            self.rope = RoPE(max_seq_len, self.d_head, **pos_encoding_kwargs)
        elif pos_encoding == "PartialRoPE":
            self.rope = PartialRoPE(max_seq_len, self.d_head, **pos_encoding_kwargs)
        elif pos_encoding == "AliBI":
            raise ValueError("Under Development")
        else:
            raise ValueError(f"Not implemented: {pos_encoding}")

        self.scale = self.d_head**-0.5
        self.dropout = dropout

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
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_cache: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        r"""
        Forward pass of MHA.

        :param x: Input tensor of shape :math:`(B, N, D)` where :math:`N` is the sequence length,
            :math:`B` is the batch size, and :math:`D` is the embedding dimension ``d_model``.
        :type x: torch.Tensor

        :param mask: Boolean mask preventing attention to certain positions. Shape :math:`(N, N)` or :math:`(B, H, N, N)`.
            **True indicates masked positions.** When Flash Attention is enabled, it is inverted internally.
        :type mask: torch.BoolTensor, optional

        :param pos: Position indices for RoPE, shape :math:`(N)` or :math:`(B, N)`
        :type pos: torch.LongTensor, optional

        :param flash_attn: Tuple controlling Flash Attention usage:
            - bool: Whether to use Flash Attention. Default: ``False``
            - Union[List[SDPBackend], SDPBackend]: Backend(s) for scaled dot product attention
            - bool: Whether backend order indicates priority. Default: ``False``
        :type flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional

        :param return_states: If ``True``, return dictionary with intermediate tensors. Default: ``False``
        :type return_states: bool, optional

        :param cache: Optional KV cache tuple `(k_prev, v_prev)` of shape `(B, H, L_prev, d)` each.
            New keys/values are concatenated with cached ones along sequence dimension.
        :type cache: Tuple[torch.Tensor, torch.Tensor], optional

        :param return_cache: If True, always return cache tuple. Used for building initial cache.
        :type return_cache: bool, optional

        :return: Output tensor :math:`(B, N, D)` if not return_states, else dict containing
            keys: `output`, `queries`, `keys`, `values`, `attn_weights`, `attn_scores`, `output_before_proj`, `input`.
            If cache is provided or return_cache=True, returns tuple `(output, new_cache)`.
        :rtype: Union[torch.Tensor, Dict, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]
        """

        B, N, D, H, d = *x.shape, self.n_heads, self.d_head

        qkv = self.qkv_proj(x).view(B, N, 3, H, d).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        if pos is not None:
            q, k = self.rope(q, k, pos, pos)

        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat((k_prev, k), dim=2)
            v = torch.cat((v_prev, v), dim=2)
        new_cache = (k, v)

        L_total = k.shape[2]

        y, A_weights, A_scores = None, None, None
        if flash_attn[0]:
            with torch.nn.attention.sdpa_kernel(backends=flash_attn[1], set_priority=flash_attn[2]):
                y = (
                    F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=(~mask) if mask is not None else None,
                        dropout_p=self.dropout,
                        is_causal=False,
                        scale=self.scale,
                        enable_gqa=False,
                    )
                    .transpose(1, 2)
                    .reshape(B, N, D)
                )
        else:
            A_weights = torch.matmul(q, k.transpose(-1, -2)).mul_(self.scale)
            if mask is not None:
                A_weights = A_weights.masked_fill_(mask, float("-inf"))
            A_scores = A_weights.softmax(dim=-1)

            if self.dropout > 0.0 and self.training:
                A_scores = F.dropout(A_scores, p=self.dropout, inplace=False)

            y = torch.matmul(A_scores, v).transpose(1, 2).reshape(B, N, D)

        out = self.out_proj(y)

        if return_cache:
            if return_states:
                result = {
                    "output": out,
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "output_before_proj": y,
                    "input": x,
                }
                if not flash_attn[0]:
                    result["attn_weights"] = A_weights
                    result["attn_scores"] = A_scores
                return result, new_cache
            return out, new_cache

        if return_states:
            result = {
                "output": out,
                "queries": q,
                "keys": k,
                "values": v,
                "output_before_proj": y,
                "input": x,
            }
            if not flash_attn[0]:
                result["attn_weights"] = A_weights
                result["attn_scores"] = A_scores
            return result
        return out


class GQA(nn.Module):
    """
    Grouped Query Attention (GQA) module using optimized scaled_dot_product_attention.

    :param d_model: Model dimension.
    :type d_model: int

    :param n_heads: Number of attention heads. ``d_model`` is split across ``n_heads``.
    :type n_heads: int

    :param n_kv_heads: Number of key/value heads (must divide n_heads).
    :type n_kv_heads: int

    :param dropout: Dropout probability on attention weights. Default: ``0.0``
    :type dropout: float, optional

    :param attn_bias: Whether to use bias in linear projections. Default: ``False``
    :type attn_bias: bool, optional

    :param qk_norm: Whether to apply RMSNorm to queries and keys. Default: ``True``
    :type qk_norm: bool, optional

    :param layer_idx: Index of the layer (used for debugging/logging).
    :type layer_idx: int, optional

    :param pos_encoding: Positional Encoding to use. Default: ``RoPE``
    :type pos_encoding: str, optional

    :param pos_encoding_kwargs: Dictionary of additional arguments for positional encoding.
    :type pos_encoding_kwargs: Dict, optional

    :param max_seq_len: Maximum sequence length for RoPE.
    :type max_seq_len: int
    """

    supports_cache = True

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        n_kv_heads: int,
        dropout: Optional[float] = 0.0,
        attn_bias: Optional[bool] = False,
        qk_norm: Optional[bool] = True,
        layer_idx: int = 0,
        pos_encoding: str = "RoPE",
        pos_encoding_kwargs: Dict = {},
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

        if pos_encoding == "RoPE":
            self.rope = RoPE(max_seq_len, self.d_head, **pos_encoding_kwargs)
        elif pos_encoding == "PartialRoPE":
            self.rope = PartialRoPE(max_seq_len, self.d_head, **pos_encoding_kwargs)
        elif pos_encoding == "AliBI":
            raise ValueError("Under Development")
        else:
            raise ValueError(f"Not implemented: {pos_encoding}")

        self.scale = self.d_head**-0.5
        self.dropout = dropout

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
        cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_cache: Optional[bool] = False,
    ) -> Union[torch.Tensor, Dict, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass of GQA.

        :param x: Input tensor of shape :math:`(B, N, D)` where :math:`N` is the Sequence Length,
            :math:`B` is the batch size, and :math:`D` is the embedding dimension ``d_model``.
        :type x: torch.Tensor

        :param mask: If specified, a 2D or 4D mask preventing attention to certain positions. Must be of shape
            :math:`(N, N)` or :math:`(B, H, N, N)`, where :math:`B` is the batch size, :math:`H` is the number of heads and
            :math:`N` is the Sequence Length. A 2D mask will be broadcasted across the batch while a 4D mask allows
            for a different mask for each entry in the batch and/or heads dimensions.
            **Note: Should be a boolean mask where True indicates masked positions.**
            When Flash Attention is enabled it is inverted because PyTorch expects True for allowed positions.
        :type mask: torch.BoolTensor, optional

        :param pos: Position indices for RoPE, shape :math:`(N)` or :math:`(B, N)`
        :type pos: torch.LongTensor, optional

        :param flash_attn: Tuple of Arguments for Flash Attention and the Context manager to select which backend to use for scaled dot product attention.
            - bool: Whether to use or not Flash Attention. Default: ``False``
            - Union[List[SDPBackend], SDPBackend]: A backend or list of backends for scaled dot product attention. Default: ``torch.nn.attention.SPDBackend.FLASH_ATTENTION``
            - bool: Whether the ordering of the backends is interpreted as their priority order. Default: ``False``
        :type flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional

        :param return_states: If ``True``, return a dictionary of intermediate tensors. Default: ``False``
        :type return_states: bool, optional

        :param cache: Optional KV cache tuple `(k_prev, v_prev)` of shape `(B, H_kv, L_prev, d)` each.
            For GQA, cache stores compressed key/value heads before repeat_interleave.
        :type cache: Tuple[torch.Tensor, torch.Tensor], optional

        :param return_cache: If True, always return cache tuple even on first pass. Used for building initial cache.
        :type return_cache: bool, optional

        :return: Output tensor of shape :math:`(B, N, D)` if not return_states, else a dict containing
            the keys: {`output`, `queries`, `keys`, `values`, `attn_weights`, `attn_scores`, `output_before_proj` and `input`}.
            If cache is provided or return_cache=True, returns a tuple `(output, new_cache)` where `new_cache` is `(k, v)`.
        :rtype: Union[torch.Tensor, Dict, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]

        """

        B, N, D, H_q, H_kv, G, d = *x.shape, self.n_heads, self.n_kv_heads, self.groups, self.d_head

        q, k, v = self.qkv_proj(x).view(B, N, H_q + (H_kv * 2), d).transpose(1, 2).split([H_q, H_kv, H_kv], dim=1)
        q, k = (self.q_norm(q), self.k_norm(k)) if self.qk_norm else (q, k)

        q, k = self.rope(q, k, pos, pos) if pos is not None else (q, k)

        if cache is not None:
            k_prev, v_prev = cache
            k = torch.cat((k_prev, k), dim=2)
            v = torch.cat((v_prev, v), dim=2)
        new_cache = (k, v)

        L_total = k.shape[2]

        y, A_weights, A_scores = None, None, None
        if flash_attn[0]:
            with torch.nn.attention.sdpa_kernel(backends=flash_attn[1], set_priority=flash_attn[2]):
                y = (
                    F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=(~mask) if mask is not None else None,
                        dropout_p=self.dropout,
                        is_causal=False,
                        scale=self.scale,
                        enable_gqa=True,
                    )
                    .transpose(1, 2)
                    .reshape(B, N, D)
                )
        else:
            k_expanded = k.repeat_interleave(self.groups, dim=1)
            v_expanded = v.repeat_interleave(self.groups, dim=1)

            A_weights = torch.matmul(q, k_expanded.transpose(-1, -2)).mul_(self.scale)
            if mask is not None:
                A_weights = A_weights.masked_fill_(mask, float("-inf"))
            A_scores = A_weights.softmax(dim=-1)

            if self.dropout > 0.0 and self.training:
                A_scores = F.dropout(A_scores, p=self.dropout, inplace=False)

            y = torch.matmul(A_scores, v_expanded).transpose(1, 2).reshape(B, N, D)

        out = self.out_proj(y)

        if return_cache:
            if return_states:
                result = {
                    "output": out,
                    "queries": q,
                    "keys": k,
                    "values": v,
                    "output_before_proj": y,
                    "input": x,
                }
                if not flash_attn[0]:
                    result["attn_weights"] = A_weights
                    result["attn_scores"] = A_scores
                return result, new_cache
            return out, new_cache

        if return_states:
            result = {
                "output": out,
                "queries": q,
                "keys": k,
                "values": v,
                "output_before_proj": y,
                "input": x,
            }
            if not flash_attn[0]:
                result["attn_weights"] = A_weights
                result["attn_scores"] = A_scores
            return result
        return out


class CrossAttention(nn.Module):
    """
    Cross-Attention module using optimized scaled_dot_product_attention.

    :param d_model: Model dimension.
    :type d_model: int

    :param n_heads: Number of attention heads. ``d_model`` is split across ``n_heads``.
    :type n_heads: int

    :param dropout: Dropout probability on attention weights. Default: ``0.0``
    :type dropout: float, optional

    :param attn_bias: Whether to use bias in linear projections. Default: ``False``
    :type attn_bias: bool, optional

    :param qk_norm: Whether to apply RMSNorm to queries and keys. Default: ``True``
    :type qk_norm: bool, optional

    :param layer_idx: Index of the layer (used for debugging/logging).
    :type layer_idx: int, optional

    :param pos_encoding: Positional Encoding to use. Default: ``RoPE``
    :type pos_encoding: str, optional

    :param pos_encoding_kwargs: Dictionary of additional arguments for positional encoding.
    :type pos_encoding_kwargs: Dict, optional

    :param max_seq_len: Maximum sequence length for RoPE.
    :type max_seq_len: int

    Note: CrossAttention does not support KV caching. This is a known limitation for
        autoregressive decoding with cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: Optional[float] = 0.0,
        attn_bias: Optional[bool] = False,
        qk_norm: Optional[bool] = True,
        layer_idx: int = 0,
        pos_encoding: str = "RoPE",
        pos_encoding_kwargs: Dict = {},
        max_seq_len: int = 1024,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head, self.layer_idx = d_model, n_heads, d_model // n_heads, layer_idx
        self.qk_norm = qk_norm
        self.pos_encoding = pos_encoding

        self.q_proj, self.kv_proj, self.out_proj = (
            nn.Linear(self.d_model, self.d_model, bias=attn_bias),
            nn.Linear(self.d_model, self.d_model * 2, bias=attn_bias),
            nn.Linear(self.d_model, self.d_model, bias=attn_bias),
        )

        if pos_encoding == "RoPE":
            self.rope = RoPE(max_seq_len, self.d_head, **pos_encoding_kwargs)
        elif pos_encoding == "PartialRoPE":
            self.rope = PartialRoPE(max_seq_len, self.d_head, **pos_encoding_kwargs)
        elif pos_encoding == "ALiBi":
            self.alibi = ALiBi(max_seq_len, self.n_heads, **pos_encoding_kwargs)
        elif pos_encoding == "None" or pos_encoding is None:
            self.rope = None
            self.alibi = None
        else:
            raise ValueError(f"Unknown positional encoding: {pos_encoding}")

        self.scale = self.d_head**-0.5
        self.dropout = dropout

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

        :param queries: Input tensor of shape :math:`(B, Lq, D)` where :math:`Lq` is the Sequence Length for the query sequence,
            :math:`B` is the batch size, and :math:`D` is the embedding dimension ``d_model``.
        :type queries: torch.Tensor

        :param kv: Input tensor of shape :math:`(B, Lq, D)` where :math:`Lk` is the Sequence Length for the key/value sequence,
            :math:`B` is the batch size, and :math:`D` is the embedding dimension ``d_model``.
        :type kv: torch.Tensor

        :param mask: If specified, a 2D or 4D mask preventing attention to certain positions. Must be of shape
            :math:`(Lq, Lk)` or :math:`(B, H, Lq, Lk)`, where :math:`B` is the batch size, :math:`H` is the number of heads,
            :math:`Lq` is the Sequence Length of the query sequence and :math:`Lk` is the Sequence Length of the key/value sequence.
            A 2D mask will be broadcasted across the batch while a 4D mask allows for a different mask for each entry
            in the batch and/or heads dimensions.
            **Note: Should be a boolean mask where True indicates masked positions.**
            When Flash Attention is enabled it is inverted because PyTorch expects True for allowed positions.
        :type mask: torch.BoolTensor, optional

        :param pos_q: Position indices for Queries, shape :math:`(Lq)` or :math:`(B, Lq)`
        :type pos_q: torch.LongTensor, optional

        :param pos_k: Position indices for Keys, shape :math:`(Lk)` or :math:`(B, Lk)`
        :type pos_k: torch.LongTensor, optional

        :param flash_attn: Tuple of Arguments for Flash Attention and the Context manager to select which backend to use for scaled dot product attention.
            - bool: Whether to use or not Flash Attention. Default: ``False``
            - Union[List[SDPBackend], SDPBackend]: A backend or list of backends for scaled dot product attention. Default: ``torch.nn.attention.SPDBackend.FLASH_ATTENTION``
            - bool: Whether the ordering of the backends is interpreted as their priority order. Default: ``False``
        :type flash_attn: Tuple[bool, Union[list[torch.nn.attention.SDPBackend], torch.nn.attention.SDPBackend], bool], optional

        :param return_states: If True, return dictionary of intermediates tensors. Default: False
        :type return_states: bool, optional

        :return: Output tensor of shape :math:`(B, N, D)` if not return_states, else a dict containing
            the keys: {`output`, `queries`, `keys`, `values`, `attn_weights`, `attn_scores`, `output_before_proj` and `input`} where `input` is a tuple (queries, kv)
        :rtype: Union[torch.Tensor, Dict]

        """

        B, Lq, D, Lk, H, d = *queries.shape, kv.shape[1], self.n_heads, self.d_head

        q, k, v = self.q_proj(queries).view(B, Lq, H, d).transpose(1, 2), *self.kv_proj(kv).view(
            B, Lk, H, d * 2
        ).transpose(1, 2).chunk(2, dim=-1)
        q, k = (self.q_norm(q), self.k_norm(k)) if self.qk_norm else (q, k)

        if hasattr(self, "rope") and self.rope is not None and pos_q is not None and pos_k is not None:
            q, k = self.rope(q, k, pos_q, pos_k)
        elif hasattr(self, "alibi") and self.alibi is not None:
            pass

        # Get ALiBi bias if applicable
        alibi_bias = None
        if hasattr(self, "alibi") and self.alibi is not None:
            alibi_bias = self.alibi(Lq + Lk, device=q.device, dtype=q.dtype)[:, :, :Lq, :Lk]

        y, A_weights, A_scores = None, None, None
        if flash_attn[0]:
            with torch.nn.attention.sdpa_kernel(backends=flash_attn[1], set_priority=flash_attn[2]):
                attn_mask_sdpa = (~mask) if mask is not None else None
                y = (
                    F.scaled_dot_product_attention(
                        q,
                        k,
                        v,
                        attn_mask=attn_mask_sdpa,
                        attn_bias=alibi_bias,
                        dropout_p=self.dropout,
                        is_causal=False,
                        scale=self.scale,
                        enable_gqa=False,
                    )
                    .transpose(1, 2)
                    .reshape(B, Lq, D)
                )
        else:
            A_weights = torch.matmul(q, k.transpose(-1, -2)).mul_(self.scale)
            if alibi_bias is not None:
                A_weights = A_weights + alibi_bias
            if mask is not None:
                A_weights = A_weights.masked_fill_(mask, float("-inf"))
            A_scores = A_weights.softmax(dim=-1)

            if self.dropout > 0.0 and self.training:
                A_scores = F.dropout(A_scores, p=self.dropout, inplace=False)

            y = torch.matmul(A_scores, v).transpose(1, 2).reshape(B, Lq, D)

        out_proj_y = self.out_proj(y)
        if return_states:
            result = {
                "output": out_proj_y,
                "queries": q,
                "keys": k,
                "values": v,
                "output_before_proj": y,
                "input": (queries, kv),
            }
            if not flash_attn[0]:
                result["attn_weights"] = A_weights
                result["attn_scores"] = A_scores
            return result
        return out_proj_y
