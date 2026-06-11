from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    r"""
    Rotary Position Embedding (RoPE) module.

    :param max_seq_len: Maximum sequence length for which to precompute frequencies.
    :type max_seq_len: int

    :param d_head: Dimension per head (must be even).
    :type d_head: int

    :param rope_base: Base for the exponential frequency calculation. Default: ``10000.0``
    :type rope_base: float, optional

    :param persistent: Whether to register the precomputed cos/sin as persistent buffers. Default: ``True``
    :type persistent: bool, optional
    """

    def __init__(self, max_seq_len: int, d_head: int, rope_base: float = 10000.0, persistent: bool = True):
        super().__init__()
        assert d_head % 2 == 0
        self.half = d_head // 2
        inv_freq, pos = (
            1.0 / (rope_base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)),
            torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1),
        )
        freqs = pos * inv_freq.unsqueeze(0)
        self.register_buffer("cos", torch.cos(freqs), persistent=persistent)
        self.register_buffer("sin", torch.sin(freqs), persistent=persistent)

    def _rot(self, x, cos, sin):
        x1, x2 = x[..., ::2], x[..., 1::2]
        if cos.dim() == 2:
            cos = cos.unsqueeze(0).unsqueeze(0)
            sin = sin.unsqueeze(0).unsqueeze(0)
        else:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, pos_q: torch.LongTensor, pos_k: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply rotary position embeddings to queries and keys.

        :param q: Query tensor of shape :math:`(B, H, N, d)`
        :type q: torch.Tensor

        :param k: Key tensor of shape :math:`(B, H, N, d)`
        :type k: torch.Tensor

        :param pos_q: Positions for queries, shape :math:`(N,)` or :math:`(B, N)`
        :type pos_q: torch.LongTensor

        :param pos_k: Positions for keys, shape :math:`(N,)` or :math:`(B, N)`
        :type pos_k: torch.LongTensor

        :return: Rotated queries and keys.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        pos_q, pos_k = pos_q.long(), pos_k.long()
        cos_q, sin_q = self.cos[pos_q].to(q.device, dtype=q.dtype), self.sin[pos_q].to(q.device, dtype=q.dtype)
        cos_k, sin_k = self.cos[pos_k].to(k.device, dtype=k.dtype), self.sin[pos_k].to(k.device, dtype=k.dtype)

        return self._rot(q, cos_q, sin_q), self._rot(k, cos_k, sin_k)


class PartialRoPE(nn.Module):
    r"""
    Partial Rotary Positional Embedding (PartialRoPE).

    Applies RoPE to only a fraction of the head dimension while leaving the rest unchanged.

    :param max_seq_len: Maximum sequence length for which to precompute cos/sin.
    :type max_seq_len: int

    :param d_head: Dimension per head (must be even).
    :type d_head: int

    :param rot_frac: Fraction of head dimensions to rotate in (0, 1]. Default: 0.5
    :type rot_frac: float, optional

    :param rope_base: Base for the exponential frequency calculation. Default: 10000.0
    :type rope_base: float, optional

    :param persistent: Whether to register cos/sin as persistent buffers. Default: True
    :type persistent: bool, optional
    """

    def __init__(
        self,
        max_seq_len: int,
        d_head: int,
        rot_frac: float = 0.5,
        rope_base: float = 10000.0,
        persistent: bool = True,
    ):
        super().__init__()
        assert d_head % 2 == 0, "d_head must be even"
        assert 0.0 <= rot_frac <= 1.0, "rot_frac must be in [0, 1]"

        d_rot = int(d_head * float(rot_frac))
        d_rot = d_rot - (d_rot % 2)
        self.d_head = d_head
        self.d_rot = d_rot
        self.d_pass = d_head - d_rot

        if self.d_rot > 0:
            half_rot = self.d_rot // 2
            inv_freq = 1.0 / (rope_base ** (torch.arange(0, half_rot, dtype=torch.float32) * 2.0 / d_head))
            pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)
            freqs = pos * inv_freq.unsqueeze(0)
            self.register_buffer("cos", torch.cos(freqs), persistent=persistent)
            self.register_buffer("sin", torch.sin(freqs), persistent=persistent)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, pos_q: torch.LongTensor, pos_k: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply partial RoPE to queries and keys.

        :param q: Query tensor of shape :math:`(B, H, N, d)`
        :type q: torch.Tensor

        :param k: Key tensor of shape :math:`(B, H, N, d)`
        :type k: torch.Tensor

        :param pos_q: Positions for queries, shape :math:`(N,)` or :math:`(B, N)`
        :type pos_q: torch.LongTensor

        :param pos_k: Positions for keys, shape :math:`(N,)` or :math:`(B, N)`
        :type pos_k: torch.LongTensor

        :return: Rotated queries and keys.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        if self.d_rot == 0:
            return (q, k)

        pos_q, pos_k = pos_q.long(), pos_k.long()

        q_rot, k_rot = q[..., : self.d_rot], k[..., : self.d_rot]
        q_pass, k_pass = q[..., self.d_rot :], k[..., self.d_rot :]

        cos_q, sin_q = self.cos[pos_q].to(q.device, dtype=q.dtype), self.sin[pos_q].to(q.device, dtype=q.dtype)
        cos_k, sin_k = self.cos[pos_k].to(k.device, dtype=k.dtype), self.sin[pos_k].to(k.device, dtype=k.dtype)

        d_half = self.d_rot // 2
        q_pairs, k_pairs = q_rot.view(*q_rot.shape[:-1], d_half, 2), k_rot.view(*k_rot.shape[:-1], d_half, 2)

        if cos_q.dim() == 2:
            cos_q_b = cos_q.unsqueeze(0).unsqueeze(0)
            sin_q_b = sin_q.unsqueeze(0).unsqueeze(0)
        else:
            cos_q_b = cos_q.unsqueeze(1)
            sin_q_b = sin_q.unsqueeze(1)

        if cos_k.dim() == 2:
            cos_k_b = cos_k.unsqueeze(0).unsqueeze(0)
            sin_k_b = sin_k.unsqueeze(0).unsqueeze(0)
        else:
            cos_k_b = cos_k.unsqueeze(1)
            sin_k_b = sin_k.unsqueeze(1)

        cos_q_b, sin_q_b = cos_q_b.unsqueeze(-1), sin_q_b.unsqueeze(-1)
        cos_k_b, sin_k_b = cos_k_b.unsqueeze(-1), sin_k_b.unsqueeze(-1)

        q0, q1 = q_pairs[..., 0:1], q_pairs[..., 1:2]
        q_rotated_pairs = torch.cat([q0 * cos_q_b - q1 * sin_q_b, q0 * sin_q_b + q1 * cos_q_b], dim=-1)

        k0, k1 = k_pairs[..., 0:1], k_pairs[..., 1:2]
        k_rotated_pairs = torch.cat([k0 * cos_k_b - k1 * sin_k_b, k0 * sin_k_b + k1 * cos_k_b], dim=-1)

        q_rot = q_rotated_pairs.view(*q_rot.shape[:-1], self.d_rot)
        k_rot = k_rotated_pairs.view(*k_rot.shape[:-1], self.d_rot)

        if self.d_pass == 0:
            return q_rot, k_rot
        return torch.cat([q_rot, q_pass], dim=-1), torch.cat([k_rot, k_pass], dim=-1)


class ALiBi(nn.Module):
    r"""
    Attention with Linear Biases (ALiBi) per-head bias module.

    This module produces additive attention biases that are linear in the
    relative distance between query and key positions. Biases are computed
    per-head using a head-specific slope and returned in a shape that can be
    directly added to attention logits.

    The bias for head h and positions i (query) and j (key) is:

        B_h[i, j] = -m_h * (i - j)

    where m_h is the slope for head h (larger slopes bias attention to local
    positions). The module returns a tensor of shape (1, n_heads, L, L)
    so it can be added to logits of shape (B, n_heads, L, L) with broadcasting.

    :param max_seq_len: nominal maximum sequence length (used for internal
        checks; biases can be computed for longer sequences on the fly).
    :type max_seq_len: int

    :param n_heads: number of attention heads.
    :type n_heads: int

    :param base: base used in slope schedule. Default follows the paper:
        slopes = 2^{-8 * h / n_heads}.
    :type base: float, optional

    :param persistent: whether to register slopes as persistent buffers.
    :type persistent: bool, optional
    """

    def __init__(self, max_seq_len: int, n_heads: int, base: float = 2.0, persistent: bool = True):
        super().__init__()
        assert n_heads > 0, "n_heads must be positive"
        assert max_seq_len > 0, "max_seq_len must be positive"

        self.max_seq_len = int(max_seq_len)
        self.n_heads = int(n_heads)

        h_idx = torch.arange(self.n_heads, dtype=torch.float32)
        slopes = base ** (-8.0 * h_idx / float(self.n_heads))

        self.register_buffer("slopes", slopes, persistent=persistent)

    def forward(
        self, seq_len: int, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None
    ) -> torch.Tensor:
        r"""
        Return ALiBi bias tensor for a square attention of length `seq_len`.

        :param seq_len: sequence length L for which to compute biases.
        :type seq_len: int

        :param device: device for the returned tensor.
            If None, uses the device of the stored slopes buffer.
        :type device: torch.device, optional

        :param dtype: dtype for the returned tensor.
            If None, uses the dtype of the stored slopes buffer.
        :type dtype: torch.dtype, optional

        :return: bias tensor of shape (1, n_heads, L, L) with dtype/device
            as requested. This can be added to attention logits of shape
            (B, n_heads, L, L).
        :rtype: torch.Tensor
        """
        L = int(seq_len)
        if L <= 0:
            raise ValueError("seq_len must be positive")

        # Ensure slopes are on the requested device/dtype
        slopes = self.slopes
        if device is not None:
            slopes = slopes.to(device)
        if dtype is not None:
            slopes = slopes.to(dtype)

        # Create relative distance matrix (i - j) of shape (L, L)
        # Create idx directly in float32 to avoid unnecessary cast
        idx = torch.arange(L, device=slopes.device, dtype=torch.float32)
        rel = idx.unsqueeze(1) - idx.unsqueeze(0)  # (L, L), positive when i>j

        # Compute per-head biases: (-slopes[:, None, None]) * rel[None, :, :]
        # Result shape: (n_heads, L, L)
        bias = -slopes.view(self.n_heads, 1, 1) * rel.view(1, L, L)

        # Return with leading batch-like dim for easy broadcasting: (1, n_heads, L, L)
        return bias.unsqueeze(0)
