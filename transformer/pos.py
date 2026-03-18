from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    r"""
    Rotary Position Embedding (RoPE) module.

    Args:
        max_seq_len (int): Maximum sequence length for which to precompute frequencies.
        d_head (int): Dimension per head (must be even).
        rope_base (float): Base for the exponential frequency calculation. Default: ``10000.0``
        persistent (bool): Whether to register the precomputed cos/sin as persistent buffers. Default: ``True``
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

        Args:
            q (torch.Tensor): Query tensor of shape :math:`(B, H, N, d)`
            k (torch.Tensor): Key tensor of shape :math:`(B, H, N, d)`
            pos_q (torch.Tensor): Positions for queries, shape :math:`(N,)` or :math:`(B, N)`
            pos_k (torch.Tensor): Positions for keys, shape :math:`(N,)` or :math:`(B, N)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated queries and keys.
        """
        pos_q, pos_k = pos_q.long(), pos_k.long()
        cos_q, sin_q = self.cos[pos_q].to(q.device, dtype=q.dtype), self.sin[pos_q].to(q.device, dtype=q.dtype)
        cos_k, sin_k = self.cos[pos_k].to(k.device, dtype=k.dtype), self.sin[pos_k].to(k.device, dtype=k.dtype)

        return self._rot(q, cos_q, sin_q), self._rot(k, cos_k, sin_k)


class PartialRoPE(nn.Module):
    r"""
    Partial Rotary Positional Embedding (PartialRoPE).

    This module applies rotary positional embeddings (RoPE) to only a fraction
    of the head dimension (the first `d_rot` channels) while leaving the
    remaining channels unchanged. The rotation is applied pairwise (as in
    standard RoPE) and supports position indices provided either per-sequence
    (shape :math:`(N,)`) or per-batch (shape :math:`(B, N)`).

    Args:
        max_seq_len (int): Maximum sequence length for which to precompute cos/sin.
        d_head (int): Dimension per head (must be even).
        rot_frac (float, optional): Fraction of head dimensions to rotate in (0, 1].
            The number of rotated dimensions is `int(d_head * rot_frac)` rounded
            down to the nearest even integer.
        rope_base (float, optional): Base for the exponential frequency calculation. Default: 10000.0
        persistent (bool, optional): Whether to register cos/sin as persistent buffers. Default: True
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
        assert 0.0 < rot_frac <= 1.0, "rot_frac must be in (0, 1]"

        # Determine number of rotated dimensions (ensure even)
        d_rot = int(d_head * float(rot_frac))
        d_rot = d_rot - (d_rot % 2)
        self.d_head = d_head
        self.d_rot = d_rot
        self.d_pass = d_head - d_rot

        # Precompute cos/sin only for the rotated half-dimensions.
        if self.d_rot > 0:
            # Use d_head in denominator to keep frequency scale consistent with full RoPE.
            half_rot = self.d_rot // 2
            inv_freq = 1.0 / (rope_base ** (torch.arange(0, half_rot, dtype=torch.float32) * 2.0 / d_head))
            pos = torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1)  # (max_seq_len, 1)
            freqs = pos * inv_freq.unsqueeze(0)  # (max_seq_len, half_rot)
            self.register_buffer("cos", torch.cos(freqs), persistent=persistent)  # (max_seq_len, half_rot)
            self.register_buffer("sin", torch.sin(freqs), persistent=persistent)  # (max_seq_len, half_rot)
        else:
            # Keep empty buffers to preserve attribute existence and API
            self.register_buffer("cos", torch.empty(0), persistent=persistent)
            self.register_buffer("sin", torch.empty(0), persistent=persistent)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, pos_q: torch.LongTensor, pos_k: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Apply partial RoPE to queries and keys.

        Args:
            q (torch.Tensor): Query tensor of shape :math:`(B, H, N, d)`
            k (torch.Tensor): Key tensor of shape :math:`(B, H, N, d)`
            pos_q (torch.Tensor): Positions for queries, shape :math:`(N,)` or :math:`(B, N)`
            pos_k (torch.Tensor): Positions for keys, shape :math:`(N,)` or :math:`(B, N)`

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated queries and keys.
        """
        pos_q, pos_k = pos_q.long(), pos_k.long()

        # If no rotation requested, return inputs unchanged
        if self.d_rot == 0:
            return (q, k)

        # Extract rotated and passthrough slices
        q_rot, k_rot = q[..., : self.d_rot], k[..., : self.d_rot]  # (B, H, N, d_rot)
        q_pass, k_pass = q[..., self.d_rot :], k[..., self.d_rot :]  # (B, H, N, d_pass) may be empty

        # Prepare cos/sin for queries and keys
        # (N, d_half) or (B, N, d_half)
        cos_q, sin_q = self.cos[pos_q].to(q.device, dtype=q.dtype), self.sin[pos_q].to(q.device, dtype=q.dtype)
        cos_k, sin_k = self.cos[pos_k].to(k.device, dtype=k.dtype), self.sin[pos_k].to(k.device, dtype=k.dtype)

        # Number of pairs in rotated slice
        d_half = self.d_rot // 2  # number of complex pairs

        # Reshape rotated slices to (..., d_half, 2) to operate on pairs efficiently.
        # This avoids slicing with ::2 which can be slightly less cache-friendly.
        # After reshape: (B, H, N, d_half, 2)
        q_pairs, k_pairs = q_rot.view(*q_rot.shape[:-1], d_half, 2), k_rot.view(*k_rot.shape[:-1], d_half, 2)

        # Align cos/sin for broadcasting to (B, 1, N, d_half)
        # If cos has shape (N, d_half) -> unsqueeze to (1, 1, N, d_half)
        # If cos has shape (B, N, d_half) -> unsqueeze to (1, 1, B, N, d_half) then we will
        # permute to (B, 1, N, d_half) for broadcasting with (B, H, N, d_half, 2).
        if cos_q.dim() == 2:
            # pos provided as (N,)
            cos_q_b = cos_q.unsqueeze(0).unsqueeze(0)  # (1, 1, N, d_half)
            sin_q_b = sin_q.unsqueeze(0).unsqueeze(0)
        else:
            # pos provided as (B, N)
            # cos_q: (B, N, d_half) -> (B, 1, N, d_half)
            cos_q_b = cos_q.unsqueeze(1)
            sin_q_b = sin_q.unsqueeze(1)

        if cos_k.dim() == 2:
            cos_k_b = cos_k.unsqueeze(0).unsqueeze(0)
            sin_k_b = sin_k.unsqueeze(0).unsqueeze(0)
        else:
            cos_k_b = cos_k.unsqueeze(1)
            sin_k_b = sin_k.unsqueeze(1)

        # Compute rotated pairs using pair arithmetic:
        cos_q_b, sin_q_b = cos_q_b.unsqueeze(-1), sin_q_b.unsqueeze(-1)  # (1 or B, 1, N, d_half, 1)
        cos_k_b, sin_k_b = cos_k_b.unsqueeze(-1), sin_k_b.unsqueeze(-1)

        # Perform rotation with fused elementwise ops (keeps memory overhead low)
        q0, q1 = q_pairs[..., 0:1], q_pairs[..., 1:2]  # (B, H, N, d_half, 1)
        q_rotated_pairs = torch.cat([q0 * cos_q_b - q1 * sin_q_b, q0 * sin_q_b + q1 * cos_q_b], dim=-1)

        k0, k1 = k_pairs[..., 0:1], k_pairs[..., 1:2]
        k_rotated_pairs = torch.cat([k0 * cos_k_b - k1 * sin_k_b, k0 * sin_k_b + k1 * cos_k_b], dim=-1)

        # Restore last-dimension layout: (..., d_rot)
        q_rot = q_rotated_pairs.view(*q_rot.shape[:-1], self.d_rot)
        k_rot = k_rotated_pairs.view(*k_rot.shape[:-1], self.d_rot)

        # Concatenate rotated and passthrough parts (passthrough may be empty)
        q_out, k_out = None, None
        if self.d_pass == 0:
            q_out = q_rot
            k_out = k_rot
        else:
            q_out = torch.cat([q_rot, q_pass], dim=-1)
            k_out = torch.cat([k_rot, k_pass], dim=-1)

        return q_out, k_out


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

    Args:
        max_seq_len (int): nominal maximum sequence length (used for internal
            checks; biases can be computed for longer sequences on the fly).
        n_heads (int): number of attention heads.
        base (float): base used in slope schedule. Default follows the paper:
            slopes = 2^{-8 * h / n_heads}.
        persistent (bool): whether to register slopes as persistent buffers.
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

        Args:
            seq_len (int): sequence length L for which to compute biases.
            device (torch.device, optional): device for the returned tensor.
                If None, uses the device of the stored slopes buffer.
            dtype (torch.dtype, optional): dtype for the returned tensor.
                If None, uses the dtype of the stored slopes buffer.

        Returns:
            torch.Tensor: bias tensor of shape (1, n_heads, L, L) with dtype/device
            as requested. This can be added to attention logits of shape
            (B, n_heads, L, L).
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
        # Using int64 then cast to slopes dtype for numerical stability
        idx = torch.arange(L, device=slopes.device, dtype=torch.long)
        rel = idx.unsqueeze(1) - idx.unsqueeze(0)  # (L, L), positive when i>j
        rel = rel.to(slopes.dtype)  # cast to float

        # Compute per-head biases: (-slopes[:, None, None]) * rel[None, :, :]
        # Result shape: (n_heads, L, L)
        bias = -slopes.view(self.n_heads, 1, 1) * rel.view(1, L, L)

        # Return with leading batch-like dim for easy broadcasting: (1, n_heads, L, L)
        return bias.unsqueeze(0)
