import torch
import torch.nn as nn
import torch.nn.functional as F


class RoPE(nn.Module):
    """
    Rotary Position Embedding (RoPE) module.

    Args:
        max_seq_len (int): Maximum sequence length for which to precompute frequencies.
        d_head (int): Dimension per head (must be even).
        rope_base (float): Base for the exponential frequency calculation. Default: 10000.0
        persistent (bool): Whether to register the precomputed cos/sin as persistent buffers. Default: True
    """
    def __init__(self, max_seq_len: int, d_head: int, rope_base: float = 10000.0, persistent: bool = True):
        super().__init__()
        assert d_head % 2 == 0
        self.half = d_head // 2
        inv_freq, pos = (1.0 / (rope_base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head)), torch.arange(max_seq_len, dtype=torch.float32).unsqueeze(1))
        freqs = pos * inv_freq.unsqueeze(0)
        self.register_buffer("cos", torch.cos(freqs), persistent=persistent)
        self.register_buffer("sin", torch.sin(freqs), persistent=persistent)

    def forward(self, q: torch.Tensor, k: torch.Tensor, pos_q: torch.Tensor, pos_k: torch.Tensor):
        """
        Apply rotary position embeddings to queries and keys.

        Args:
            q (torch.Tensor): Query tensor of shape (batch_size, n_heads, seq_len, d_head)
            k (torch.Tensor): Key tensor of shape (batch_size, n_heads, seq_len, d_head)
            pos_q (torch.Tensor): Positions for queries, shape (seq_len,) or (batch_size, seq_len)
            pos_k (torch.Tensor): Positions for keys, shape (seq_len,) or (batch_size, seq_len)

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Rotated queries and keys.
        """
        pos_q, pos_k = pos_q.long(), pos_k.long()
        cos_q, sin_q = self.cos[pos_q].to(q.device, dtype=q.dtype), self.sin[pos_q].to(q.device, dtype=q.dtype)
        cos_k, sin_k = self.cos[pos_k].to(k.device, dtype=k.dtype), self.sin[pos_k].to(k.device, dtype=k.dtype)

        def _rot(x, cos, sin):
            x1, x2 = x[..., ::2], x[..., 1::2]
            if cos.dim() == 2:
                cos = cos.unsqueeze(0).unsqueeze(0)
                sin = sin.unsqueeze(0).unsqueeze(0)
            else:
                cos = cos.unsqueeze(1)
                sin = sin.unsqueeze(1)
            return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

        return _rot(q, cos_q, sin_q), _rot(k, cos_k, sin_k)
