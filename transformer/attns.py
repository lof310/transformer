import torch
import torch.nn as nn
import torch.nn.functional as F

class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_bias: bool = False, qk_norm: bool = True, layer_idx: int = 0, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head, self.layer_idx = d_model, n_heads, d_model//n_heads, layer_idx
        self.qk_norm = qk_norm

        self.qkv_proj = nn.Linear(self.d_model, self.d_model*3, bias=attn_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=attn_bias)

        self.rope = RoPE(max_block_size, self.d_head)
        self.scale = self.d_head**-0.5

        if qk_norm:
            self.q_norm, self.k_norm = nn.RMSNorm(self.d_head), nn.RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, pos: torch.Tensor = None, return_states: bool = False):
        B, N, D, H, d = *x.shape, self.n_heads, self.d_head

        q, k, v = self.qkv_proj(x).view(B, N, H, d*3).transpose(1,2).chunk(3, dim=-1)
        q, k = (self.q_norm(q), self.k_norm(k)) if self.qk_norm else (q,k)
        q, k = self.rope(q, k, pos, pos) if pos is not None else (q, k)
        A_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        A_scores = F.softmax(A_weights.masked_fill_(mask, float('-inf')), dim=-1) if mask is not None else F.softmax(A_weights, dim=-1)
        y = torch.matmul(A_scores, v).transpose(1, 2).contiguous().view(B, N, D)
        if return_states:
            return {"output": self.out_proj(y), "queries": q, "keys": k, "values": v, "attn_weights": A_weights, "attn_scores": A_scores, "output_before_proj": y, "input": x}
        else:
            return self.out_proj(y)

class GQA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, attn_bias: bool = False, qk_norm: bool = True, layer_idx: int = 0, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.d_model, self.n_heads, self.n_kv_heads, self.groups, self.d_head, self.layer_idx = d_model, n_heads, n_kv_heads, n_heads//n_kv_heads, d_model//n_heads, layer_idx
        self.qk_norm = qk_norm

        self.qkv_proj = nn.Linear(self.d_model, d_model+(self.d_head*self.n_kv_heads*2), bias=attn_bias)
        self.out_proj = nn.Linear(self.d_model, self.d_model, bias=attn_bias)

        self.rope = RoPE(max_block_size, self.d_head)
        self.scale = self.d_head**-0.5

        if qk_norm:
            self.q_norm, self.k_norm = nn.RMSNorm(self.d_head), nn.RMSNorm(self.d_head)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None, pos: torch.Tensor = None, return_states: bool = False):
        B, N, D, H_q, H_kv, G, d = *x.shape, self.n_heads, self.n_kv_heads, self.groups, self.d_head

        q, k, v = self.qkv_proj(x).view(B, N, H_q+(H_kv*2), d).transpose(1,2).split([H_q, H_kv, H_kv], dim=1)
        q, k = (self.q_norm(q), self.k_norm(k)) if qk_norm else q, k
        q, k = self.rope(q, k, pos, pos) if pos is not None else (q, k)
        k, v = k.repeat_interleave(G, dim=1), v.repeat_interleave(G, dim=1)
        A_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        A_scores = F.softmax(A_weights.masked_fill_(mask, float('-inf')), dim=-1) if mask is not None else F.softmax(A_weights, dim=-1)
        y = torch.matmul(A_scores, v).transpose(1, 2).contiguous().view(B, N, D)
        if return_states:
            return {"output": self.out_proj(y), "queries": q, "keys": k, "values": v, "attn_weights": A_weights, "attn_scores": A_scores, "output_before_proj": y, "input": x}
        else:
            return self.out_proj(y)

class CrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, attn_bias: bool = False, qk_norm: bool = True, layer_idx: int = 0, max_seq_len: int = 1024):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model, self.n_heads, self.d_head, self.layer_idx = d_model, n_heads, d_model//n_heads, layer_idx
        self.qk_norm = qk_norm

        self.q, self.kv_proj, self.out_proj = (
            nn.Linear(self.d_model, self.d_model, bias=attn_bias),
            nn.Linear(self.d_model, self.d_model*2, bias=attn_bias),
            nn.Linear(self.d_model, self.d_model, bias=attn_bias)
        )

        self.rope = RoPE(max_block_size, self.d_head)
        self.scale = self.d_head**-0.5

        if qk_norm:
            self.q_norm, self.k_norm = nn.RMSNorm(self.d_head), nn.RMSNorm(self.d_head)

    def forward(self, queries: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor = None, pos_q: torch.Tensor = None, pos_k: torch.Tensor = None, return_states: bool = False):
        B, Lq, D, Lk, H, d = *q.shape, kv.shape[1], self.n_heads, self.d_head

        q, k, v = self.q_proj(queries).view(B, Lq, H, d).transpose(1,2), *self.kv_proj(kv).view(B, Lk, H, d*2).transpose(1,2).chunk(2, dim=-1)
        q, k = (self.q_norm(q), self.k_norm(k)) if self.qk_norm else (q,k)
        q, k = self.rope(q, k, pos, pos) if pos is not None else (q, k)
        A_weights = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        A_scores = F.softmax(A_weights.masked_fill_(mask, float('-inf')), dim=-1) if mask is not None else F.softmax(A_weights dim=-1)
        y = torch.matmul(A_scores, v).transpose(1, 2).contiguous().view(B, Lq, D)
        if return_states:
            return {"output": self.out_proj(y), "queries": q, "keys": k, "values": v, "attn_weights": A_weights, "attn_scores": A_scores, "output_before_proj": y, "input": (queries, kv)}
        else:
            return self.out_proj(y)
