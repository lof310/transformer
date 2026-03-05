import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.W1 = nn.Linear(d_model, d_ff, bias=bias)
        self.silu = nn.SiLU()
        self.W2 = nn.Linear(d_ff // 2, d_model, bias=bias)

    def forward(self, x: torch.Tensor, return_states: bool = False):
        y1, y2 = self.W1(x).chunk(2, dim=-1)
        if return_states:
            return {"output": self.W2(y1 * self.silu(y2)), "y1": y1, "y2": y2, "input": x}
        else:
            return self.W2(y1 * self.silu(y2))

class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, bias: bool = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=bias),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=bias),
        )

    def forward(self, x):
        if return_states:
            h1 = self.net[0](x)
            h2 = self.net[1](h1)
            out = self.net[2](h2)
            return {"output": out, "input": x, "h1": h1, "h2": h2}
        else:
            return self.net(x)
