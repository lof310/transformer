# Usage Examples
**This section provides complete and diverse usage examples of all modules and classes**

## Basic Usage

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer, TransformerConfig

# Define configuration
config = TransformerConfig(
    n_layers=6,
    n_heads=8,
    d_model=384,
    vocab_size=65,
    seq_len=256,
    max_seq_len=1024,
    tied_weights=False
)

# Create model
model = Transformer(config)

# Prepare input
batch_size, seq_len = 2, 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass
outputs = model(input_ids)
logits = outputs.logits # shape: [B, N, V]
print(logits.shape)
```

## Visualization
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer, TransformerConfig

import matplotlib.pyplot as plt

# Define configuration
config = TransformerConfig(
    n_layers=6,
    n_heads=8,
    d_model=384,
    vocab_size=65,
    seq_len=256,
    max_seq_len=1024,
    tied_weights=True
)

# Create model
model = Transformer(config)

# Prepare input
batch_size, seq_len = 1, 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Forward pass
with torch.no_grad():
    output = model(input_ids, return_states=True)

logits = output.logits # shape: [batch_size, seq_len, vocab_size]
hidden_states = output.hidden_states # Tuple: (input_embs, hidden_states)
hidden_states = hidden_states[1] # The hidden_states

# The hidden_states is a list of Dictionaries for each layer containing the following dictionary structure for each layer:
r"""
{
    "output": torch.Tensor,  # Shape: [B, N, D]
    "attn_output": {
        "output": torch.Tensor,       # Shape: [B, N, D]
        "queries": torch.Tensor,      # Shape: [B, H, N, d]
        "keys": torch.Tensor,         # Shape: [B, H, N, d]
        "values": torch.Tensor,       # Shape: [B, H, N, d]
        "attn_weights": torch.Tensor, # Shape: [B, H, N, N] or [B, H, Lq, Lk] depending on if you use the Cross Attention module
        "attn_scores": torch.Tensor,  # Shape: [B, H, N, N] or [B, H, Lq, Lk] depending on if you use the Cross Attention module
        "output_before_proj": torch.Tensor, # Shape: [B, N, D]
        "input": Union[torch.Tensor, Tuple] # Can be Tensor of shape [B, N, D] or Tuple (query, kv) both of shape [B, N, D]
                                             # depending on if you use the Cross Attention module
    },
    "ffn_output": {
        "output": torch.Tensor, # Shape: [B, N, D]
        "y1": torch.Tensor, # Shape: [B, N, D_ff//2]
        "y2": torch.Tensor, # Shape: [B, N, D_ff//2]
        "input": torch.Tensor # Shape: [B, N, D]
    }
}
"""

layer, batch, head = (0, 0, 0)

# Visualization of Attention Scores
# Note: Use .detach() always to avoid RuntimeError
attn_matrix = hidden_states[layer]["attn_output"]["attn_scores"][batch, head].detach().cpu() # Shape [N, N]

plt.imshow(attn_matrix) # No need to convert to numpy this is handled automatically
plt.colorbar()
plt.show()

# Visualization of Attention Weights
attn_matrix = hidden_states[layer]["attn_output"]["attn_weights"][batch, head].detach().cpu() # Shape [N, N]

plt.imshow(attn_matrix)
plt.colorbar()
plt.show()

# Visualization of the weights of the first linear layer of SwiGLU as a HeatMap
weights = model.blocks[layer].ffn.W1.weight.mT.detach().cpu() # Shape [d_ff, d_model]

plt.imshow(weights)
plt.colorbar()
plt.show()

# Visualization of the weights of the first linear layer of SwiGLU as lines
weights = weights.mT # Shape [d_model, d_ff]

plt.plot(weights)
plt.show()
```
