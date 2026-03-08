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
# Note: The keys "attn_weights" and "attn_scores" will not be available when FlashAttention is used

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

## Training a Simple Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from transformer import Transformer, TransformerConfig

# Model configuration
config = TransformerConfig(
    n_layers=4,
    n_heads=4,
    d_model=256,
    vocab_size=1000,
    seq_len=128,
    max_seq_len=512
)
model = Transformer(config)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)

# Dummy data
batch_size = 8
seq_len = 64
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# Training step
model.train()
optimizer.zero_grad()
outputs = model(input_ids, labels=labels)
loss = outputs.loss
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
```

## Text Generation with HuggingFace GenerationMixin

The model inherits from `GenerationMixin`, so you can use `generate()`.

```python
# Assume model is trained or loaded
model.eval()

# Prompt
prompt = torch.tensor([[1, 2, 3, 4]])  # (B, N)

# Generate
with torch.no_grad():
    generated = model.generate(
        input_ids=prompt,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
        top_k=40
    )
print(generated.shape)
```

## Using Flash Attention

Flash Attention can be enabled via the `flash_attn` tuple passed in the forward call. The tuple contains:
1. `use_flash` (bool): whether to use flash attention.
2. `backends`: a backend or list of backends (e.g., `torch.nn.attention.SDPBackend.FLASH_ATTENTION`).
3. `set_priority` (bool): whether the list order is priority.

```python
from torch.nn.attention import SDPBackend

# Enable flash attention with default backend
flash_attn = (True, SDPBackend.FLASH_ATTENTION, False)

outputs = model(input_ids, flash_attn=flash_attn)

# Use a list of backends with priority
flash_attn = (True, [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH], True)
outputs = model(input_ids, flash_attn=flash_attn)
```
_**Note:** When flash attention is used, `attn_weights` and `attn_scores` are not returned (they are `None` in the state dict)._


## Custom Attention Mask
You can provide any boolean mask to control which positions attend to which.

```python
# Causal mask (upper triangular)
seq_len = 16
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

# Padding mask (batch-specific)
pad_mask = torch.randint(0, 2, (2, seq_len)).bool()  # (B, N)

# Combine masks (broadcasted)
# For 4D mask: (B, H, N, N)
combined_mask = causal_mask.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)
combined_mask = combined_mask | pad_mask.unsqueeze(1).unsqueeze(2)  # (B,1,N,N)

outputs = model(input_ids, attn_mask=combined_mask)
```
