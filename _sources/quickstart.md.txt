# Quick Start

## Basic Usage

```python
import torch
from transformer import Transformer, TransformerConfig

# 1. Define configuration
config = TransformerConfig(
    vocab_size=32000,
    n_layers=6,
    n_heads=8,
    d_model=512,
    d_ff=2048,
    max_seq_len=1024,
    attn_type="MHA", # or "GQA", "MQA", "CrossAttention", etc
    attn_qk_norm=True,
    tied_weights=False,
)

# 2. Create model
model = Transformer(config)

# 3. Prepare input
batch_size, seq_len = 2, 128
input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

# 4. Forward pass
outputs = model(input_ids)
logits = outputs.logits # shape: (2, 128, 32000)
print(logits.shape)
```

## Training Loop Example

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
for epoch in range(3):
    # ... dataloader ...
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
```

## Using HuggingFace Generation

Since the model inherits from `PreTrainedModel`, you can use the built-in generation methods:

```python
from transformers import GenerationConfig

gen_config = GenerationConfig(max_new_tokens=128, do_sample=True, temperature=0.8)
output_ids = model.generate(input_ids, generation_config=gen_config)
```

## Returning Intermediate States

```python
outputs = model(input_ids, return_states=True)
hidden_states = outputs.hidden_states   # tuple of (embedding, list of hidden_states for each layer)
```
