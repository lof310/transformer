# Transformer Library Documentation

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```

```{toctree}
:maxdepth: 1
:caption: Project Info

contributing
```

## Overview

A Polished PyTorch implementation of the current State-Of-The-Art(SOTA) Transformer, designed to be a Baseline for Research and Engineering.

**Features:**

- Fully configurable architecture (layers, heads, dimensions, etc.)
- HuggingFace-compatible API (`PreTrainedModel`, `GenerationMixin`)
- Multi-Head Attention (MHA), Grouped-Query Attention (GQA) and others
- Rotary Position Embeddings (RoPE) and SwiGLU feed-forward
- Optional weight tying, QK normalization, and bias control

## Quick Example

```python
import torch
from transformer import Transformer, TransformerConfig

config = TransformerConfig(vocab_size=32000, n_layers=12, n_heads=16, d_model=1024)
model = Transformer(config)

input_ids = torch.randint(0, 32000, (2, 512))
outputs = model(input_ids)
logits = outputs.logits  # shape: (2, 512, 32000) [batch_size, seq_len, vocab_size]
```

## Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
