# Transformer Library Documentation

```{toctree}
:maxdepth: 2
:caption: Getting Started

installation
quickstart
```

```{toctree}
:maxdepth: 3
:caption: Guide

guide
```

```{toctree}
:maxdepth: 2
:caption: API Reference

api
```

```{toctree}
:maxdepth: 3
:caption: Usage Examples

examples
```

```{toctree}
:maxdepth: 2
:caption: Project Info

contributing
```

## Overview

A Polished PyTorch implementation of the current State-Of-The-Art(SOTA) Transformer, designed to be a Baseline for Research and Engineering.

**Features:**

- Fully configurable architecture (layers, heads, dimensions, etc.)
- HuggingFace-compatible API (`PreTrainedModel`, `GenerationMixin`)
- Multi-Head Attention (MHA), Grouped-Query Attention (GQA) and Cross-Attention
- Rotary Position Embeddings (RoPE), PartialRoPE, and ALiBi
- SwiGLU and MLP feed-forward networks
- Encoder-Decoder architecture support with cross-attention
- Vision Transformer (ViT) support for image processing
- LoRA integration for parameter-efficient fine-tuning
- KV-Cache support for fast incremental decoding
- Optional weight tying, QK normalization, and bias control
- Flash Attention support for accelerated training and inference

### Quick Example

#### Text Processing
```python
import torch
from transformer import Transformer, TransformerConfig

config = TransformerConfig(vocab_size=32000, n_layers=12, n_heads=16, d_model=1024)
model = Transformer(config)

input_ids = torch.randint(0, 32000, (2, 512))
outputs = model(input_ids=input_ids)
logits = outputs.logits  # shape: (2, 512, 32000) [batch_size, seq_len, vocab_size]
```

#### Image Processing with Vision Transformer (ViT)
```python
import torch
from transformer import Transformer, TransformerConfig

# Configure ViT with patch_size and img_size
config = TransformerConfig(
    vocab_size=1000,  # Output vocabulary for classification
    n_layers=12,
    n_heads=16,
    d_model=1024,
    patch_size=16,     # Patch size for image tokenization
    img_size=224,      # Input image size (can be int or tuple)
    in_channels=3,     # Number of input image channels (RGB)
    max_seq_len=512    # Must accommodate num_patches + 1 (cls token)
)
model = Transformer(config)

# Process images: shape (batch_size, channels, height, width)
images = torch.randn(2, 3, 224, 224)
outputs = model(images=images)
logits = outputs.logits  # shape: (2, 197, 1000) [batch_size, num_patches+1, vocab_size]
```

### Indices and tables

- {ref}`genindex`
- {ref}`modindex`
- {ref}`search`
