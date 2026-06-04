# Transformer

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/lof310/transformer/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.10-orange)](https://pytorch.org)
[![HuggingFace Compatible](https://img.shields.io/badge/HF-compatible-ff69b4)](#)
[![Stars](https://img.shields.io/github/stars/lof310/transformer)](#)
[![Downloads](https://img.shields.io/github/downloads/lof310/transformer/total)](https://github.com/lof310/transformer/releases)

_A polished **PyTorch implementation** of the current **State-Of-The-Art(SOTA) Transformer**. Designed for clarity, reproducibility, and interoperability with **HuggingFace Transformers**, this repository provides a robust baseline for **Research** and **Engineering** being **Fully Configurable**. The codebase emphasizes **readable and well-documented components** so you can iterate on **Feed-Forward**, **Attention** and **Normalization** blocks and other **architectural variants** with minimal friction._

## Features

- **Fully Configurable** architecture (layers, heads, model dimensions, dropout, etc.)
- **HuggingFace-compatible** API alignment with `past_key_values` support for efficient generation
- **KV-Cache** support for fast incremental decoding
- **Encoder-Decoder** architecture support with cross-attention
- **Multiple Attention Variants**: MHA, GQA (Grouped Query Attention), CrossAttention
- **Flexible Position Encodings**: RoPE, PartialRoPE, ALiBi
- **LoRA Integration** for parameter-efficient fine-tuning
- **Compact and easily extensible** design for rapid prototyping and research experiments
- **Clear, well-documented modules** to facilitate experimentation with attention, FFNs, etc.

## Download the code
```bash
git clone --depth=1 https://github.com/lof310/transformer
cd transformer
```

## Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Install on developer mode (Recommended)
pip install -e .

# Install Normally
pip install .
```

## Quick Start

### Decoder-Only Model
```python
import torch
from transformer import Transformer, TransformerConfig

# Configure the model
config = TransformerConfig(
    n_layers=12,
    n_heads=32,
    d_model=1536,
    attn_qk_norm=True,
    tied_weights=False,
    seq_len=1024,
    max_seq_len=4096,
)

# Initialize model
model = Transformer(config)

# Forward Pass
B, N = 16, 1024
input_ids = torch.randint(low=0, high=config.vocab_size, size=(B, N))
output = model(input_ids, return_states=False)
```

### Incremental Decoding with KV-Cache
```python
import torch
from transformer import Transformer, TransformerConfig

config = TransformerConfig(n_layers=6, n_heads=8, d_model=512)
model = Transformer(config)
model.eval()

# Initial prompt
input_ids = torch.randint(0, config.vocab_size, (1, 10))

# First forward pass (no cache)
with torch.no_grad():
    output = model(input_ids, use_cache=True)
    logits = output.logits
    past_key_values = output.past_key_values  # Cache for next step

# Incremental decoding (one token at a time)
next_token_id = logits[:, -1:].argmax(dim=-1)
with torch.no_grad():
    output = model(next_token_id, past_key_values=past_key_values, use_cache=True)
    new_past_key_values = output.past_key_values  # Updated cache
```

### Encoder-Decoder Model
```python
import torch
from transformer import Transformer, TransformerConfig, EncoderDecoderModel

# Encoder config
encoder_config = TransformerConfig(
    n_layers=6,
    n_heads=8,
    d_model=512,
    pos_encoding="RoPE",
)

# Decoder config (with cross-attention)
decoder_config = TransformerConfig(
    n_layers=6,
    n_heads=8,
    d_model=512,
    attn_class="GQA",
    n_kv_heads=4,
    pos_encoding="RoPE",
    add_cross_attention=True,  # Enable cross-attention
)

# Create encoder-decoder model
model = EncoderDecoderModel(encoder_config, decoder_config)

# Forward pass
encoder_input = torch.randint(0, encoder_config.vocab_size, (4, 20))
decoder_input = torch.randint(0, decoder_config.vocab_size, (4, 10))

output = model(
    input_ids=decoder_input,
    encoder_input_ids=encoder_input,
    return_dict=True
)
```

### Applying LoRA Adapters
```python
from transformer import apply_lora_to_model

# After creating your model
model = Transformer(config)

# Apply LoRA to specific layers (e.g., query/key/value projections)
apply_lora_to_model(model, target_modules=["qkv_proj"], lora_rank=8, lora_alpha=16)

# Now only LoRA parameters are trainable
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True
```

## Default Configuration

The default configuration implements the latest SOTA Transformer design.

```python
from transformer import TransformerConfig

TransformerConfig(
    n_layers=12,
    d_model=1536,
    n_heads=32,
    n_kv_heads=None,  # GQA Disabled (MHA by default)
    vocab_size=50000,
    d_ff=None,  # Chosen Automatically (ratio 8/3 ≈ 2.666)
    norm_design="pre_norm",
    norm_class="rms_norm",
    ffn_class="SwiGLU",
    attn_class="MHA",  # Options: "MHA", "GQA", "CrossAttention"
    block_class=None,  # Uses default TransformerBlock
    attn_bias=False,
    ffn_bias=True,
    lm_head_bias=False,
    attn_qk_norm=True,
    attn_dropout=0.0,
    tied_weights=False,
    seq_len=1024,
    pos_encoding="RoPE",  # Options: "RoPE", "PartialRoPE", "ALiBi"
    rope_base=10000.0,
    max_seq_len=4096,
    add_cross_attention=False,  # Enable for encoder-decoder
)
```

## Architecture Overview

### Attention Mechanisms
- **MHA (Multi-Head Attention)**: Standard self-attention with equal query/key/value heads
- **GQA (Grouped Query Attention)**: Efficient attention with fewer KV heads, sharing across query groups
- **CrossAttention**: Attention between decoder queries and encoder key/values for seq2seq tasks

### Position Encodings
- **RoPE (Rotary Position Embeddings)**: Rotates query/key vectors based on absolute positions
- **PartialRoPE**: Applies RoPE to only a subset of dimensions
- **ALiBi (Attention with Linear Biases)**: Adds distance-based biases to attention scores

### Normalization Designs
- **pre_norm**: Normalize before attention/FFN (recommended for deep models)
- **post_norm**: Normalize after attention/FFN (original Transformer)
- **parallel**: Apply normalization once, then both attention and FFN in parallel
- **both**: Normalize both before and after (not compatible with CrossAttention)

## Documentation

Full documentation available at [This Page](https://lof310.github.io/transformer)

## Contributing

Contributions are welcome!

## License

Distributed under the Apache License 2.0. See `LICENSE` for more information.

## Citation

If you use `transformer` in your research, please cite:

```bibtex
@software{transformer2026,
  author = {Leinier Orama},
  title = {transformer: PyTorch implementation of the current State-Of-The-Art(SOTA) Transformer},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/lof310/transformer}
}
```
