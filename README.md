# Transformer

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/lof310/transformer/actions)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![PyTorch](https://img.shields.io/badge/pytorch-%3E%3D1.10-orange)](https://pytorch.org)
[![HuggingFace Compatible](https://img.shields.io/badge/HF-compatible-ff69b4)](#)
[![Stars](https://img.shields.io/github/stars/lof310/transformer)](#)
[![Downloads](https://img.shields.io/github/downloads/lof310/transformer/total)](https://github.com/lof310/transformer/releases)

A polished PyTorch implementation of the current State-Of-The-Art(SOTA) Transformer. Designed for clarity, reproducibility, and interoperability with HuggingFace Transformers, this repository provides a robust baseline for research and engineering being fully configurable. The codebase emphasizes readable, well-documented components so you can iterate on attention mechanisms, Feed-Forward, Attention and Normalization blocks and other architectural variants with minimal friction.

## Features
- **Fully Configurable** architecture (layers, heads, model dimensions, dropout, etc.)
- HuggingFace-compatible API alignment.
- Compact and easily extensible design for rapid prototyping and research experiments.
- Clear, well-documented modules to facilitate experimentation with attention, FFNs, etc.

## Download the code
```bash
git clone --depth=1 https://github.com/lof310/transformer
cd transformer
```

## Installation
```python
# Install dependencies
pip install -r requirements.txt

# Install on developer mode (Recommended)
pip install -e .

# Install Normally
pip install .
```

## Quick Start
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer import Transformer, TransformerConfig

# Configure the model
config = TransformerConfig(
    n_layers = 12,
    n_heads: int = 32,
    d_model: int = 1536,
    qk_norm: bool = False,     
    tied_weights: bool = False,
    seq_len: int = 1024,
    max_seq_len: int = 4096,
)

# Initialize model
model = Transformer(config)

# Forward Pass
B, N = 16, 1024
input_ids = torch.randint(low=0, high=config.vocab_size, size(B, N))
output = model(input_ids, return_states=False)
```

## Documentation

Full Documentation available at [![Documentation](https://lof310.github.io/transformer)](https://lof310.github.io/transformer)

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
