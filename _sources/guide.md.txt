# Transformer: A PyTorch SOTA Transformer Implementation

## Introduction

`transformer` is a polished and fully configurable implementation of the latest SOTA Transformer architecture design in PyTorch. It is designed with the following goals:

- **Clarity**: Each component is self-contained, well-documented, and easy to understand.
- **Reproducibility**: The default configuration matches current state-of-the-art practices (e.g., pre‑RMSNorm, SwiGLU, RoPE, no attention dropout).
- **Extensibility**: The modular design allows swapping attention mechanisms, feed‑forward networks, and positional encodings.
- **HuggingFace Compatibility**: The main `Transformer` class inherits from `PreTrainedModel` and `GenerationMixin`, enabling seamless integration with the HuggingFace ecosystem (`.from_pretrained()`, `.save_pretrained()`, `.generate()`, etc.).

This guide will walk you through installation, configuration, module usage, training, generation, and advanced features.

---

## Installation

```bash
# Clone the repository
git clone --depth=1 https://github.com/lof310/transformer
cd transformer

# Install dependencies (PyTorch, transformers, etc.)
pip install -r requirements.txt

# Install the package in editable mode (recommended for development)
pip install -e .

# Or install normally
pip install .
```

---

## Quick Start

```python
import torch
from transformer import Transformer, TransformerConfig

# 1. Define configuration
config = TransformerConfig(
    n_layers=6,
    n_heads=8,
    d_model=384,
    vocab_size=65,
    seq_len=256,
    max_seq_len=1024,
)

# 2. Create model
model = Transformer(config)

# 3. Prepare input
input_ids = torch.randint(0, config.vocab_size, (2, 128))

# 4. Forward pass
outputs = model(input_ids)
logits = outputs.logits  # shape: [2, 128, 65]
print(logits.shape)
```

---

# Configuration

The `TransformerConfig` class provides a comprehensive set of hyperparameters to control every aspect of the model architecture. The **default configuration** is carefully chosen to reflect current State‑Of‑The‑Art (SOTA) practices in large language model design, as seen in models like LLaMA, PaLM, and modern GPT variants.

## Default Configuration (SOTA Design)

```python
from transformer import TransformerConfig

config = TransformerConfig()  # All defaults
```

| Parameter         | Default Value |      Why?      |
|-------------------|---------------|----------------|
| `n_layers`        | `12` | --- |
| `n_heads`         | `32` | Multi-head attention with 32 heads allows the model to attend to information from different representation subspaces; works well with `d_model=1536` (48 dimensions per head). |
| `d_model`         | `1536` | Embedding dimension of 1536 offers a good trade‑off between expressiveness and parameter count (≈150M parameters with 12 layers). Matches the "base" scale of many modern LMs. |
| `d_ff`            | `None` (automatically set to `ceil(d_model * 2.666) ≈ 4096`) | The feed‑forward hidden dimension is typically 2–4× the model dimension. 2.666× (8/3) is the exact ratio used in LLaMA and PaLM, yielding 4096 for `d_model=1536`. This value is chosen to keep the FFN parameters roughly 2× the attention parameters. |
| `attn_type`       | `"MHA"` | Multi‑Head Attention is the standard and most flexible. GQA can be enabled for memory efficiency when scaling to very large models. |
| `n_kv_heads`      | `n_heads` | For `attn_type="GQA"`, defaults to the same as `n_heads` (equivalent to MHA). Must be explicitly set lower to activate grouped‑query attention. |
| `attn_bias`       | `False` | SOTA models (LLaMA, GPT‑3) omit biases in attention projections to reduce parameters and improve throughput; normalization layers provide sufficient learnable shifts. |
| `attn_dropout`    | `0.0` | Modern large transformers typically use **no dropout** in attention, relying on other regularisation (weight decay, gradient clipping) and large‑scale training. Using dropout is not usually not recomended for Research Purposes|
| `ffn_bias`        | `True` | Biases in feed‑forward layers are retained because they add minimal overhead and can help with training stability; some models (e.g., LLaMA) also use biases in FFNs. |
| `attn_qk_norm`    | `True` | Applying RMSNorm to queries and keys **before** the attention computation stabilises training, especially at deeper layers and with low‑precision (FP16/BF16). Adopted by LLaMA and others. |
| `lm_head_bias`    | `False` | The language modeling head typically does not include a bias term; the logits are directly projected from the final hidden states. This is consistent with GPT‑2/3, Llama,etc. |
| `tied_weights`    | `False` | Weight tying (sharing embeddings with the output layer) reduces parameters but may limit expressiveness. SOTA decoupled models (LLaMA, GPT‑3) do not tie weights by default. |
| `seq_len`         | `1024` | Default context length for training; many models are trained on 1024 tokens before extrapolation to longer sequences. |
| `max_seq_len`     | `4096` | Maximum sequence length for which RoPE frequencies are precomputed. |
| `rope_base`       | `10000.0` | Base for the exponential frequency computation in Rotary Position Embedding. The value 10000 is standard from the original RoPE paper and works well across various lengths. |

---

## Module Overview

The package consists of several independent modules that can be used separately or assembled into a full transformer.

### 1. Attention Modules

- **`MHA`** – Standard Multi‑Head Attention with optional QK normalization and RoPE.
- **`GQA`** – Grouped‑Query Attention (Ainslie et al.) for reduced memory footprint.
- **`CrossAttention`** – Cross‑attention between two sequences (e.g., encoder‑decoder).

All attention modules share a similar interface:
```python
output = attn(x, mask=None, pos=None, flash_attn=(False,), return_states=False)
```
For `CrossAttention`, the signature is `(queries, kv, mask=None, pos_q=None, pos_k=None, ...)`.

### 2. Feed‑Forward Modules

- **`SwiGLU`** – SwiGLU activation (as used in LLaMA, PaLM, etc.).
- **`MLP`** – Classic two‑layer MLP with GELU activation.

Both have a `return_states` option to retrieve intermediate activations.

### 3. Positional Embedding

- **`RoPE`** – Rotary Position Embedding (Su et al.). Applied directly to queries and keys.

### 4. Transformer Components

- **`TransformerBlock`** – A single decoder block: attention → residual → FFN → residual, all with pre‑RMSNorm.
- **`Transformer`** – The full language model: embedding → stack of blocks → final RMSNorm → LM head.

---

## Detailed Module Usage

For complete runnable examples, see the [examples.md](examples.md) file. Here we highlight the key patterns.

### Using `MHA` or `GQA`

```python
from transformer import MHA

mha = MHA(d_model=512, n_heads=8, max_seq_len=1024)
x = torch.randn(2, 32, 512)
mask = torch.triu(torch.ones(32, 32, dtype=torch.bool), diagonal=1)  # causal
pos = torch.arange(32)

out = mha(x, mask=mask, pos=pos)
```

### Using `SwiGLU`

```python
from transformer import SwiGLU

swiglu = SwiGLU(d_model=512, d_ff=2048)
x = torch.randn(2, 32, 512)
out = swiglu(x)
states = swiglu(x, return_states=True)  # includes y1, y2
```

### Using `RoPE` independently

```python
from transformer import RoPE

rope = RoPE(max_seq_len=1024, d_head=64)
q = torch.randn(2, 8, 32, 64)
k = torch.randn(2, 8, 32, 64)
pos = torch.arange(32)
q_rot, k_rot = rope(q, k, pos, pos)
```

### Using `TransformerBlock`

```python
block = TransformerBlock(config, layer_idx=0)
out = block(x, attn_mask=mask, pos=pos)
```

---

## Training a Model

A typical training loop:

```python
import torch.optim as optim
from torch.utils.data import DataLoader
from transformer import Transformer, TransformerConfig

config = TransformerConfig(vocab_size=10000, d_model=256, n_layers=4, n_heads=4)
model = Transformer(config)
optimizer = optim.AdamW(model.parameters(), lr=3e-4)
dataloader = DataLoader(your_dataset, batch_size=16)

model.train()
for batch in dataloader:
    input_ids, labels = batch
    optimizer.zero_grad()
    outputs = model(input_ids, labels=labels)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
```

The model returns a `CausalLMOutput` object (from `transformers`), which contains `.loss`, `.logits` and optionally `hidden_states`.

---

## Text Generation

Because `Transformer` inherits from `GenerationMixin`, you can use the `.generate()` method:

```python
input_ids = tokenizer.encode("Hello", return_tensors="pt")
output_ids = model.generate(
    input_ids,
    max_new_tokens=50,
    do_sample=True,
    temperature=0.8,
    top_p=0.9,
)
generated_text = tokenizer.decode(output_ids[0])
```

All generation arguments supported by HuggingFace (e.g., `num_beams`, `repetition_penalty`) are available.

---

## Saving and Loading

Models can be saved and loaded using HuggingFace’s `save_pretrained` and `from_pretrained`:

```python
# Save
model.save_pretrained("./my_model")

# Load
model = Transformer.from_pretrained("./my_model")
```

The configuration is automatically saved in `config.json`.

---

## Using Flash Attention

Flash Attention (Dao et al.) can dramatically speed up training and reduce memory usage. Enable it by passing the `flash_attn` tuple to any attention module or the top‑level model:

```python
outputs = model(input_ids, flash_attn=(True,))
```

You can also specify which backends to use and whether to set priority:

```python
from torch.nn.attention import SDPBackend

outputs = model(
    input_ids,
    flash_attn=(True, [SDPBackend.FLASH_ATTENTION, SDPBackend.MATH], True)
)
```

When FlashAttention is enabled, the attention modules **do not return** attention weights or scores (`attn_weights` and `attn_scores` are omitted from the state dictionary).

---

## Accessing Intermediate States

Set `return_states=True` in any forward call to get the intermediate tensors:

- For `MHA`/`GQA`/`CrossAttention`: returns a dict with keys `output`, `queries`, `keys`, `values`, `output_before_proj`, `input`, and optionally `attn_weights`/`attn_scores`.
- For `SwiGLU`/`MLP`: returns dict with `output`, `input`, and intermediate activations (`y1`, `y2` or `h1`, `h2`).
- For `TransformerBlock`: returns dict with `output`, `attn_output` (dict), `ffn_output` (dict).
- For `Transformer`: returns a `CausalLMOutput` where `hidden_states` is a tuple `(input_embeddings, list_of_layer_dicts)`.

This is useful for debugging, visualization, or extracting features for downstream tasks.

---

## Custom Attention Mask

The attention mask is a **boolean tensor** where `True` indicates positions that should be **masked out** (i.e., not attended to). The shape can be:

- 2D: `(N, N)` – broadcasted over batch and heads.
- 4D: `(B, H, N, N)` or `(B, 1, N, N)` – allows per‑sample/head masks.

If `is_causal=True` in the model’s forward, a causal mask is automatically generated and combined with the provided `attn_mask` using logical OR.

Example of a padding + causal mask:

```python
padding_mask = (input_ids == pad_token_id)  # [B, N]
causal_mask = torch.triu(torch.ones(N, N, dtype=torch.bool), diagonal=1)
# Expand padding to [B, 1, 1, N] and combine
combined = causal_mask[None, None, :, :] | padding_mask[:, None, None, :]
outputs = model(input_ids, attn_mask=combined, is_causal=False)
```

---

## Extending the Library

The modular design makes it easy to add new components.

### Adding a new attention mechanism

1. Create a new class inheriting from `nn.Module` in `attns.py`.
2. Implement `forward(x, mask, pos, flash_attn, return_states)` (or similar).
3. Add it to the `__init__.py` exports.
4. Modify `TransformerBlock` to accept the new attention type.

### Adding a new feed‑forward variant

1. Create a class in `ffn.py` with a `forward(x, return_states)` method.
2. Export it.
3. Use it in `TransformerBlock` (you may need to extend the configuration).

### Adding a new positional encoding

1. Create a class in `pos.py` with a `forward(q, k, pos_q, pos_k)` method.
2. Export it.
3. Modify attention modules to use the new positional encoding.

---

## FAQ / Troubleshooting

**Why are attention weights/scores missing when I use FlashAttention?**
FlashAttention does not materialize the attention matrix, so weights and scores are not available. Set `flash_attn=(False,)` to obtain them.

**How do I use CrossAttention in a decoder block?**
The current `TransformerBlock` does not yet support CrossAttention. You can use the `CrossAttention` module directly in a custom encoder‑decoder model.

**Can I use this model with HuggingFace tokenizers?**
Yes, the model is compatible with HuggingFace’s `PreTrainedTokenizer` classes. You just need to ensure the vocabulary size matches.

## License

This project is licensed under the Apache License 2.0.

---

## Citation

If you use this library in your research, please cite:

```bibtex
@software{transformer2026,
  author = {Leinier Orama},
  title = {transformer: PyTorch implementation of the current State-Of-The-Art(SOTA) Transformer},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/lof310/transformer}
}
```
