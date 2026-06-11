# API Reference

This section provides detailed documentation of all public modules and classes.

## Configuration

```{eval-rst}
.. automodule:: transformer.config
   :members:
   :undoc-members:
   :show-inheritance:
```

## Attention Modules

### Multi-Head Attention (MHA)

```{eval-rst}
.. autoclass:: transformer.attns.MHA
   :members:
   :undoc-members:
   :show-inheritance:
```

### Grouped-Query Attention (GQA)

```{eval-rst}
.. autoclass:: transformer.attns.GQA
   :members:
   :undoc-members:
   :show-inheritance:
```

### Cross-Attention

```{eval-rst}
.. autoclass:: transformer.attns.CrossAttention
   :members:
   :undoc-members:
   :show-inheritance:
```

## Positional Embeddings

### RoPE (Rotary Position Embedding)

```{eval-rst}
.. autoclass:: transformer.pos.RoPE
   :members:
   :undoc-members:
   :show-inheritance:
```

### PartialRoPE

```{eval-rst}
.. autoclass:: transformer.pos.PartialRoPE
   :members:
   :undoc-members:
   :show-inheritance:
```

### ALiBi (Attention with Linear Biases)

```{eval-rst}
.. autoclass:: transformer.pos.ALiBi
   :members:
   :undoc-members:
   :show-inheritance:
```

## Feed-Forward Modules

### SwiGLU

```{eval-rst}
.. autoclass:: transformer.ffn.SwiGLU
   :members:
   :undoc-members:
   :show-inheritance:
```

### MLP

```{eval-rst}
.. autoclass:: transformer.ffn.MLP
   :members:
   :undoc-members:
   :show-inheritance:
```

## Transformer Model

### TransformerBlock

```{eval-rst}
.. autoclass:: transformer.transformer.TransformerBlock
   :members:
   :undoc-members:
   :show-inheritance:
```

### Transformer Class (Main Model)

```{eval-rst}
.. autoclass:: transformer.transformer.Transformer
   :members:
   :undoc-members:
   :show-inheritance:
   :special-members: __init__
```

### EncoderDecoderModel

```{eval-rst}
.. autoclass:: transformer.encoder_decoder.EncoderDecoderModel
   :members:
   :undoc-members:
   :show-inheritance:
```

## LoRA (Parameter-Efficient Fine-Tuning)

### LoRALinear

```{eval-rst}
.. autoclass:: transformer.lora.LoRALinear
   :members:
   :undoc-members:
   :show-inheritance:
```

### apply_lora_to_model

```{eval-rst}
.. autofunction:: transformer.lora.apply_lora_to_model
```

## Utilities

```{eval-rst}
.. automodule:: transformer.utils
   :members:
   :undoc-members:
   :show-inheritance:
```
