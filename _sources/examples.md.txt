# Usage Examples
**This section provides complete and diverse usage examples of all modules and classes**

## Basic Usage

### Text Processing
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
outputs = model(input_ids=input_ids)
logits = outputs.logits # shape: [B, N, V]
print(logits.shape)
```

### Image Processing with Vision Transformer (ViT)

The library supports Vision Transformer (ViT) architecture for image processing. Configure the model with `patch_size` and `img_size` parameters to enable ViT mode.

```python
import torch
from transformer import Transformer, TransformerConfig

# Configure ViT for image classification
config = TransformerConfig(
    n_layers=12,
    n_heads=16,
    d_model=1024,
    vocab_size=1000,      # Number of output classes
    patch_size=16,        # Patch size (16x16 pixels per token)
    img_size=224,         # Input image size (224x224)
    in_channels=3,        # RGB images
    max_seq_len=512,      # Must be >= num_patches + 1 (for CLS token)
                          # For 224x224 with patch_size=16: (224/16)^2 + 1 = 197
    pos_encoding="RoPE"   # Positional encoding type
)

model = Transformer(config)
model.eval()

# Process a batch of images
batch_size = 4
images = torch.randn(batch_size, 3, 224, 224)  # (B, C, H, W)

with torch.no_grad():
    outputs = model(images=images)
    logits = outputs.logits  # Shape: (B, num_patches+1, vocab_size)
    
    # For classification, use the CLS token (first position)
    cls_logits = logits[:, 0, :]  # Shape: (B, vocab_size)
    predictions = cls_logits.argmax(dim=-1)
    print(f"Predictions: {predictions}")
```

#### ViT for Feature Extraction

Extract intermediate features from different layers of the ViT:

```python
import torch
from transformer import Transformer, TransformerConfig

config = TransformerConfig(
    n_layers=12,
    n_heads=16,
    d_model=1024,
    vocab_size=1000,
    patch_size=16,
    img_size=224,
    max_seq_len=512
)

model = Transformer(config)
model.eval()

images = torch.randn(2, 3, 224, 224)

# Get hidden states from all layers
with torch.no_grad():
    outputs = model(images=images, return_states=True)
    hidden_states = outputs.hidden_states
    
    # Extract features from the last layer's CLS token
    final_cls_features = hidden_states[-1]["output"][:, 0, :]  # Shape: (B, d_model)
    print(f"CLS features shape: {final_cls_features.shape}")
```

#### Multi-Resolution Support

ViT can handle different image resolutions by adjusting `max_seq_len`:

```python
from transformer import Transformer, TransformerConfig

# Small images (e.g., CIFAR-10: 32x32)
config_small = TransformerConfig(
    n_layers=6,
    n_heads=8,
    d_model=512,
    vocab_size=10,
    patch_size=4,
    img_size=32,
    max_seq_len=128  # (32/4)^2 + 1 = 65 tokens
)

# Large images (e.g., ImageNet: 384x384)
config_large = TransformerConfig(
    n_layers=12,
    n_heads=16,
    d_model=1024,
    vocab_size=1000,
    patch_size=16,
    img_size=384,
    max_seq_len=1024  # (384/16)^2 + 1 = 577 tokens
)
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

## Encoder-Decoder Model (Seq2Seq)

The `EncoderDecoderModel` class provides a complete encoder-decoder architecture with cross-attention support.

```python
import torch
from transformer import EncoderDecoderModel, TransformerConfig

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
logits = output.logits
print(f"Logits shape: {logits.shape}")  # [batch_size, decoder_seq_len, vocab_size]
```

### Incremental Decoding with KV-Cache

```python
import torch
from transformer import EncoderDecoderModel, TransformerConfig

encoder_config = TransformerConfig(n_layers=6, n_heads=8, d_model=512)
decoder_config = TransformerConfig(n_layers=6, n_heads=8, d_model=512)
model = EncoderDecoderModel(encoder_config, decoder_config)
model.eval()

# Encode source sequence
encoder_input = torch.randint(0, encoder_config.vocab_size, (1, 20))
encoder_outputs = model.encode(encoder_input, return_dict=True)
encoder_hidden_states = encoder_outputs["last_hidden_state"]

# Initial decoder prompt
decoder_input = torch.randint(0, decoder_config.vocab_size, (1, 5))

# First forward pass (no cache)
with torch.no_grad():
    output = model.decode(
        decoder_input,
        encoder_hidden_states=encoder_hidden_states,
        use_cache=True
    )
    logits = output.logits
    past_key_values = output.past_key_values

# Incremental decoding (one token at a time)
next_token_id = logits[:, -1:].argmax(dim=-1)
with torch.no_grad():
    output = model.decode(
        next_token_id,
        encoder_hidden_states=encoder_hidden_states,
        past_key_values=past_key_values,
        use_cache=True
    )
    new_past_key_values = output.past_key_values
```

## Applying LoRA Adapters

LoRA (Low-Rank Adaptation) enables parameter-efficient fine-tuning by adding small trainable adapters to frozen model weights.

```python
from transformer import Transformer, TransformerConfig, apply_lora_to_model

# Create model
config = TransformerConfig(n_layers=6, n_heads=8, d_model=512)
model = Transformer(config)

# Apply LoRA to query/key/value projections
apply_lora_to_model(model, target_modules=["qkv_proj"], lora_rank=8, lora_alpha=16)

# Freeze base model parameters
for param in model.parameters():
    param.requires_grad = False

# Only train LoRA parameters
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad = True

# Now you can train with only LoRA parameters being updated
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)
```

### LoRA on Multiple Module Types

```python
# Apply LoRA to both attention projections and feed-forward layers
apply_lora_to_model(
    model,
    target_modules=["qkv_proj", "W1", "W2"],  # attention and FFN layers
    lora_rank=16,
    lora_alpha=32,
    lora_dropout=0.1
)
```
