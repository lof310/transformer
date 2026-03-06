# Installation

## Requirements

- Python 3.9+
- PyTorch 1.10+
- Transformers 4.20+ (for HuggingFace compatibility)

## Install from source

Clone the repository and install in editable mode (recommended for development):

```bash
git clone --depth=1 https://github.com/lof310/transformer
cd transformer
pip install -e .
```

## Install normally

```bash
pip install .
```

## Dependencies

The required dependencies are listed in `requirements.txt` and include:
- `torch`
- `transformers`
- `Nothing Else`

Optional development dependencies:
- `pytest` (testing)
- `sphinx`, `myst-parser`, `furo` (documentation)
