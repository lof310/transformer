# Contributing

We welcome contributions! Please follow these guidelines to ensure a smooth collaboration.

## Development Setup

1. Fork the repository.
2. Clone your fork: `git clone https://github.com/your-username/transformer.git`
3. Install in development mode with development dependencies:
```bash
pip install -r requirements.txt
pip install -r docs/requirements-docs.txt # Optional
pip install -e .
```
4. Create a branch for your feature: `git checkout -b feature/amazing-feature`

## Code Style

We use [Black](https://black.readthedocs.io/) for code formatting and [isort](https://pycqa.github.io/isort/) for import sorting. Please run these before committing:

```bash
black transformer/ tests/
isort transformer/ tests/
```

## Testing

Run tests with `pytest`:

```bash
pytest tests/
```

## Documentation

If you change any public API, please update the docstrings accordingly. To build the documentation locally:

```bash
cd docs
pip install -r requirements-docs.txt
make clean
make html
```

Then check in your browser with:
```bash
cd build/html
python -m http.server
```

## Pull Request Process

1. Update the `README.md` and documentation if needed.
2. Ensure all tests pass.
3. Submit a pull request with a clear title and description.

## Code of Conduct

Please note that this project adheres to a **Contributor Covenant Code of Conduct**. By participating, you are expected to uphold this code.
