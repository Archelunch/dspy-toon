# Contributing to DSPy-TOON

Thank you for your interest in contributing to DSPy-TOON! This document provides guidelines and instructions for contributing.

## Development Setup

1. **Clone the repository**

   ```bash
   git clone https://github.com/Archelunch/dspy-toon.git
   cd dspy-toon
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install development dependencies**

   ```bash
   pip install -e ".[dev,benchmark]"
   ```

## Code Quality

### Linting

We use `ruff` for linting and formatting:

```bash
# Check for issues
ruff check src/ tests/

# Auto-fix issues
ruff check --fix src/ tests/

# Format code
ruff format src/ tests/
```

### Type Checking

We use `mypy` for type checking:

```bash
mypy src/
```

### Testing

We use `pytest` for testing:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=dspy_toon --cov-report=term

# Run specific test file
pytest tests/test_adapter.py -v
```

## Pull Request Process

1. **Fork the repository** and create your branch from `main`

2. **Make your changes** following the code style guidelines

3. **Add tests** for any new functionality

4. **Update documentation** if needed

5. **Run the test suite** to ensure nothing is broken

6. **Submit a pull request** with a clear description of your changes

## Commit Messages

Please use clear, descriptive commit messages:

- `feat: Add support for custom delimiters`
- `fix: Handle edge case in tabular array parsing`
- `docs: Update README with new examples`
- `test: Add tests for nested model encoding`
- `refactor: Simplify TOON encoder logic`

## Code Style

- Follow PEP 8 guidelines
- Use type hints for function signatures
- Write docstrings for public functions and classes
- Keep functions focused and reasonably sized

## Reporting Issues

When reporting issues, please include:

1. **Python version** and **DSPy version**
2. **Minimal reproducible example**
3. **Expected behavior** vs **actual behavior**
4. **Full error traceback** if applicable

## Feature Requests

Feature requests are welcome! Please:

1. Check existing issues to avoid duplicates
2. Clearly describe the use case
3. Explain why this would be valuable

## Questions?

Feel free to open an issue for questions or discussions about the project.

---

Thank you for contributing! 🙏
