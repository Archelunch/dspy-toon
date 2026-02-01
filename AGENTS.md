# DSPy-TOON

DSPy adapter using TOON (Token-Oriented Object Notation) format for 40%+ token reduction in structured LLM outputs.

## Build & Test

### Development Setup
```bash
# Install in editable mode with all dev dependencies
pip install -e ".[dev,benchmark]"
```

### Core Commands
- **Lint check**: `ruff check src/ tests/`
- **Auto-fix lint issues**: `ruff check --fix src/ tests/`
- **Format code**: `ruff format src/ tests/`
- **Type check**: `mypy src/`
- **Run all tests**: `pytest tests/ -v`
- **Run with coverage**: `pytest tests/ --cov=dspy_toon --cov-report=term`
- **Run single test file**: `pytest tests/test_adapter.py -v`

### Pre-commit Verification
Always run before committing:
```bash
ruff check src/ tests/ && ruff format src/ tests/ && mypy src/ && pytest tests/ -v
```

## Project Layout

```
dspy-toon/
├── src/dspy_toon/          # Main library code
│   ├── __init__.py         # Public API exports
│   ├── adapter.py          # ToonAdapter implementation
│   ├── toon.py             # TOON encoder/decoder
│   └── streaming.py        # Streaming support
├── tests/                  # Unit tests
│   ├── test_adapter.py     # Adapter tests
│   └── test_toon.py        # TOON format tests
├── benchmarks/             # Performance benchmarks
├── examples/               # Usage examples
├── datasets/               # Test datasets
└── pyproject.toml          # Project config, tool settings
```

## Conventions & Patterns

### Code Style
- **Python**: 3.10+ with type hints for all function signatures
- **Line length**: 120 characters (configured in ruff)
- **Quotes**: Double quotes for strings
- **Formatting**: Ruff with space-based indentation
- **Linting**: ruff checks E, W, F, I, UP rules

### Type Checking
- Use type hints for all public function signatures
- Pydantic models for structured data
- `mypy` runs in CI but not strict mode
- Ignore rules: N (naming conventions)

### Testing
- **Framework**: pytest with pytest-asyncio for async tests
- **Naming**: `test_*.py` files, `test_*` functions
- **Coverage**: pytest-cov tracks coverage, uploaded to codecov
- **Structure**: Use class-based test groups (e.g., `TestSchemaRendering`)

### Documentation
- Docstrings for all public functions and classes
- Follow Google-style or reStructuredText format
- README examples should be runnable

### Dependencies
- Core: `dspy>=3.0.0`, `pydantic>=2.0.0`, `datasets>=4.4.1`
- Dev: `pytest`, `ruff`, `mypy`, `pytest-cov`, `pytest-asyncio`
- Benchmark: `tiktoken`, `pandas`, `tabulate`

## Git Workflow

1. **Branch from `main`**: Use descriptive names like `feature/<name>` or `fix/<name>`
2. **Commit conventions**:
   - `feat: Add support for X`
   - `fix: Handle edge case in Y`
   - `docs: Update README`
   - `test: Add tests for Z`
   - `refactor: Simplify logic`
3. **Pre-commit**: Run lint, format, type check, and tests
4. **Force pushes**: Allowed only on feature branches with `--force-with-lease`
5. **Never force-push `main`**

## PR Checklist

Before submitting a pull request:
- [ ] All tests pass (`pytest tests/ -v`)
- [ ] Lint check passes (`ruff check src/ tests/`)
- [ ] Code is formatted (`ruff format src/ tests/`)
- [ ] Type check passes (`mypy src/`)
- [ ] New functionality has tests
- [ ] Documentation updated if needed
- [ ] Diff is focused and minimal
- [ ] Clear PR description explaining intent

## When Stuck

If an agent is unsure about implementation details:
- Ask clarifying questions
- Propose a short plan before implementing
- Reference existing examples in `examples/` directory
- Check similar implementations in `src/dspy_toon/adapter.py` or `src/dspy_toon/toon.py`

## Good Examples to Copy

- **Adapter implementation**: `src/dspy_toon/adapter.py` - Shows proper type hints, error handling, logging
- **TOON encoding**: `src/dspy_toon/toon.py` - Shows recursive parsing, schema building
- **Tests**: `tests/test_adapter.py` - Shows pytest patterns, model-based testing
- **Examples**: `examples/simple_extraction.py` - Shows typical usage patterns

## CI/CD

- Tests run on Python 3.10, 3.11, 3.12, 3.13 via GitHub Actions
- Coverage uploaded to codecov for Python 3.11 builds
- PyPI publishing automated on GitHub releases
