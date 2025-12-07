# DSPy-TOON

[![Tests](https://github.com/Archelunch/dspy-toon/actions/workflows/test.yml/badge.svg)](https://github.com/Archelunch/dspy-toon/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-orange?logo=buy-me-a-coffee)](https://buymeacoffee.com/mike_pavlukhin)


> **DSPy adapter using TOON (Token-Oriented Object Notation) for 40%+ token reduction in structured LLM outputs.**

DSPy-TOON provides a custom adapter for [DSPy](https://github.com/stanfordnlp/dspy) that uses TOON format instead of JSON for structured outputs. TOON is a compact, human-readable serialization format optimized for LLM contexts, achieving **65% fewer output tokens** for tabular data.

## Key Features

- **40%+ Total Token Reduction** - Significant savings on both input and output tokens
- **65% Output Token Reduction** - Tabular format dramatically reduces response tokens for lists
- **Seamless DSPy Integration** - Drop-in replacement for JSONAdapter
- **Async & Streaming Support** - Full support for `dspy.asyncify()` and `dspy.streamify()`

## Installation

```bash
# Install from GitHub (recommended during beta)
pip install git+https://github.com/Archelunch/dspy-toon.git

# With benchmark dependencies
pip install "dspy-toon[benchmark] @ git+https://github.com/Archelunch/dspy-toon.git"

# For development
git clone https://github.com/Archelunch/dspy-toon.git
cd dspy-toon
pip install -e ".[dev]"
```

## Quick Start

```python
import dspy
from pydantic import BaseModel, Field
from dspy_toon import ToonAdapter

# Define your Pydantic models
class UserInfo(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job title")

# Define DSPy signature
class ExtractUser(dspy.Signature):
    """Extract user information from text."""
    text: str = dspy.InputField()
    user: UserInfo = dspy.OutputField()

# Configure DSPy with ToonAdapter
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, adapter=ToonAdapter())

# Use as normal
extractor = dspy.Predict(ExtractUser)
result = extractor(text="Alice Johnson is a 35-year-old software engineer.")
print(result.user)
# UserInfo(name='Alice Johnson', age=35, occupation='software engineer')
```

## TOON Format

TOON uses compact syntax that LLMs can easily produce and parse:

**JSON:**
```json
[{"id":1,"name":"Person 1","age":21},{"id":2,"name":"Person 2","age":22},{"id":3,"name":"Person 3","age":23}]
```

**TOON:**
```
[3]{id,name,age}:
  1,Person 1,21
  2,Person 2,22
  3,Person 3,23
```

## Benchmarks

Real token usage measured with DSPy's `track_usage=True` on `gemini/gemini-2.5-flash-lite`:

### Token Usage by Test Case

| Test Case | ToonAdapter | BAMLAdapter | JSONAdapter | ChatAdapter |
|-----------|-------------|-------------|-------------|-------------|
| Single person | **214** 🏆 | 219 | 326 | 310 |
| List of 3 people | **272** 🏆 | 308 | 453 | 405 |
| List of 10 people | **389** 🏆 | 599 | 729 | 597 |
| List of 5 products | **335** 🏆 | 420 | 573 | 507 |
| Nested address | 334 | **313** 🏆 | 512 | 472 |

### Total Token Usage (All Tests)

| Adapter | Input | Output | Total | vs JSON |
|---------|-------|--------|-------|---------|
| **ToonAdapter** | 1282 | **262** | **1544** 🏆 | **-40.5%** |
| BAMLAdapter | 1181 | 678 | 1859 | -28.3% |
| ChatAdapter | 1779 | 512 | 2291 | -11.7% |
| JSONAdapter | 1855 | 738 | 2593 | - |

**ToonAdapter wins** with **40.5% total savings** and **65% output token reduction** vs JSONAdapter!

Run benchmarks:
```bash
python -m benchmarks.adapter_comparison --model gemini/gemini-2.5-flash-lite
```

## Examples

### Sentiment Analysis

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal
from dspy_toon import ToonAdapter

class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(description="Confidence score 0-1")
    key_phrases: list[str] = Field(description="Key phrases that influenced sentiment")

class AnalyzeSentiment(dspy.Signature):
    """Analyze sentiment of the given text."""
    text: str = dspy.InputField()
    result: SentimentResult = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, adapter=ToonAdapter())

analyzer = dspy.Predict(AnalyzeSentiment)
result = analyzer(text="I absolutely love this product! Best purchase ever.")
print(result.result)
```

### Extract Multiple Entities (Tabular)

```python
import dspy
from pydantic import BaseModel
from dspy_toon import ToonAdapter

class Person(BaseModel):
    name: str
    age: int
    occupation: str

class ExtractPeople(dspy.Signature):
    """Extract all people mentioned in the text."""
    text: str = dspy.InputField()
    people: list[Person] = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, adapter=ToonAdapter())

extractor = dspy.Predict(ExtractPeople)
result = extractor(text="""
    Alice (35) is a software engineer. Bob is 28 and works as a designer.
    Carol, aged 42, is the project manager.
""")

# ToonAdapter uses tabular format for lists - saves 30%+ tokens
for person in result.people:
    print(f"{person.name}, {person.age}, {person.occupation}")
```

### Nested Models

```python
import dspy
from pydantic import BaseModel, Field
from typing import Literal
from dspy_toon import ToonAdapter

class Address(BaseModel):
    street: str
    city: str
    country: Literal["US", "UK", "DE"]

class UserProfile(BaseModel):
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    address: Address | None = Field(description="Home address")

class ExtractProfile(dspy.Signature):
    """Extract user profile from text."""
    text: str = dspy.InputField()
    profile: UserProfile = dspy.OutputField()

lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, adapter=ToonAdapter())

extractor = dspy.Predict(ExtractProfile)
result = extractor(text="Contact John at john@example.com. He lives at 123 Main St, Boston, US.")
print(result.profile)
```

See the `examples/` directory for complete working examples.

## Async & Streaming

ToonAdapter fully supports DSPy's async operations and token-level streaming.

### Async Operations

Use `dspy.asyncify()` for async operations:

```python
import asyncio
import dspy
from dspy_toon import ToonAdapter

dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=ToonAdapter())

predict = dspy.Predict("question -> answer")
async_predict = dspy.asyncify(predict)

async def main():
    result = await async_predict(question="What is the capital of France?")
    print(result.answer)

asyncio.run(main())
```

### Token-Level Streaming

For real-time token streaming, enable ToonAdapter streaming support:

```python
import asyncio
import dspy
from dspy_toon import ToonAdapter, enable_toon_streaming

# Enable streaming support (call once at startup)
enable_toon_streaming()

# Configure DSPy
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=ToonAdapter())

predict = dspy.Predict("question -> answer")

# Create streaming predictor
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
)

async def stream_response():
    async for chunk in stream_predict(question="Explain quantum computing briefly."):
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk.chunk, end="", flush=True)  # Print tokens as they arrive
        elif isinstance(chunk, dspy.Prediction):
            print(f"\n\nFinal: {chunk.answer}")

asyncio.run(stream_response())
```

### Synchronous Streaming

For sync streaming, set `async_streaming=False`:

```python
import dspy
from dspy_toon import ToonAdapter, enable_toon_streaming

enable_toon_streaming()
dspy.configure(lm=dspy.LM("openai/gpt-4o-mini", cache=False), adapter=ToonAdapter())

predict = dspy.Predict("question -> answer")
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    async_streaming=False,  # Sync mode
)

for chunk in stream_predict(question="What is 2+2?"):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk.chunk, end="", flush=True)
```

## Development

```bash
# Clone repository
git clone https://github.com/Archelunch/dspy-toon.git
cd dspy-toon

# Install with dev dependencies
pip install -e ".[dev,benchmark]"

# Run tests
pytest tests/ -v

# Run tests with coverage
pytest tests/ --cov=dspy_toon --cov-report=term

# Type checking
mypy src/

# Linting
ruff check src/ tests/
ruff format src/ tests/
```

## Roadmap

- [x] Core ToonAdapter implementation
- [x] Token usage benchmarks
- [x] BAML adapter comparison benchmarks
- [x] Async support via `dspy.asyncify()`
- [x] Token-level streaming via `enable_toon_streaming()`
- [ ] Integration with DSPy optimizers (MIPROv2, BootstrapFewShot)
- [ ] More benchmarks on complex data and optimizations

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License

## Acknowledgments

- [DSPy](https://github.com/stanfordnlp/dspy) - The foundation framework
- [TOON Format](https://github.com/toon-format/spec) - Original TOON specification
- [toon-python](https://github.com/toon-format/toon-python) - Python TOON encoder/decoder
- [BAML](https://github.com/BoundaryML/baml) - Inspiration for adapter approach

