# DSPy-TOON

[![Tests](https://github.com/Archelunch/dspy-toon/actions/workflows/test.yml/badge.svg)](https://github.com/Archelunch/dspy-toon/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-orange?logo=buy-me-a-coffee)](https://buymeacoffee.com/mike_pavlukhin)

[DSPy](https://dspy.ai/) adapter for structured LLM inputs and outputs in [TOON](https://github.com/toon-format/spec).

- **Typed outputs:** Pydantic models, nested objects, lists, dictionaries, unions and `Literal` values.
- **Schema guidance:** field descriptions, enum choices, bounds and nullable fields in model instructions.
- **Compact serialization:** shared headers for uniform records, nested field groups and keyed tables.
- **DSPy integration:** synchronous and asynchronous predictions, demonstrations and conversation history.
- **Standalone codec:** `encode()` and `decode()` for Python data.
- **Streaming:** optional incremental output for scalar fields.

[Examples](https://github.com/Archelunch/dspy-toon/tree/main/examples) · [Changelog](https://github.com/Archelunch/dspy-toon/blob/main/CHANGELOG.md) · [Migration guide](https://github.com/Archelunch/dspy-toon/blob/main/UPGRADE.md)

## Contents

- [Installation](#installation)
- [Quick start](#quick-start)
- [Adapter behavior](#adapter-behavior)
- [Typed outputs](#typed-outputs)
- [TOON format](#toon-format)
- [Encoder and decoder API](#encoder-and-decoder-api)
- [Async calls](#async-calls)
- [Streaming](#streaming)
- [Benchmarks](#benchmarks)
- [Migration](#migration)
- [Development and contributions](#development-and-contributions)

## Installation

Python 3.10–3.14, DSPy `>=3.3.1,<4`, and Pydantic 2.

```bash
pip install dspy-toon
```

To install from source:

```bash
pip install "git+https://github.com/Archelunch/dspy-toon.git"
```

## Quick start

Set `OPENAI_API_KEY` for this example:

```python
import dspy
from pydantic import BaseModel, Field
from dspy_toon import ToonAdapter


class UserInfo(BaseModel):
    name: str = Field(description="Full name")
    age: int = Field(ge=0, description="Age in years")
    occupation: str = Field(description="Job title")


class ExtractUser(dspy.Signature):
    """Extract user information from text."""

    text: str = dspy.InputField()
    user: UserInfo = dspy.OutputField()


lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, adapter=ToonAdapter())

extractor = dspy.Predict(ExtractUser)
result = extractor(text="Alice Johnson is a 35-year-old software engineer.")

print(result.user.name)
print(result.user.model_dump())
```

`result.user` is a validated `UserInfo` instance.

Use `dspy.context()` to scope the adapter to a block:

```python
with dspy.context(adapter=ToonAdapter()):
    result = extractor(text="Bob is 28 and works as a designer.")
```

## Adapter behavior

`ToonAdapter` formats input fields and demonstrations as TOON, generates output examples from the signature, and validates returned fields with Pydantic. Instructions include nested constraints and syntax rules for the declared input and output types.

The adapter uses DSPy's call, async and demonstration handling. Conversation history supports `dspy.History` and legacy lists of user/assistant pairs.

To encode a payload without changing the output adapter, use [`encode()`](#encoder-and-decoder-api).

### Output validation and errors

| Response | Behavior |
|---|---|
| Complete TOON document | Decoded and validated against the signature |
| Complete JSON object | Accepted as a fallback, with the same validation |
| Document inside a code fence | Outer fence removed before parsing |
| Missing required or unexpected output fields | Raises `AdapterParseError` |
| Missing defaulted or nullable output fields | Uses DSPy's output-default handling |
| Malformed TOON, mismatched counts or duplicate keys | Rejected during decoding |
| Fragments embedded in prose | Not recovered |

```python
from dspy.utils.exceptions import AdapterParseError

try:
    result = extractor(text="Alice is 35 and works as an engineer.")
except AdapterParseError as error:
    print(error)
```

Validation uses Pydantic's coercion rules unless the annotation specifies strict types. It checks types and constraints; it does not verify factual correctness.

### Inspect the prompt

`format()` returns the messages without calling the model:

```python
adapter = ToonAdapter()
messages = adapter.format(
    signature=ExtractUser,
    demos=[],
    inputs={"text": "Alice is 35 and works as an engineer."},
)
for message in messages:
    print(message["role"], message["content"], sep="\n")
```

## Typed outputs

The examples below use the model configuration from the quick start.

### Lists of records

Declare lists with `list[Model]`:

```python
class ExtractPeople(dspy.Signature):
    """Extract every person mentioned in the text."""

    text: str = dspy.InputField()
    people: list[UserInfo] = dspy.OutputField()


extract_people = dspy.Predict(ExtractPeople)
result = extract_people(
    text="Alice is a 35-year-old engineer. Bob is a 28-year-old designer."
)
for person in result.people:
    print(person.name, person.age, person.occupation)
```

Pydantic models are also supported in input fields, including nested lists and dictionaries.

### Nested models and nullable fields

```python
from typing import Literal


class Address(BaseModel):
    street: str
    city: str
    country: Literal["US", "UK", "DE"]


class UserProfile(BaseModel):
    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    address: Address | None = Field(
        description="Home address, or null if the text does not provide it"
    )


class ExtractProfile(dspy.Signature):
    """Extract a profile. Use null for an unknown address."""

    text: str = dspy.InputField()
    profile: UserProfile = dspy.OutputField()


extract_profile = dspy.Predict(ExtractProfile)
result = extract_profile(text="Contact John at john@example.com.")
print(result.profile)
```

`Address | None` permits an explicit `null`. Within a Pydantic model, a nullable field without a default is still required; add `= None` when omission should also be accepted.

### Classification with constraints

```python
class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0, le=1, description="Confidence from 0 to 1")
    key_phrases: list[str] = Field(description="Phrases supporting the label")


class AnalyzeSentiment(dspy.Signature):
    """Classify the sentiment of the text."""

    text: str = dspy.InputField()
    result: SentimentResult = dspy.OutputField()


analyzer = dspy.Predict(AnalyzeSentiment)
result = analyzer(text="I love this product. It works exactly as promised.")
print(result.result.sentiment, result.result.confidence)
```

`Literal` restricts the accepted labels. `ge` and `le` validate numeric bounds.

## TOON format

The codec targets TOON 4.1. Uniform records share field names in a header.

### Flat records

JSON:

```json
{"people":[{"name":"Alice","age":35},{"name":"Bob","age":28}]}
```

TOON:

```text
people[2]{name,age}:
  Alice,35
  Bob,28
```

### Nested field groups

```text
people[2]{name,address{city,country}}:
  Alice,Paris,FR
  Bob,Berlin,DE
```

Each row reconstructs an object with `name` and a nested `address` object.

### Keyed tables

Keyed tables encode dictionaries of uniform records:

```text
people[2:]{age,city}:
  alice: 35,Paris
  bob: 28,Berlin
```

This decodes to `{"people": {"alice": {"age": 35, "city": "Paris"}, "bob": {"age": 28, "city": "Berlin"}}}`.

### Mixed structures and empty values

Arrays that do not fit a shared table use list items. With the default indentation, rows under a field on a list-item line are four spaces deeper than the hyphen:

```text
items[2]:
  - users[2]{id,name}:
      1,Alice
      2,Bob
    status: active
  - users[1]{id,name}:
      3,Carol
    status: pending
```

Empty arrays use `items: []`. Nullable values use `address: null`. Strings that resemble numbers, booleans or null are quoted to preserve their type. The encoder also quotes delimiter-sensitive strings and escapes control characters.

Specification and compatibility details: [UPGRADE.md](https://github.com/Archelunch/dspy-toon/blob/main/UPGRADE.md).

## Encoder and decoder API

```python
from dspy_toon import ToonDecodeError, decode, encode

records = {"people": [{"name": "Alice", "age": 35}, {"name": "Bob", "age": 28}]}
text = encode(records)
assert decode(text) == records
```

| Export | Purpose |
|---|---|
| `ToonAdapter()` | DSPy prompt formatting and validated output parsing |
| `encode(value, options=None)` | Convert a Python value to TOON text |
| `decode(text, options=None)` | Decode TOON into Python dictionaries, lists and scalar values |
| `ToonDecodeError` | `ValueError` subclass for malformed TOON |
| `enable_toon_streaming()` | Install the optional scalar streaming integration |
| `is_streaming_enabled()` | Report whether that integration is installed |

### Codec options

Options are passed as a dictionary:

```python
text = encode(records, {"indentSize": 4, "delimiter": "|"})
restored = decode(text, {"indentSize": 4, "strict": True})
assert restored == records
```

| Option | Applies to | Default | Behavior |
|---|---|---|---|
| `indentSize` | Encode and decode | `2` | Positive integer specifying the indentation step |
| `indent` | Encode and decode | `2` | Compatibility alias; `indentSize` takes precedence |
| `delimiter` | Encode | `","` | Comma, tab (`"\t"`) or pipe (`"|"`); the decoder reads it from the header |
| `strict` | Decode | `True` | Validate structure, counts, widths and duplicate keys |

The adapter uses the codec defaults. These options configure standalone `encode()` and `decode()` calls; `ToonAdapter` does not expose a codec-options argument.

Non-strict decoding relaxes indentation, tolerates count and width mismatches, and keeps the last duplicate key. Missing cells may be omitted and extra cells ignored. Use strict decoding for model output. The obsolete `lengthMarker` option raises an error.

### Python value conversion

Pydantic models use `model_dump()` recursively. Dictionaries become objects, tuples and sets become arrays, and dates use ISO 8601 strings. Dictionary keys become strings. `decode()` returns plain Python values; the adapter reconstructs typed outputs separately.

Integers retain Python precision. Decimal inputs use float approximation. Non-finite floats and unsupported Python objects encode as `null`; convert custom objects before encoding.

```python
try:
    decode("people[2]{name,age}:\n  Alice,35")
except ToonDecodeError as error:
    print(error)  # The document declares two rows but contains only one.
```

## Async calls

Use `acall()` for an asynchronous predictor call:

```python
import asyncio


async def main():
    result = await extractor.acall(
        text="Alice Johnson is a 35-year-old software engineer."
    )
    print(result.user)


asyncio.run(main())
```

Inside an existing event loop, call `await extractor.acall(...)` directly. DSPy's `asyncify()` remains available for synchronous programs:

```python
async_extractor = dspy.asyncify(extractor)
# Inside an async function:
# result = await async_extractor(text="Alice is a 35-year-old engineer.")
```

## Streaming

Enable the integration before creating stream listeners. Choose a scalar output field, such as a text answer, for incremental display:

```python
import asyncio
import dspy
from dspy_toon import ToonAdapter, enable_toon_streaming, is_streaming_enabled


enable_toon_streaming()
assert is_streaming_enabled()

dspy.configure(
    lm=dspy.LM("openai/gpt-4o-mini", cache=False),
    adapter=ToonAdapter(),
)
predict = dspy.Predict("question -> answer")
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="answer")
    ],
)


async def stream_response():
    async for chunk in stream_predict(question="Explain how a hash table works."):
        if isinstance(chunk, dspy.streaming.StreamResponse):
            print(chunk.chunk, end="", flush=True)
        elif isinstance(chunk, dspy.Prediction):
            print("\nFinal answer:", chunk.answer)


asyncio.run(stream_response())
```

For synchronous iteration, create the stream with `async_streaming=False`:

```python
stream_predict = dspy.streamify(
    predict,
    stream_listeners=[
        dspy.streaming.StreamListener(signature_field_name="answer")
    ],
    async_streaming=False,
)
for chunk in stream_predict(question="What is a hash collision?"):
    if isinstance(chunk, dspy.streaming.StreamResponse):
        print(chunk.chunk, end="", flush=True)
    elif isinstance(chunk, dspy.Prediction):
        print("\nFinal answer:", chunk.answer)
```

Incremental chunks contain raw TOON text, including string quotes and escape sequences. Use the final `Prediction` for decoded, validated values. Arrays, tables and keyed objects arrive in that final prediction rather than as incrementally decoded records.

`enable_toon_streaming()` is idempotent and patches DSPy's internal `StreamListener` methods for the process. Other adapters use the original methods. The integration is tested against DSPy 3.3.1 and depends on its internal listener API.

See DSPy's [streaming documentation](https://dspy.ai/tutorials/streaming/) for listener and program configuration.

## Benchmarks

Token savings depend on data shape and the complete prompt. Uniform records can be smaller in TOON; compact JSON can be smaller for irregular structures.

| Experiment | Scope | Report |
|---|---|---|
| Initial adapter comparison | TOON, JSON and chat adapters with reasoning on and off | [Findings](https://github.com/Archelunch/dspy-toon/blob/main/benchmark_results/qwen_20260905/FINDINGS.md) |
| Expanded comparison | 4,376 scored calls across 544 fresh cases, including input/output format ablations | [Expanded report](https://github.com/Archelunch/dspy-toon/blob/main/benchmark_results/qwen_blog_expanded/BLOG_DATA.md) |
| Input-format study | Upstream reading tasks and synthetic tool-result workloads with controlled outputs | [Interpretation](https://github.com/Archelunch/dspy-toon/blob/main/benchmark_results/qwen_input_study/INTERPRETATION.md) |

Reports cover token usage, parsing and task accuracy on one Qwen3.8-27B-FP8 deployment. See [data availability](https://github.com/Archelunch/dspy-toon/blob/main/benchmark_results/DATA_AVAILABILITY.md) for archive access details.

## Migration

See [UPGRADE.md](https://github.com/Archelunch/dspy-toon/blob/main/UPGRADE.md) for changes to array syntax, keyed tables, comments and output validation. Upgrade readers before writers when exchanging TOON with another system. Regenerate stored prompts and demonstrations after a format upgrade.

Release history: [CHANGELOG.md](https://github.com/Archelunch/dspy-toon/blob/main/CHANGELOG.md).

## Development and contributions

```bash
git clone https://github.com/Archelunch/dspy-toon.git
cd dspy-toon
pip install -e ".[dev]"

pytest tests/ -v
mypy src/
ruff check src/ tests/
ruff format src/ tests/
```

For coverage, run `pytest tests/ --cov=dspy_toon --cov-report=term`.

To run experiments from a checkout, add their optional dependencies:

```bash
pip install -e ".[benchmark]"
python -m benchmarks.adapter_comparison --model gemini/gemini-2.5-flash-lite
```

Run benchmark scripts from a repository checkout; they are not included in the pip package. Each experiment's protocol lists its dataset and model requirements.

See [CONTRIBUTING.md](https://github.com/Archelunch/dspy-toon/blob/main/CONTRIBUTING.md) for contribution guidelines. Include a reproducible input and the expected output when reporting a codec or adapter issue.

## License

MIT. See [LICENSE](https://github.com/Archelunch/dspy-toon/blob/main/LICENSE) and [NOTICE](https://github.com/Archelunch/dspy-toon/blob/main/NOTICE) for copyright and attribution notices.
