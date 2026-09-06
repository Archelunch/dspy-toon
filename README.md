# DSPy-TOON

[![Tests](https://github.com/Archelunch/dspy-toon/actions/workflows/test.yml/badge.svg)](https://github.com/Archelunch/dspy-toon/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Buy Me a Coffee](https://img.shields.io/badge/Buy%20Me%20a%20Coffee-orange?logo=buy-me-a-coffee)](https://buymeacoffee.com/mike_pavlukhin)

A [DSPy](https://dspy.ai/) adapter that reads and writes [TOON](https://github.com/toon-format/spec), with a Python codec you can also use on its own.

TOON stores uniform records under a shared header instead of repeating field names in every row. It can make structured prompts smaller, especially for lists of records, uniformly nested objects and dictionaries of records. Irregular data may be smaller as compact JSON.

The 0.4.0 release targets TOON 4.1 and requires Python 3.10 through 3.14 and DSPy 3.3.1 or later, below version 4. DSPy-TOON maintains its codec in this package.

## Install

```bash
pip install dspy-toon
```

Once 0.4.0 is published, pin that release with:

```bash
pip install "dspy-toon==0.4.0"
```

The package contains the `dspy_toon` module and its license notices. Experiments, benchmark data, tests and blog assets stay in the repository. Its direct runtime dependencies are DSPy and Pydantic; pip also installs their dependencies.

## Use it with DSPy

Configure your language model as usual, then pass `ToonAdapter` to DSPy:

```python
import dspy
from pydantic import BaseModel
from dspy_toon import ToonAdapter

class Person(BaseModel):
    name: str
    age: int

class ExtractPeople(dspy.Signature):
    """Extract the people mentioned in the text."""

    text: str = dspy.InputField()
    people: list[Person] = dspy.OutputField()

# This example uses OPENAI_API_KEY from the environment.
lm = dspy.LM("openai/gpt-4o-mini")
dspy.configure(lm=lm, adapter=ToonAdapter())

extract = dspy.Predict(ExtractPeople)
result = extract(text="Alice is 35. Bob is 28.")

for person in result.people:
    print(person.name, person.age)
```

The adapter formats the input fields as TOON, describes the expected output, and validates returned values against the signature. It supports nested Pydantic models, lists, unions, nullable fields and field constraints. Invalid outputs raise a DSPy `AdapterParseError` rather than passing unchecked values to your application.

`ToonAdapter` changes both input and output formatting. If you only want to encode data before placing it in a prompt, use `encode()` directly.

## Encode and decode data

```python
from dspy_toon import encode, decode

records = {
    "people": [
        {"name": "Alice", "age": 35},
        {"name": "Bob", "age": 28},
    ]
}

text = encode(records)
print(text)
assert decode(text) == records
```

```text
people[2]{name,age}:
  Alice,35
  Bob,28
```

Uniform nested fields can share a header too:

```text
people[2]{name,address{city,country}}:
  Alice,Paris,FR
  Bob,Berlin,DE
```

Use `ToonDecodeError` to handle invalid documents. The decoder checks declared array lengths and row widths. It also distinguishes numbers from numeric strings, and `null` from an empty array.

## Async and streaming

The adapter supports DSPy's synchronous and asynchronous call paths, demos and conversation history. For an asynchronous prediction, use `await extract.acall(text="Alice is 35.")` with the predictor above.

Streaming is opt-in:

```python
from dspy_toon import enable_toon_streaming

enable_toon_streaming()
```

Call this before creating DSPy stream listeners. Incremental scalar chunks contain raw TOON text, including quotes and escapes. The final prediction contains decoded values; arrays and keyed tables arrive in that final prediction.

Streaming currently patches DSPy's internal `StreamListener`. Check compatibility when changing DSPy versions. The [examples](https://github.com/Archelunch/dspy-toon/tree/main/examples) include complete usage patterns.

## Does it save tokens?

It depends on the data, model and output requirements. Reading TOON input and generating TOON output are separate tasks, so a good result on one does not establish the other.

Our Qwen3.8-27B-FP8 input experiment found lower total token use for uniform inventory, nested customer records and keyed service maps when the model returned JSON. Irregular logs and nested order arrays used more tokens. Output parsing also changed some apparent accuracy differences.

The [experiment report](https://github.com/Archelunch/dspy-toon/blob/main/benchmark_results/qwen_input_study/INTERPRETATION.md) includes sample sizes, paired uncertainty and parser analysis. These measurements describe one deployment and a set of synthetic workloads. Measure your own prompts before choosing an adapter.

## Upgrade to 0.4.0

This release updates the codec to TOON 4.1, uses DSPy 3.3.1's integration paths, and validates output types more strictly. Nested and keyed records can serialize differently from earlier versions. Update stored TOON examples and check callers that relied on permissive parsing.

See the [migration notes](https://github.com/Archelunch/dspy-toon/blob/main/UPGRADE.md) and [changelog](https://github.com/Archelunch/dspy-toon/blob/main/CHANGELOG.md) for details.

## Work on the package

```bash
git clone https://github.com/Archelunch/dspy-toon.git
cd dspy-toon
pip install -e ".[dev]"
pytest
```

To run the repository's experiments, install their dependencies separately with `pip install -e ".[benchmark]"`. This extra installs dependencies; it does not put benchmark scripts into the published package. Run those scripts from a checkout.

## License

MIT. See [LICENSE](https://github.com/Archelunch/dspy-toon/blob/main/LICENSE) and [NOTICE](https://github.com/Archelunch/dspy-toon/blob/main/NOTICE) for copyright and attribution notices.
