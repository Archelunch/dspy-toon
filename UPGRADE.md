# DSPy 3.3.1 / TOON 4.1 upgrade

Prepared locally on 2026-09-05 as package version 0.4.0 (unreleased), from repository commit `d0cf479cc54b85082179266802b0ddec5f9f4544`.

## Assessment

TOON 4.1 is useful for this extension. Its main compression improvements come from 4.0: nested field groups let uniform nested records share one header, and keyed tables do the same for dictionaries of records. Version 4.1 tightens encoding and parsing rules so implementations agree on the wire format. It does not improve every shape, and serialization savings do not establish better LLM accuracy.

The [specification changelog](https://github.com/toon-format/spec/blob/d6db4b04303bdea132351ce45aed612311c850b2/CHANGELOG.md) and [normative specification](https://github.com/toon-format/spec/blob/d6db4b04303bdea132351ce45aed612311c850b2/SPEC.md) were reviewed at commit `d6db4b04303bdea132351ce45aed612311c850b2`. Version 4.1 is dated 2026-07-26 and is still a working draft.

Replacing the codec with the official Python package would reduce local ownership, but the [Python implementation inspected](https://github.com/toon-format/toon-python/tree/e475c82e9da03dfaf88c0b277dee6b5d17100b13) identifies itself as `0.9.0-beta.1` and lacks the new nested-field/keyed-table machinery. Depending on it would not deliver this upgrade. This branch keeps a local codec, with shared parsing paths instead of the old duplicated list-item branches. It introduces no codec dependency or JavaScript subprocess.

## What changed

- **DSPy:** upgrade the locked version from 3.0.4 to the latest stable release returned by [PyPI](https://pypi.org/project/dspy/3.3.1/), 3.3.1. Require `dspy>=3.3.1,<4`; retain Python 3.10–3.14 support. Reuse DSPy's inherited call/async, callbacks, demo formatting, native response handling and History support. Missing optional outputs use DSPy's defaults/default factories/nullable handling.
- **Codec:** nested field groups, keyed tables, canonical empty arrays, comments, Unicode control escapes, surrogate rejection, BOM handling, canonical numeric encoding, exact numeric token grammar, duplicate-key checks, and shared structural/count/width validation.
- **Prompt formatting:** encode whole field dictionaries. Previously nested values could be emitted without their required indentation, models inside lists became null, and explicit null demo values disappeared. Schema examples now come from the encoder and actual annotation shape, replacing hardcoded Alice/Bob rows with arbitrary column counts.
- **Output parsing:** decode a complete document and validate each field's annotation and constraints with Pydantic. The old regex paths could split quoted commas, ignore declared counts, confuse null with missing values and return unchecked strings after conversion failures. Complete JSON and fenced documents remain accepted. Invalid fields produce useful `AdapterParseError` messages.
- **History:** DSPy History dictionaries previously fell into a legacy user/assistant branch and could disappear. Native History now uses DSPy's formatter; legacy user/assistant pairs remain supported.
- **Streaming:** keep the opt-in API, preserve spaces across chunks, anchor scalar starts to field boundaries and handle a following scalar or array header split across chunks. Other adapters delegate to their original methods. Incremental chunks are raw TOON text, not decoded strings. Arrays and keyed tables are returned in the final prediction. This remains a shim over DSPy's internal listener API, not a stable upstream extension point.
- **Packaging:** move `datasets` out of core dependencies into the benchmark extra; retain the existing benchmark data and scripts. Package `__version__` now comes from installed metadata instead of the stale hardcoded 0.1.0.

The package retains its existing four-module layout: public exports, DSPy adapter, codec and optional streaming support. The adapter and codec share serialization logic; benchmark-only concerns remain outside the library. No new tests, checker scripts or CI jobs were added. The subsequent user-requested Qwen comparison adds a benchmark runner and archived experiment artifacts. Three existing test expectations were updated for changed schema examples, canonical empty arrays and corrected row indentation.

## Local token measurements

Measured with `tiktoken` 0.12.0, `cl100k_base`, using the original codec at the base commit and the new codec on the same Python values. JSON uses `json.dumps(data, separators=(",", ":"), ensure_ascii=False)`. These counts cover serialized data only, excluding instructions, type schemas, reasoning and any model response variability.

| Dataset | Old codec | TOON 4.1 | Compact JSON |
|---|---:|---:|---:|
| Existing simple object | 26 | 26 | 26 |
| Existing nested object | 64 | 64 | 55 |
| Existing small user list | 69 | 69 | 102 |
| Existing medium user list | 249 | 249 | 402 |
| Existing large user list | 1,209 | 1,209 | 2,002 |
| Existing product catalog | 957 | 957 | 1,398 |
| Existing API response | 111 | 111 | 104 |
| Existing mixed array | 61 | 61 | 42 |
| 20 uniform nested records | 543 | 252 | 423 |
| 20 keyed uniform records | 301 | 189 | 224 |

The first eight datasets are the unchanged `BENCHMARK_DATASETS` in `benchmarks/token_comparison.py`. The two additional synthetic shapes were evaluated in the terminal, without adding datasets or scripts:

```python
nested = {
    "people": [
        {"id": i, "name": f"Person {i}", "address": {"city": "Paris", "country": "FR"}}
        for i in range(20)
    ]
}
keyed = {"people": {f"user{i}": {"age": 20 + i, "city": "Paris"} for i in range(20)}}
```

The nested shape saves **53.6%** against the old codec and **40.4%** against compact JSON. The keyed shape saves **37.2%** and **15.6%**, respectively. Existing flat tables stay byte/token-equivalent on these samples. Compact JSON remains smaller for the existing irregular nested, API-response and mixed-array samples.

The serialization measurements above involve no model calls. A subsequent Qwen 3.8 model comparison is documented in [the benchmark findings](benchmark_results/qwen_20260905/FINDINGS.md), including native thinking, failures, token use, and prompt-contract corrections. Historical README results remain separate; their accuracy and total-token claims cannot be transferred to 0.4.0. No model comparison against the old TOON 3.x adapter was run, so the experiment does not establish a causal quality gain from the specification upgrade.

## Migration

- Upgrade readers before writers when sending TOON to another system: old decoders do not understand nested field groups or keyed tables.
- `[]` is the canonical empty array; `null` is a separate nullable value. The decoder still reads legacy `[0]:` / `key[0]:`.
- Full lines beginning with `#` after leading spaces are comments in v4. Re-encode stored v3 hash-leading data with the old decoder before reading it under v4; otherwise it can disappear as a comment. The new encoder quotes such data.
- The old codec's tabular rows following `- field[N]{...}:` were under-indented even for v3. Rows must be **four spaces deeper than the hyphen** with default indentation; sibling fields are two spaces deeper. Strict decoding now rejects the old erroneous layout. Re-encode affected data with the previous reader and new writer; do not rely on non-strict mode to repair it.
- Valid TOON is required for structured outputs. Malformed regex-only layouts, wrong counts, duplicate keys, wrong field types and undeclared output fields now error. A complete JSON object remains a fallback. Explicit nulls and missing nullable/defaulted fields follow DSPy 3.3.1 semantics.
- `indentSize` is supported; `indent` remains an alias. Indentation must be a positive integer. Non-strict decoding floors space indentation and expands leading tabs to `indentSize` tab stops. Count mismatches do not truncate data; missing row cells omit fields and extra cells are ignored in non-strict mode.
- The obsolete `lengthMarker` option is rejected; it was removed upstream in TOON 2.0. Dotted keys are always literal; no key-folding/path-expansion option is introduced.
- As an explicitly retained compatibility extension, the decoder accepts old non-keyed `[N,]` headers. The encoder emits the 4.1 comma form `[N]`. `[N:,]` keyed headers remain invalid.
- Python integers retain arbitrary precision within Python's runtime conversion limits. Decimal/exponent tokens use finite Python floats; out-of-range values error. Decimal inputs use float approximation; dates use ISO 8601; model values use `model_dump()` recursively; unsupported Python values become null. Unicode normalization is not applied.
- Install `.[benchmark]` to use dataset benchmarks; `datasets` is no longer guaranteed by the core package.
- Regenerate/review serialized demos and compiled optimizer artifacts before reusing them. Existing artifacts remain unchanged in this branch.

## Verification

On Python 3.12.9 with DSPy 3.3.1:

- Existing suite: **71 passed** (also 71 passed on the original DSPy 3.0.4 baseline).
- Existing Ruff lint/format and mypy checks pass.
- Upstream fixtures at the pinned spec commit: **179 encoding + 359 decoding = 538 passed**, including non-strict cases. Run directly from the upstream JSON fixtures; no harness or copied fixtures added to this repository. Fixture coverage supports the implementation but is not a proof of every normative rule.
- Terminal integration checks: synchronous and asynchronous adapter calls through a local fake LM; nested Pydantic models; nullable outputs; constrained invalid output rejection; fenced JSON fallback; DSPy History without input mutation; scalar streaming with split headers.
- Serialization round trips pass for all ten measured datasets.
- Source distribution and wheel build successfully. Wheel metadata records version 0.4.0, the DSPy constraint, benchmark-only datasets and the packaged `py.typed` marker.

Other Python versions were not exercised locally. The subsequent benchmark exercises the adapters against the user-provided Qwen 3.8 vLLM endpoint. DSPy's internal streaming API remains a compatibility risk for future DSPy releases; the lockfile records the exact verified version.
