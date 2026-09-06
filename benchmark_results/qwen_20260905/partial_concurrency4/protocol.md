# Qwen 3.8 adapter comparison protocol

Frozen before the main run, 2026-09-05. This is a paired 100-case pilot: 20 cases per workload, four adapters, native thinking off/on (800 scored calls). Forty additional transport/scoring preflight calls are stored separately and excluded from the main comparison. Cases were selected before any model response, with seed 42; no performance-based case selection or prompt tuning.

## Workloads and sources

- SOB text extraction: 20 cases, balanced across medium/hard schema categories, from 5,000 test records.
- SOB OCR-document extraction: 20 cases, balanced across medium/hard schema categories, from 209 records.
- SOB meeting-transcript extraction: 20 cases, five from each context-length quartile, from 115 records.
- TableBench NumericalReasoning: 20 randomly sampled cases from 397 records, retaining original questions and answers. Tables become typed row dictionaries with original column order.
- BBEH mini: 20 randomly sampled cases from 460 records. This small sample is not stratified across BBEH task categories.

These are five workloads from three benchmark families, not five independent benchmark releases. SOB is a 2026 structured-output benchmark; TableBench and BBEH are established 2025 benchmarks relevant to table and hard reasoning. SOB image/audio inputs are the supplied OCR/transcript text, not raw images/audio. This is an adapted adapter comparison, not a reproduction of official leaderboards.

Primary sources: [SOB paper](https://arxiv.org/abs/2604.25359), [SOB repository](https://github.com/InterfazeAI/sob), [TableBench](https://github.com/TableBench/TableBench), [BBEH](https://github.com/google-deepmind/bbeh).

Pinned data revisions:
- interfaze-ai/sob: c118e38abdef6a8e1beba183405c70b28ff7d5a8
- Multilingual-Multimodal-NLP/TableBench: a23c244f9ccae1ea238d614fe7620984707e411a
- BBEH: 80d12ca916b7158f22293fcf3144f4d3d854d4be

Dataset and prompt hashes are in manifest.json. Official scorer revisions and hashes are in scorer_sources.json. SOB code is MIT; underlying text/meeting/OCR data retain their upstream licenses. BBEH data is CC BY 4.0.

## Compared implementations

- TOON: this repository's upgraded TOON 4.1 adapter, current source frozen by hash.
- JSON: DSPy 3.3.1 JSONAdapter.
- BAML: this repository's BAML-inspired DSPy adapter, which renders compact schemas and uses DSPy's JSON parser. This is not BoundaryML's full BAML runtime or SAP parser.
- Chat: DSPy 3.3.1 ChatAdapter, JSON fallback disabled.

Every arm receives the same typed signature, field descriptions, case content, and task instruction. Adapter.format renders its own messages; a shared HTTP client sends one unconstrained request; Adapter.parse and common Pydantic plus original JSON Schema validation precede the official task scorer. No retries, repair calls, few-shot examples, tools, constrained decoding, or fallback model calls. This isolates prompt/serialization/parser behavior; it does not compare native JSON-schema decoding or full BAML.

## Generation and accounting

Model reported by the server: Qwen/Qwen3.8-27B-FP8; server fingerprint observed during probes: vllm-0.27.1-ef403332. Endpoint: http://192.168.36.11:8007/v1. The model name is server-reported; checkpoint files were not inspected.

Temperature 0.6, top_p 0.95, maximum 8,192 completion tokens, four concurrent requests. The same per-case seed is used across arms. All 800 jobs are shuffled once. Native thinking uses chat_template_kwargs.enable_thinking; no visible DSPy ChainOfThought field is added. Sampling and vLLM batching can still introduce nondeterminism.

Store every response, native reasoning, finish reason, token usage, latency and error. Completion tokens include reasoning; the server does not report a reliable separate reasoning-token count. HTTP, parse, schema and truncation failures remain in denominators. Resume skips all recorded jobs including failed ones. Stop on three consecutive transport failures. Latencies are end-to-end at concurrency four, not isolated inference timings; prefix caching and other server traffic are not controlled.

Primary scores: SOB official gated leaf-value exact match, TableBench official normalized/rounded answer-match score, BBEH official correctness. Whole-document SOB exact match and token F1 are secondary. These primary scores have different meanings and should be compared within each workload, not treated as a universal accuracy score. Zero-shot adapter wrapping and typed validation can differ from each benchmark's original prompting. All selected golds satisfy the original JSON Schema. Literal annotation defects and answer ambiguity can still remain; no golds are repaired after seeing responses.

Twenty cases per cell provide exploratory evidence only. Pair comparisons by case, include failures, and show uncertainty; small differences are not a leaderboard claim. Thinking modes share the same total completion cap, so truncation must be reported when interpreting hard-reasoning accuracy.

## Reproduction

From the repository root with benchmark/dev extras installed:

```sh
DSPY_CACHEDIR=/private/tmp/dspy-toon-cache .venv/bin/python -m benchmarks.qwen_adapter_comparison run
DSPY_CACHEDIR=/private/tmp/dspy-toon-cache .venv/bin/python -m benchmarks.qwen_adapter_comparison score
```

The frozen requests can also be replayed independently from requests.jsonl. Scoring currently uses the pinned upstream checkouts under /private/tmp; restore those revisions if absent. cases.jsonl contains the selected source records and golds; requests.jsonl contains only model-visible inputs and schemas, never reference answers.
