# Expanded TOON vs JSON study for a blog

Frozen before inference on 2026-09-05. Selection seed 20260906. This is a larger evaluation of already-corrected adapters; library source and prompts will not be tuned during this run.

## Questions

1. Do the current TOON 4.1 and DSPy 3.3.1 JSON prompt/parser adapters differ in task quality, reliability, and total tokens on new cases?
2. Does input compression help independently of the output format? Compare JSON/JSON, TOON/JSON, JSON/TOON, and TOON/TOON in a normalized 2x2 experiment.
3. Does native reasoning change the tradeoff? Use the same 32768-token cap in both modes and retain truncations.

## Matrix

- SOB text: 100 new contexts, actual JSON and TOON adapters, thinking off/on (400 calls).
- SOB OCR text: 100 new documents, actual adapters plus four factorial arms, thinking off/on (1200 calls).
- SOB meeting transcripts: all 94 eligible new transcripts, actual adapters, thinking off/on (376 calls).
- TableBench NumericalReasoning: 150 new, distinct tables, actual adapters plus four factorial arms, thinking off/on (1800 calls).
- BBEH mini: 100 new questions, actual adapters, thinking off/on (400 calls).

Total: 544 unique cases, 4176 calls. The previous pilot's contexts/tables/questions are excluded. SOB records are unique by record ID and exact context. TableBench tables are unique by normalized row dictionaries. BBEH excludes exact previous questions. This does not rule out semantic near-duplicates, shared source documents represented differently, or model training contamination.

Sampling is shuffled uniform over eligible rows, not the previous pilot's balanced difficulty sampling. One OCR gold and one transcript gold violated their JSON Schemas; five candidate tables had duplicate columns or inconsistent widths. These seven exclusions were made before any inference and are preserved verbatim in eligibility_exclusions.jsonl. The other 94 remaining transcript cases are all used. Source reference answers and questions are unchanged.

## Adapter comparison versus factorial experiment

Arms `json` and `toon` use each adapter's unmodified format and parse methods. A common unconstrained HTTP transport makes one request per case/arm/mode. This compares DSPy's prompt/parser path, not its complete __call__ backend selection: native JSON Schema constrained decoding, function calling, repair model calls and fallback model requests are not enabled. JSONAdapter's ordinary local json_repair parsing is retained; report raw syntax compliance separately from adapter-accepted output.

Arms `json_json`, `toon_json`, `json_toon`, and `toon_toon` name input_output serialization. They share a normalized envelope: the complete input dictionary is serialized as compact JSON or TOON in the user message. For a fixed output format, system messages differ only in the input-format label, and user messages differ only in input serialization. Output schema descriptions/instructions and response parser are held fixed for that output format. For JSON output, the stock JSONAdapter output-instruction section is retained while its input-marker section is removed. For TOON output, the current TOON rules are retained. These are experimental prompt variants, not shipped adapters. They isolate input encoding conditional on an output contract; changing output format also changes its instructions/schema rendering/parser, not just punctuation.

Do not pool factorial and actual-adapter arms as interchangeable implementations. The normalized JSON baseline uses compact whole-object JSON inputs, whereas stock DSPy JSON uses field markers around inputs.

## Runtime and accounting

Server-reported model: Qwen/Qwen3.8-27B-FP8, http://192.168.36.11:8007/v1. Model context limit returned by /models: 131072 tokens. DSPy 3.3.1; TOON 4.1 working-draft implementation in this repository. Temperature 0.6, top_p 0.95, per-case seed 20260906 + case index, maximum completion tokens 32768, native chat_template_kwargs.enable_thinking false/true. No explicit DSPy reasoning output field.

All 4176 requests are shuffled together once and executed at concurrency 50. Preserve every response, usage, finish reason, raw native reasoning and client-observed latency. One sample per cell; sampling and batching can remain nondeterministic despite seeds. No retries. The existing transport stops dispatch after three consecutive transport failures; recorded failures remain scored. A 600-second read timeout applies. Latency includes queueing/prefill/decoding and is not an isolated model speed benchmark. Server traffic and prefix caching are uncontrolled. Completion usage includes native reasoning, without a trustworthy separate reasoning-token count.

## Outcomes

Primary: SOB official schema-gated leaf-value exact match; TableBench official normalized numerical match; BBEH official correctness. Use the pinned official scorers preserved under ../qwen_20260905/scorers/. Also retain SOB token F1 and whole-document exact match. All errors remain in denominators.

Report adapter-accepted schema validity, raw format/schema compliance, truncation, prompt/completion/total token means, successful-answer yield per token budget, latency distributions, and paired differences with bootstrap confidence intervals. Do not average different families into a universal accuracy score. Analyze input and output format effects within the factorial experiment. Bootstrap by case, keeping paired arms together. Treat intervals as descriptive and disclose multiple comparisons.

Cost per successful BBEH/TableBench answer can be calculated as total tokens / correct answers, including failed calls' tokens. For SOB, leaf-level partial credit is a different quantity; do not label token/partial-score ratios as cost per correct document. Report strict document correctness separately. No monetary pricing is inferred for the user's server.

## Provenance and reproduction

Data/scorer revisions and licenses: ../qwen_20260905/manifest.json, scorer_sources.json and protocol.md. Primary sources: https://arxiv.org/abs/2604.25359 ; https://github.com/InterfazeAI/sob ; https://github.com/TableBench/TableBench ; https://github.com/google-deepmind/bbeh . This is five workloads from three benchmark families (SOB 2026; TableBench and BBEH 2025), not five new 2026 benchmark releases. OCR and audio modalities use supplied text, not raw media.

cases.jsonl, requests.jsonl, manifest.json and source_snapshot/ preserve the exact selected data and implementation. The model never sees gold/reference fields. The 32K budget differs from the original 8K pilot; do not attribute cross-run differences solely to more data or prompt fixes.

From the repository root:

```sh
DSPY_CACHEDIR=/private/tmp/dspy-toon-cache .venv/bin/python -m benchmarks.blog_comparison run
DSPY_CACHEDIR=/private/tmp/dspy-toon-cache .venv/bin/python -m benchmarks.blog_comparison score
```

## Analysis implementation

`benchmarks/blog_analysis.py` exports case-level raw-format diagnostics, grouped metrics and paired contrasts. It uses 10000 case-bootstrap resamples with seed 20260906. Both 95% and 99.5% score intervals are exported; the latter provide a conservative view of the ten actual-adapter comparisons. Ratios bootstrap paired token totals, not a mean of per-case ratios. Individual-arm confidence intervals in the chart are not substitutes for paired difference intervals.

Raw compliance uses an unfenced standard JSON load (rejecting non-JSON numeric constants) or the strict TOON decoder, followed by the exact result wrapper and original JSON Schema. It does not apply the adapter's code-fence removal, JSON repair or type coercion. Adapter validity is separately measured using the ordinary adapter parser and common validation. Truncated responses retain whatever task credit the parser/scorer can recover, while truncation remains explicitly counted.

A separately planned native-schema comparison uses matched compatible cases and is documented in ../qwen_blog_native_json/PROTOCOL.md. It will run after this main batch; it does not change the frozen main experiment.
