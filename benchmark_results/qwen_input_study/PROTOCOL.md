# Frozen input-comprehension study

Prepared before inference on 2026-09-05. Endpoint: user-provided Qwen/Qwen3.8-27B-FP8 at http://192.168.36.11:8007/v1. No adapter/product source tuning during the run.

## Track A: upstream reproduction on Qwen

Pin toon-format/toon f151a5d830d001bc244395b891183cba37e0d935 (package 4.1.1). Use its unmodified dataset/question generators, reference encoding including post-encoding corruption, type-aware answer normalizer and evaluation prompt. 244 questions across 13 datasets. Compare JSON pretty, compact JSON and TOON on all questions; CSV only on the 109 supported flat questions. Two thinking modes: 1,682 requests. Short-value output identical across input formats.

This reproduces dataset/prompt/scoring on a different deployment, not the original provider settings: explicit temperature .6, top_p .95, seed and 32,768-token cap replace upstream provider defaults. Four format subset; YAML/XML omitted. Separate ordinary questions, structure awareness and structural validation. For generalization, the 244 questions share only 13 source datasets; question-level intervals are conditional on that catalog. Report paired source-dataset cluster intervals on ordinary questions as an additional sensitivity check. Do not count repeated formats as independent examples. Declared-length advantage in corruption cases is a metadata benefit; no claim of equal information.

## Track B: independent tool-result inputs

500 deterministic synthetic datasets, independently seeded: 100 inventory, 100 nested customer records, 100 keyed service maps, 100 semi-uniform logs, 100 orders with nested item arrays. Each family has 25 datasets at each of 10/30/100/300 rows. Balanced tasks: numeric lookup, count, sum, multi-condition count and string lookup. One question per independent dataset. Gold computed directly from structured records before encoding.

Compare compact JSON with this extension's TOON input encoding. Both request exactly the same JSON result field as a string. Short input-format primers differ. Frozen inputs are verified lossless by the reference decoder; all 500 encodings are also byte-identical to the reference encoder. These are application-shaped synthetic tool-result payloads inside user messages, not live external APIs or end-to-end tool-using agents.

500 datasets × 2 formats × 2 thinking modes = 2,000 calls. A preselected 100 datasets (index divisible by five, covering all families/sizes/task types) receive a second seed in all four configurations: 400 additional calls. Report first-seed results separately; repeats estimate seed sensitivity, not sample-size expansion.

Tool primary score: parse standard JSON and compare result scalar with deterministic gold using case-insensitive stripped string equality (numeric gold are integers). Accept numeric scalars for semantic scoring but report strict output contract separately: exactly one result key and string value, no fences/repair. Also report a separate outer-code-fence-removal sensitivity score. Do not silently change the primary scorer based on observed failures.

## Transport and accounting

4,082 scored requests in one deterministically shuffled queue at concurrency 50. One HTTP call per job; no retries, fallback, native constrained decoding or LLM repair. Generation settings identical across paired formats: temperature .6, top_p .95, max_tokens 32768, native enable_thinking false/true, per-case seed 20260907+index (+10000 for repeat). Save raw reasoning and final output. Request-level 600-second timeout; runner stops after three consecutive observed transport failures. Resume skips already recorded jobs including errors.

Report failures and truncations in denominator; schema/answer success, prompt/completion/total tokens and end-to-end latency. Completion tokens include reasoning. Latency at concurrency 50 is descriptive; caching and server load are uncontrolled. No token cost estimate from an unrelated tokenizer. Actual provider usage is authoritative.

Freeze cases, requests, source SHA-256 and upstream git archive before calls. Report 10,000-resample paired bootstrap 95% intervals as exploratory; multiple slices are not independent confirmatory tests. For tool cases, bootstrap independent datasets and keep repeated seeds together when summarizing repeats. No claim of multi-model universality or training-contamination control for upstream public data.
