# Qwen 3.8 adapter benchmark findings

**TOON 4.1 is useful for compact table inputs, but this pilot does not support using it as a universal replacement for JSON.** JSON was more reliable without thinking, while reasoning dominated end-to-end token costs. The experiment also found and corrected two concrete prompt-contract defects.

Model: `Qwen/Qwen3.8-27B-FP8`, as reported by the user’s vLLM endpoint; fingerprint `vllm-0.27.1-ef403332`; DSPy 3.3.1. Twenty fixed cases per workload, native thinking off/on, concurrency **50**. No model retries, repair calls, native constrained decoding, or fallback model calls.

## Runs and comparison boundaries

- Initial comparison: 800 calls, four adapters × five workloads × 20 cases × two thinking modes, 8,192-token cap.
- Prompt correction follow-up: 400 calls, TOON/BAML only, all the same cases and generation settings. TOON descriptions changed 25 of 100 case prompts; BAML output instructions changed all 100.
- BBEH budget check: 80 calls, all 20 BBEH cases × four adapters, thinking on, 32,768-token cap.
- Separate preflight: 40 calls. A partial concurrency-four run contains 88 completed calls; it was archived when the user requested 50 concurrent requests. Four in-flight requests were cancelled, with unknown usage. Two initial endpoint probes are also separate.

The tables below use **corrected TOON/BAML from the follow-up and JSON/Chat from the initial run**. All cases are paired, but these arms ran in separate batches. Corrections were diagnosed on this dataset: these are exploratory development results, not held-out validation. Full initial results remain in [REPORT.md](REPORT.md). BAML means the repository’s **BAML-inspired DSPy adapter**, not BoundaryML’s runtime or SAP parser.

## Five workloads, both thinking modes

Cell values are **thinking off / thinking on**, in percent, with the 8K completion cap. SOB uses official schema-gated leaf-value exact match, TableBench uses official normalized numerical answer match, and BBEH uses official correctness. Invalid outputs stay in the denominator and score zero. Compare within rows; these metrics are not interchangeable.

| Workload | TOON 4.1 | JSON | BAML-inspired | Chat |
|---|---:|---:|---:|---:|
| SOB text | 67.6 / 82.9 | 77.4 / 79.5 | 75.3 / 82.8 | 77.4 / 69.3 |
| SOB OCR text | 63.2 / 56.6 | 61.3 / 58.3 | 55.1 / 55.9 | 57.3 / 56.5 |
| SOB meeting transcripts | 24.5 / 10.6 | 23.8 / 12.7 | 20.1 / 18.5 | 23.7 / 13.2 |
| TableBench numerical | 35.0 / 85.0 | 50.0 / 85.0 | 45.0 / 75.0 | 40.0 / 80.0 |
| BBEH mini | 20.0 / 30.0 | 10.0 / 30.0 | 20.0 / 30.0 | 20.0 / 30.0 |

SOB is a 2026 benchmark, used here in three modalities: text, supplied OCR text, and supplied meeting transcripts. TableBench and BBEH are 2025 releases, included for numerical and hard reasoning. Thus these are five workloads from three benchmark families, not five newly released 2026 datasets. No raw images or audio were sent.

## Reliability and token use

Each arm below has the same 100 cases. Completion tokens include native reasoning. Validity includes parsing, typed validation, and the original JSON Schema. Truncations can overlap invalid outputs.

| Thinking | Adapter | Valid / 100 | Truncated | Prompt tokens | Completion tokens | Mean latency |
|---|---|---:|---:|---:|---:|---:|
| off | toon | 92 | 2 | 541,298 | 32,331 | 6.0s |
| off | json | 97 | 2 | 554,116 | 39,706 | 6.8s |
| off | baml | 97 | 1 | 532,068 | 32,124 | 5.6s |
| off | chat | 91 | 7 | 551,471 | 84,080 | 13.6s |
| on | toon | 71 | 28 | 544,898 | 415,584 | 61.3s |
| on | json | 76 | 22 | 557,716 | 335,485 | 51.2s |
| on | baml | 78 | 20 | 535,668 | 333,809 | 49.4s |
| on | chat | 73 | 22 | 555,071 | 344,175 | 53.5s |

TOON’s TableBench prompt tokens were **31.3% lower** than JSON’s (20,477 vs 29,822). Across all five workloads, TOON used **3.4% fewer total tokens with thinking off**, but **7.5% more with thinking on**. These include failures, so token savings are not equivalent to cost per successful answer. Larger reasoning completions can erase serialization savings.

Latencies are descriptive end-to-end measurements at concurrency 50, not isolated inference timings. Prefix-cache state, batching, server traffic, and the separately executed follow-up limit speed comparisons. No price or monetary cost is inferred for the user’s server.

## BBEH: does a larger reasoning budget help?

All 20 cases were repeated for every adapter at 32K, not only failed cases. The 8K reference uses corrected TOON/BAML and initial JSON/Chat. Both columns have native thinking enabled.

| Adapter | Score: 8K → 32K | Valid: 8K → 32K | Truncated: 8K → 32K | 32K completion tokens |
|---|---:|---:|---:|---:|
| toon | 30.0 → 60.0 | 9/20 → 17/20 | 11 → 3 | 309,421 |
| json | 30.0 → 55.0 | 9/20 → 18/20 | 11 → 2 | 287,415 |
| baml | 30.0 → 50.0 | 9/20 → 13/20 | 11 → 6 | 361,902 |
| chat | 30.0 → 55.0 | 8/20 → 15/20 | 12 → 5 | 343,076 |

This measures sensitivity to the completion budget, not an adapter-independent estimate of reasoning ability. One stochastic sample per cell and the repeated dataset still limit causal conclusions. There is no thinking-off 32K arm.

## Code changes from the benchmark

- **TOON:** preserve nested array field descriptions. The frozen baseline contained 388 of 489 SOB schema descriptions; corrected prompts contain all 489. This affects 25 of the 60 SOB cases and keeps the original typed output contract.
- **BAML-inspired adapter:** replace conflicting Chat-style output markers with one JSON object containing the output field keys its inherited JSON parser expects. Also make recursion tracking local to each branch. Off-mode validity changed from 29/100 to 97/100 in the follow-up; this is an adapter implementation correction, not evidence against BAML as a format/runtime.
- Add the requested resumable benchmark runner with frozen requests, raw responses, official scorers, and explicit benchmark-only dependencies. Existing tests remain at 71; no tests or CI checkers were added. Lint, formatting, typing and existing tests pass.

Remaining TOON failures include missing quotes around delimiter-containing strings, wrong array lengths, and invalid typed values. Strict decoding catches these; accepting malformed arrays would hide data loss. Preserving schema information helps, but spec compliance of the codec does not guarantee model compliance.

## What I would use

- **JSON as the conservative general default** for this Qwen endpoint. It has 97/100 valid off-mode outputs in this pilot and avoids the additional TOON grammar requirements.
- **TOON selectively for table-heavy inputs**, where repeated keys dominate prompt size. Measure end-to-end task quality and token use; the 31% TableBench prompt saving did not establish better accuracy.
- **Native thinking for tasks that benefit from computation**, especially TableBench. Avoid enabling it indiscriminately for extraction: the 8K cap caused many transcript outputs to truncate, and additional reasoning increased cost.
- Treat the corrected BAML-inspired adapter as a credible local baseline. A comparison against the full BAML runtime or native JSON-schema constrained decoding remains a different experiment.
- Keep TOON 4.1 for the codec improvements and interoperability, but do not claim the specification upgrade itself improved model accuracy: no TOON 3.x model arm was included.

## Uncertainty

Only 20 cases per workload: one binary success changes accuracy by five percentage points. Below are descriptive 95% paired bootstrap intervals for corrected TOON minus JSON (10,000 resamples; not adjusted for multiple comparisons). Cases are shared, but phase, sampling and prompt-correction effects remain.

| Workload | Thinking | TOON − JSON, percentage points [95% CI] |
|---|---|---:|
| SOB text | off | -9.7 [-20.3, -1.7] |
| SOB text | on | +3.4 [-2.1, +9.4] |
| SOB OCR text | off | +1.9 [-8.5, +10.8] |
| SOB OCR text | on | -1.7 [-17.9, +13.8] |
| SOB meeting transcripts | off | +0.7 [-5.4, +7.8] |
| SOB meeting transcripts | on | -2.1 [-9.9, +6.6] |
| TableBench numerical | off | -15.0 [-30.0, +0.0] |
| TableBench numerical | on | +0.0 [-15.0, +15.0] |
| BBEH mini | off | +10.0 [+0.0, +25.0] |
| BBEH mini | on | +0.0 [-15.0, +15.0] |

SOB’s exact-match metric can penalize paraphrases and literal annotation artifacts; source golds were not modified. All selected golds satisfy their original schemas and self-score correctly. For a secondary view, the scored artifacts retain SOB token F1 and whole-document exact match. Cases were selected before model calls; text/OCR were balanced by schema difficulty, transcripts stratified by length quartile, and TableBench/BBEH randomly sampled. The sample is not weighted to the full benchmark distributions.

## Sources and reproducibility

Primary sources: [SOB paper (2026)](https://arxiv.org/abs/2604.25359), [SOB repository/scorer](https://github.com/InterfazeAI/sob), [TableBench](https://github.com/TableBench/TableBench), [BBEH](https://github.com/google-deepmind/bbeh).

- [Initial protocol](protocol.md), [full initial report](REPORT.md), [initial metrics](metrics.csv)
- [Correction follow-up protocol](../qwen_20260905_prompt_fixes/protocol.md), [correction metrics](../qwen_20260905_prompt_fixes/metrics.csv)
- [32K protocol](../qwen_20260905_bbeh_32k/protocol.md), [32K metrics](../qwen_20260905_bbeh_32k/metrics.csv)
- [Final comparison CSV](final_comparison.csv), [JSON](final_comparison.json), [paired intervals](final_toon_vs_json_paired.json)
- Each run directory contains frozen requests, case identities, full raw responses/reasoning, per-case scoring, manifests and source snapshots. Pinned upstream scorer files and available licenses are in scorers/.

Total scored experimental calls: **1,280**, with **0 transport errors**, consuming **10,359,040 reported tokens**. Separate preflight, archived partial run, probes and cancelled work are additional. This accounting is not a price estimate.
