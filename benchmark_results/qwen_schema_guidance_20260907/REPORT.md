# Schema guidance iteration 1

100 fresh cases, 400 requests. Each case was run through the original name-free adapter and this candidate, with reasoning off and on. Requests were shuffled and interleaved at concurrency 50 on Qwen/Qwen3.8-27B-FP8. Temperature 0.6, top_p 0.95, per-case matched seeds, maximum 32768 output tokens. Parsing, fallback and scoring logic were unchanged.

## Changes

Nested constraint/type descriptions, field-specific reminders, type-selected syntax rules and explicit escape guidance.

## Paired results

| Task | Reasoning | Cases per arm | Score baseline / candidate | Parse baseline / candidate | Total tokens change | Score delta 95% CI, pp |
|---|---|---:|---:|---:|---:|---:|
| bbeh_mini | off | 20 | 25.0% / 25.0% | 65.0% / 80.0% | -18.3% | [-15.0, +15.0] |
| bbeh_mini | on | 20 | 50.0% / 50.0% | 95.0% / 85.0% | +15.6% | [-20.0, +20.0] |
| sob_constraints | off | 20 | 75.0% / 79.7% | 75.0% / 80.0% | +16.6% | [-10.7, +20.0] |
| sob_constraints | on | 20 | 20.0% / 100.0% | 20.0% / 100.0% | -11.8% | [+60.0, +95.0] |
| sob_image | off | 20 | 61.8% / 55.7% | 85.0% / 75.0% | +16.6% | [-17.6, +2.1] |
| sob_image | on | 20 | 54.9% / 49.9% | 75.0% / 70.0% | -0.1% | [-18.1, +6.4] |
| sob_text | off | 20 | 82.4% / 77.8% | 90.0% / 90.0% | +0.8% | [-16.9, +4.1] |
| sob_text | on | 20 | 86.5% / 82.8% | 100.0% / 95.0% | -8.7% | [-16.0, +5.6] |
| tablebench_numeric | off | 20 | 22.5% / 37.5% | 100.0% / 100.0% | +5.9% | [+0.0, +30.0] |
| tablebench_numeric | on | 20 | 85.0% / 85.0% | 100.0% / 100.0% | +13.4% | [+0.0, +0.0] |

Scores are benchmark-specific: SOB leaf-value exact match, the archived TableBench metric and BBEH task correctness. They should not be pooled into a single accuracy score. Confidence intervals use 10,000 paired bootstrap resamples within each task and reasoning setting; they are unadjusted for multiple comparisons and unstable with small samples. A zero-width interval can reflect no observed discordance in a small sample, not proven equivalence.

## Synthetic constraint checks

The synthetic group copies generated nested/keyed records with enum choices, bounded numbers, leading-zero strings, embedded commas and backslashes. It is an engineering stress test, not an external benchmark. Exact decoded-value equality below is the primary metric for this group; it treats numerically equal integers and floats as equal. The generic SOB strict serialized-JSON metric is unsuitable here because it distinguishes number spellings such as 0 and 0.0.

| Reasoning | Exact decoded payload, baseline | Exact decoded payload, candidate |
|---|---:|---:|
| off | 15/20 | 15/20 |
| on | 4/20 | 20/20 |

## Accounting

Baseline: 161/200 accepted, 1 truncated. Candidate: 175/200 accepted, 3 truncated. Both arms had zero transport errors. Truncated outputs remain in all scores.

No source-input overlap with earlier adapter experiments or previous iteration rounds. SOB image tasks use supplied OCR text rather than image requests. No fresh audio suite was used because the local source had only one unused transcript context.

Raw requests, responses, preparation code, source snapshots, manifests and detailed comparisons remain local. This report contains aggregates only. No model calls used JSON or Chat adapters.
