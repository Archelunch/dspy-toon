# Schema guidance iteration 2

100 fresh cases, 400 requests. Each case was run through the original name-free adapter and this candidate, with reasoning off and on. Requests were shuffled and interleaved at concurrency 50 on Qwen/Qwen3.8-27B-FP8. Temperature 0.6, top_p 0.95, per-case matched seeds, maximum 32768 output tokens. Parsing, fallback and scoring logic were unchanged.

## Changes

Remove redundant type and required-field lines while retaining constraints; improve rule selection for unions and keyed entries.

## Paired results

| Task | Reasoning | Cases per arm | Score baseline / candidate | Parse baseline / candidate | Total tokens change | Score delta 95% CI, pp |
|---|---|---:|---:|---:|---:|---:|
| bbeh_mini | off | 20 | 15.0% / 15.0% | 80.0% / 90.0% | -18.7% | [-15.0, +15.0] |
| bbeh_mini | on | 20 | 50.0% / 40.0% | 80.0% / 70.0% | +10.1% | [-35.0, +15.0] |
| sob_constraints | off | 20 | 70.0% / 85.0% | 70.0% / 85.0% | +8.6% | [-10.0, +40.0] |
| sob_constraints | on | 20 | 50.0% / 95.0% | 50.0% / 95.0% | +4.6% | [+25.0, +65.0] |
| sob_image | off | 20 | 60.7% / 49.0% | 95.0% / 80.0% | -0.2% | [-23.6, -2.1] |
| sob_image | on | 20 | 58.4% / 55.3% | 100.0% / 95.0% | -3.9% | [-12.6, +4.0] |
| sob_text | off | 20 | 68.6% / 78.3% | 95.0% / 100.0% | -3.9% | [-1.2, +23.8] |
| sob_text | on | 20 | 79.1% / 80.6% | 100.0% / 100.0% | -12.6% | [-3.8, +6.8] |
| tablebench_numeric | off | 20 | 35.0% / 40.0% | 95.0% / 100.0% | +5.3% | [+0.0, +15.0] |
| tablebench_numeric | on | 20 | 85.0% / 90.0% | 100.0% / 100.0% | -3.2% | [+0.0, +15.0] |

Scores are benchmark-specific: SOB leaf-value exact match, the archived TableBench metric and BBEH task correctness. They should not be pooled into a single accuracy score. Confidence intervals use 10,000 paired bootstrap resamples within each task and reasoning setting; they are unadjusted for multiple comparisons and unstable with small samples. A zero-width interval can reflect no observed discordance in a small sample, not proven equivalence.

## Synthetic constraint checks

The synthetic group copies generated nested/keyed records with enum choices, bounded numbers, leading-zero strings, embedded commas and backslashes. It is an engineering stress test, not an external benchmark. Exact decoded-value equality below is the primary metric for this group; it treats numerically equal integers and floats as equal. The generic SOB strict serialized-JSON metric is unsuitable here because it distinguishes number spellings such as 0 and 0.0.

| Reasoning | Exact decoded payload, baseline | Exact decoded payload, candidate |
|---|---:|---:|
| off | 14/20 | 17/20 |
| on | 10/20 | 19/20 |

## Accounting

Baseline: 173/200 accepted, 4 truncated. Candidate: 183/200 accepted, 6 truncated. Both arms had zero transport errors. Truncated outputs remain in all scores.

No source-input overlap with earlier adapter experiments or previous iteration rounds. SOB image tasks use supplied OCR text rather than image requests. No fresh audio suite was used because the local source had only one unused transcript context.

Raw requests, responses, preparation code, source snapshots, manifests and detailed comparisons remain local. This report contains aggregates only. No model calls used JSON or Chat adapters.
