# Schema guidance iteration 3

70 fresh cases, 280 requests. Each case was run through the original name-free adapter and this candidate, with reasoning off and on. Requests were shuffled and interleaved at concurrency 50 on Qwen/Qwen3.8-27B-FP8. Temperature 0.6, top_p 0.95, per-case matched seeds, maximum 32768 output tokens. Parsing, fallback and scoring logic were unchanged.

## Changes

Retain concise constraints and reminders; explicitly name commas and require double quotes around each string item/cell in arrays.

## Paired results

| Task | Reasoning | Cases per arm | Score baseline / candidate | Parse baseline / candidate | Total tokens change | Score delta 95% CI, pp |
|---|---|---:|---:|---:|---:|---:|
| bbeh_mini | off | 10 | 0.0% / 10.0% | 80.0% / 100.0% | -37.5% | [+0.0, +30.0] |
| bbeh_mini | on | 10 | 60.0% / 60.0% | 70.0% / 80.0% | -5.1% | [-30.0, +30.0] |
| sob_constraints | off | 20 | 75.0% / 90.0% | 75.0% / 90.0% | +12.1% | [-5.0, +35.0] |
| sob_constraints | on | 20 | 50.0% / 100.0% | 50.0% / 100.0% | -20.3% | [+30.0, +70.0] |
| sob_image | off | 20 | 34.0% / 37.1% | 65.0% / 75.0% | +3.2% | [-4.2, +11.0] |
| sob_image | on | 20 | 40.8% / 44.5% | 80.0% / 85.0% | -4.9% | [-1.5, +8.8] |
| sob_text | off | 10 | 88.7% / 88.8% | 100.0% / 100.0% | -3.2% | [-8.6, +7.4] |
| sob_text | on | 10 | 90.7% / 90.7% | 100.0% / 100.0% | -18.0% | [+0.0, +0.0] |
| tablebench_numeric | off | 10 | 80.0% / 60.0% | 100.0% / 100.0% | +9.9% | [-50.0, +0.0] |
| tablebench_numeric | on | 10 | 100.0% / 90.0% | 100.0% / 100.0% | +1.0% | [-30.0, +0.0] |

Scores are benchmark-specific: SOB leaf-value exact match, the archived TableBench metric and BBEH task correctness. They should not be pooled into a single accuracy score. Confidence intervals use 10,000 paired bootstrap resamples within each task and reasoning setting; they are unadjusted for multiple comparisons and unstable with small samples. A zero-width interval can reflect no observed discordance in a small sample, not proven equivalence.

## Synthetic constraint checks

The synthetic group copies generated nested/keyed records with enum choices, bounded numbers, leading-zero strings, embedded commas and backslashes. It is an engineering stress test, not an external benchmark. Exact decoded-value equality below is the primary metric for this group; it treats numerically equal integers and floats as equal. The generic SOB strict serialized-JSON metric is unsuitable here because it distinguishes number spellings such as 0 and 0.0.

| Reasoning | Exact decoded payload, baseline | Exact decoded payload, candidate |
|---|---:|---:|
| off | 15/20 | 18/20 |
| on | 10/20 | 20/20 |

## Accounting

Baseline: 109/140 accepted, 3 truncated. Candidate: 128/140 accepted, 2 truncated. Both arms had zero transport errors. Truncated outputs remain in all scores.

No source-input overlap with earlier adapter experiments or previous iteration rounds. SOB image tasks use supplied OCR text rather than image requests. No fresh audio suite was used because the local source had only one unused transcript context.

Raw requests, responses, preparation code, source snapshots, manifests and detailed comparisons remain local. This report contains aggregates only. No model calls used JSON or Chat adapters.
