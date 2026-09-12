# Qwen 3.8: TOON 4.1 vs JSON, BAML-inspired and Chat

Completed 2026-09-05 against the user-provided vLLM endpoint. **800/800 main requests**, 100 fixed cases, 20 cases per workload × four adapters × native thinking off/on. The separate 40-call preflight is excluded from all comparison scores.

This is an exploratory, zero-shot adapter comparison on five workloads from three benchmark families. It is not an official leaderboard run. BAML here means the repository’s BAML-inspired DSPy adapter, not BoundaryML’s full runtime. SOB document/audio tasks use supplied OCR/transcript text.

Server-reported model: `Qwen/Qwen3.8-27B-FP8`; DSPy `3.3.1`; observed fingerprints: `vllm-0.27.1-ef403332`.

## Task scores

Scores are percentages; higher is better. SOB uses official schema-gated leaf-value exact match, TableBench uses its official normalized numerical answer match, and BBEH uses official correctness. Invalid outputs score zero. Compare adapters within a row; the score definitions differ between families.

### Native thinking off

| Workload | TOON | JSON | BAML-inspired | Chat |
|---|---:|---:|---:|---:|
| SOB text | 63.5 | 77.4 | 11.5 | 77.4 |
| SOB OCR text | 53.6 | 61.3 | 4.5 | 57.3 |
| SOB meeting transcripts | 17.4 | 23.8 | 1.8 | 23.7 |
| TableBench numerical | 40.0 | 50.0 | 35.0 | 40.0 |
| BBEH mini | 20.0 | 10.0 | 5.0 | 20.0 |

### Native thinking on

| Workload | TOON | JSON | BAML-inspired | Chat |
|---|---:|---:|---:|---:|
| SOB text | 75.8 | 79.5 | 74.4 | 69.3 |
| SOB OCR text | 57.8 | 58.3 | 57.0 | 56.5 |
| SOB meeting transcripts | 9.9 | 12.7 | 8.5 | 13.2 |
| TableBench numerical | 85.0 | 85.0 | 85.0 | 80.0 |
| BBEH mini | 30.0 | 30.0 | 35.0 | 30.0 |

## Output reliability and cost

Each row aggregates the same 100 cases. “Valid” includes parsing, typed validation and the original schema. Truncations may overlap invalid outputs. Completion tokens include native reasoning; the server did not expose a separate reasoning-token count. End-to-end latency was measured at concurrency 50; server traffic and prefix caching were not controlled.

| Thinking | Adapter | Valid / 100 | Truncated | Prompt tokens | Completion tokens | Mean latency | Median latency |
|---|---|---:|---:|---:|---:|---:|---:|
| off | toon | 86 | 2 | 539,473 | 36,420 | 6.8s | 1.5s |
| off | json | 97 | 2 | 554,116 | 39,706 | 6.8s | 1.8s |
| off | baml | 29 | 3 | 531,508 | 51,352 | 8.4s | 2.0s |
| off | chat | 91 | 7 | 551,471 | 84,080 | 13.6s | 2.2s |
| on | toon | 73 | 27 | 543,073 | 401,409 | 60.3s | 49.7s |
| on | json | 76 | 22 | 557,716 | 335,485 | 51.2s | 29.5s |
| on | baml | 73 | 23 | 535,108 | 331,759 | 51.5s | 25.1s |
| on | chat | 73 | 22 | 555,071 | 344,175 | 53.5s | 32.6s |

Main-run model usage: **5,991,922 tokens** over approximately **9.2 minutes** wall time. Separate preflight usage: 267,231 tokens. Two initial reasoning-toggle probes are also excluded. HTTP/transport failures: 0. Records without usage: 0.

### Reliability by workload

| Workload | Thinking | TOON valid / truncated | JSON valid / truncated | BAML valid / truncated | Chat valid / truncated |
|---|---|---:|---:|---:|---:|
| SOB text | off | 18/20 / 0 | 19/20 / 0 | 4/20 / 0 | 20/20 / 0 |
| SOB text | on | 20/20 / 0 | 20/20 / 0 | 19/20 / 0 | 17/20 / 0 |
| SOB OCR text | off | 17/20 / 0 | 20/20 / 0 | 2/20 / 0 | 19/20 / 0 |
| SOB OCR text | on | 18/20 / 2 | 19/20 / 0 | 19/20 / 0 | 20/20 / 0 |
| SOB meeting transcripts | off | 15/20 / 0 | 20/20 / 0 | 1/20 / 0 | 19/20 / 0 |
| SOB meeting transcripts | on | 7/20 / 13 | 8/20 / 11 | 7/20 / 11 | 10/20 / 8 |
| TableBench numerical | off | 19/20 / 0 | 20/20 / 0 | 16/20 / 0 | 20/20 / 0 |
| TableBench numerical | on | 19/20 / 1 | 20/20 / 0 | 19/20 / 1 | 18/20 / 2 |
| BBEH mini | off | 17/20 / 2 | 18/20 / 2 | 6/20 / 3 | 13/20 / 7 |
| BBEH mini | on | 9/20 / 11 | 9/20 / 11 | 9/20 / 11 | 8/20 / 12 |

## Paired uncertainty

TOON minus the comparator, in percentage points. Intervals are descriptive 95% paired percentile-bootstrap intervals (10,000 case resamples, seed 4201); they are not adjusted for multiple comparisons. With only 20 cases, especially binary outcomes, small or tied differences are weak evidence. Win/tie/loss counts refer to per-case primary score.

| Workload | Thinking | Comparator | Difference [95% CI], pp | Wins / ties / losses |
|---|---|---|---:|---:|
| SOB text | off | json | -13.9 [-28.3, -2.6] | 2 / 11 / 7 |
| SOB text | off | baml | +52.0 [+28.7, +73.0] | 13 / 6 / 1 |
| SOB text | off | chat | -14.0 [-28.8, -2.1] | 1 / 13 / 6 |
| SOB text | on | json | -3.7 [-13.2, +3.3] | 1 / 16 / 3 |
| SOB text | on | baml | +1.4 [-13.6, +16.1] | 3 / 15 / 2 |
| SOB text | on | chat | +6.6 [-11.6, +25.1] | 5 / 12 / 3 |
| SOB OCR text | off | json | -7.7 [-25.1, +6.6] | 5 / 10 / 5 |
| SOB OCR text | off | baml | +49.1 [+32.4, +65.2] | 15 / 5 / 0 |
| SOB OCR text | off | chat | -3.7 [-21.4, +11.7] | 7 / 7 / 6 |
| SOB OCR text | on | json | -0.5 [-10.4, +8.5] | 7 / 8 / 5 |
| SOB OCR text | on | baml | +0.8 [-4.9, +7.4] | 6 / 8 / 6 |
| SOB OCR text | on | chat | +1.3 [-9.0, +10.9] | 7 / 7 / 6 |
| SOB meeting transcripts | off | json | -6.4 [-13.5, +1.7] | 6 / 2 / 12 |
| SOB meeting transcripts | off | baml | +15.7 [+6.3, +25.9] | 14 / 5 / 1 |
| SOB meeting transcripts | off | chat | -6.2 [-13.9, +2.0] | 6 / 2 / 12 |
| SOB meeting transcripts | on | json | -2.9 [-10.6, +5.0] | 3 / 10 / 7 |
| SOB meeting transcripts | on | baml | +1.3 [-5.0, +7.7] | 5 / 12 / 3 |
| SOB meeting transcripts | on | chat | -3.3 [-10.3, +3.8] | 4 / 9 / 7 |
| TableBench numerical | off | json | -10.0 [-25.0, +0.0] | 0 / 18 / 2 |
| TableBench numerical | off | baml | +5.0 [+0.0, +15.0] | 1 / 19 / 0 |
| TableBench numerical | off | chat | +0.0 [-15.0, +15.0] | 1 / 18 / 1 |
| TableBench numerical | on | json | +0.0 [+0.0, +0.0] | 0 / 20 / 0 |
| TableBench numerical | on | baml | +0.0 [+0.0, +0.0] | 0 / 20 / 0 |
| TableBench numerical | on | chat | +5.0 [+0.0, +15.0] | 1 / 19 / 0 |
| BBEH mini | off | json | +10.0 [+0.0, +25.0] | 2 / 18 / 0 |
| BBEH mini | off | baml | +15.0 [+0.0, +30.0] | 3 / 17 / 0 |
| BBEH mini | off | chat | +0.0 [-15.0, +15.0] | 1 / 18 / 1 |
| BBEH mini | on | json | +0.0 [-15.0, +15.0] | 1 / 18 / 1 |
| BBEH mini | on | baml | -5.0 [-15.0, +0.0] | 0 / 19 / 1 |
| BBEH mini | on | chat | +0.0 [-15.0, +15.0] | 1 / 18 / 1 |

## Interpretation limits

- Same 8,192-token completion cap for both modes; reasoning competes with the final answer for this budget. Truncated thinking responses do not establish the model’s capability with a larger reasoning budget.
- Temperature 0.6, top_p 0.95, one sampled response per cell. Identical per-case seeds do not guarantee identical token sampling or determinism across adapter prompts and vLLM batching.
- SOB text/image selection is balanced by schema complexity; meeting transcripts are stratified by length quartile. These are deliberately varied samples, not estimates weighted to each full dataset’s distribution.
- All reference answers pass their original schemas and self-score correctly, but gold wording, answer ambiguity and literal escape artifacts can affect exact-match scores. No reference answers were edited.
- JSON and Chat are DSPy 3.3.1 adapters; BAML is the repository’s implementation. Calls use adapter formatting and parsing with a shared unconstrained transport, without native JSON-schema decoding, repair calls, or fallback model requests.
- The user requested an increase from four to 50 concurrent requests after 88 completed calls. The partial run is retained separately and excluded; its known usage was 703,723 tokens, plus unknown usage for four cancelled in-flight requests. The unchanged 800-case/request matrix was restarted at concurrency 50. No cases or prompts were selected or changed based on scores.
- No TOON 3.x model arm was run. These results compare upgraded TOON against other adapters; they do not causally establish a quality gain from TOON 4.1 itself.

## Sources and artifacts

Sources: [SOB, 2026](https://arxiv.org/abs/2604.25359), [SOB repository and scorer](https://github.com/InterfazeAI/sob), [TableBench](https://github.com/TableBench/TableBench), [BBEH](https://github.com/google-deepmind/bbeh). TableBench and BBEH are 2025 releases; they were selected alongside SOB for table and hard-reasoning coverage, not represented as new 2026 releases.

- [Protocol and reproduction notes](protocol.md)
- [Per-cell metrics](metrics.csv) and [summary JSON](summary.json)
- [Paired comparisons](paired_comparisons.json)
- [Scored cases](scored.jsonl), [raw model responses](responses.jsonl), [frozen requests](requests.jsonl), [selected cases and golds](cases.jsonl)
- [Manifest](manifest.json), [scorer provenance](scorer_sources.json), and source_snapshot/ preserve the exact run inputs and implementation.
