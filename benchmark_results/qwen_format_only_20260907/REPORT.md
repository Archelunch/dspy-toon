# Format-only instruction rerun

400 TOON-only requests on Qwen/Qwen3.8-27B-FP8: 40 cases in each of five suites, with reasoning disabled and enabled. Cases were selected with seed 20260907 from the archived expanded comparison without consulting outcomes. No other adapters were rerun.

The new prompts remove the format name and version from adapter instructions and replace generic adapter references in benchmark task instructions with structure guidance. Syntax rules, schemas, source inputs, per-case generation seed, temperature (0.6), top_p (0.95) and output limit (32768) are unchanged. The codec and scorer are unchanged.

## Paired results

| Task | Reasoning | n | Score before / after | Parse before / after | Mean total tokens before / after |
|---|---|---:|---:|---:|---:|
| bbeh_mini | off | 40 | 12.5% / 17.5% | 80.0% / 62.5% | 3076 / 5740 |
| bbeh_mini | on | 40 | 55.0% / 55.0% | 90.0% / 85.0% | 14566 / 15271 |
| sob_audio | off | 40 | 14.9% / 18.5% | 57.5% / 70.0% | 20084 / 20104 |
| sob_audio | on | 40 | 20.0% / 19.9% | 87.5% / 82.5% | 30605 / 31113 |
| sob_image | off | 40 | 46.5% / 50.5% | 82.5% / 87.5% | 1994 / 2006 |
| sob_image | on | 40 | 52.0% / 53.0% | 87.5% / 82.5% | 7338 / 7929 |
| sob_text | off | 40 | 83.8% / 84.1% | 100.0% / 100.0% | 2047 / 2045 |
| sob_text | on | 40 | 82.2% / 83.6% | 97.5% / 97.5% | 3490 / 3229 |
| tablebench_numeric | off | 40 | 22.5% / 25.0% | 95.0% / 97.5% | 1017 / 1011 |
| tablebench_numeric | on | 40 | 87.5% / 80.0% | 100.0% / 95.0% | 1957 / 2040 |
| ALL | off | 200 | 36.0% / 39.1% | 83.0% / 83.5% | 5644 / 6181 |
| ALL | on | 200 | 59.3% / 58.3% | 92.5% / 88.5% | 11591 / 11917 |

Scores use the original benchmark metrics: leaf-value exact match for structured extraction, task correctness for BBEH, and the archived TableBench exact-match scorer. ALL is an equal-suite descriptive average of these different metrics, not leaderboard accuracy.

## Interpretation and trace review

Across 200 requests per reasoning setting, parsing changed from 83.0% to 83.5%
with reasoning off (166 to 167 accepted), and from 92.5% to 88.5% with reasoning
on (185 to 177 accepted). There were no transport errors. Seven responses
reached the output-token limit, compared with three in the matched older sample.

Mean total tokens increased 9.5% without reasoning and 2.8% with reasoning.
Prompt length itself fell by only four tokens on TableBench and five tokens on
the other suites. Longer generated outputs dominate the cost difference.

Representative changed traces showed unquoted numeric scalar answers where the
signature requires strings, row-width errors from quoting an entire row, invalid
backslash escapes in mathematical text, and nulls rejected by non-nullable
schemas. One TableBench answer was `result: 4.576`; decoding succeeded, but the
required string type rejected the number. Six of the seven truncated responses
used reasoning, and the remaining one was a BBEH call with reasoning disabled.

Each per-suite score-difference 95% paired bootstrap interval includes zero
(or touches zero). The intervals use 10,000 resamples within each suite and
reasoning setting. The sample supports neither a general quality improvement
nor equivalence. Removing format names makes the instructions self-contained;
this run does not establish a reliability or total-token benefit.

Both runs report model ID Qwen/Qwen3.8-27B-FP8 and server fingerprint
`vllm-0.27.1-ef403332`. These identifiers do not prove that every backend setting
was identical across dates.

## Limits

This is a historical paired comparison, not a simultaneous randomized control. The endpoint reports the same model ID, but backend changes and stochastic generation can affect differences even with matching seeds. These cases were already used in the earlier study; they are not fresh held-out evidence. SOB image/audio suites use textual OCR/transcript inputs, not native multimodal requests.

Token counts come directly from vLLM response usage and include generated reasoning. Latency is not compared across runs. Raw request/response archives, changed-case diagnostics and numeric bootstrap estimates remain local; no raw payload publication is implied.
