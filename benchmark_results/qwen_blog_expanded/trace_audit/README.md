# Payload and trace audit

This follow-up inspects archived calls; no new inference was run. Earlier analysis covered all responses automatically (parsing, original-schema validity, scoring, truncation and usage) plus format-recovery inspection. It was not a comprehensive human semantic review of every answer.

## What was sent

Each main request has a system message and a user message, model Qwen/Qwen3.8-27B-FP8, temperature 0.6, top_p 0.95, max_tokens 32768, a case-specific seed and chat_template_kwargs.enable_thinking. No response_format was sent in the main comparison. The separate native experiment adds JSON Schema.

Actual DSPy JSON uses DSPy field markers in the user message and requests a result field in JSON. TOON encodes input fields as TOON and supplies a general TOON 4.1 primer, output type/field descriptions and output shape examples. Both use the same task instructions. This compares complete adapter prompt/parser contracts, not punctuation alone. The harness calls adapter.format, sends the saved payload directly to vLLM, and later invokes adapter.parse; it is not a full DSPy LM call with automatic provider negotiation or fallback.

The normalized factorial holds output instructions fixed when changing input serialization. See case_299_json_json_request.json and case_299_toon_json_request.json. Stock-adapter examples are case_299_json_request.json and case_299_toon_request.json. All exported JSON files contain the full actual payload, without omissions.

Only 150/544 cases have structured table inputs. SOB supplies prose/OCR/transcript strings, and BBEH supplies a question string. These cannot exercise repeated-field input compression in the same way as record arrays. TableBench covers numerical reasoning; it is not a broad retrieval/lookup benchmark.

## All 25 table score disagreements, thinking off

There are 20 cases with a higher official JSON score and five with a higher TOON score. The full paired predictions, gold, input table and raw responses are in table_disagreements.json. This is a targeted review of these disagreements, not every output in the study.

Of the 20 JSON-higher cases:

- Four TOON responses contain the gold numeric value but violate the required string type: result: 2005, result: 5, result: 1251, result: 10.12. These are type-contract failures, not incorrect calculations.
- Two TOON responses contain the requested gold content but use an explanation or “and” instead of the requested answer presentation.
- Fourteen have different or incomplete answer content relative to gold. These were not all independently recomputed from the source tables.

Among the five TOON-higher cases, one JSON answer differs only in capitalization (australia versus Australia), and one TOON answer receives partial credit for supplying one of two gold names.

Concrete wrong-answer example: case 299 asks for primary schools with dcsf number below 2200. Boxmoor and South Hill each contribute 30, so the gold/JSON answer is 60; TOON returns 105. This verifies the final-answer mismatch but does not establish the model's internal cause.

Other inspected traces include an inline array declaring three elements but emitting one quoted string, and a two-element array whose entries are emitted on separate lines without list markers. These are output-generation failures. Earlier recovery audit found 80/88 actual-JSON recovered outputs needed only fence removal.

The published score remains the official contract-based score. These observations justify a separately labeled semantic/format sensitivity analysis; they do not silently replace the original metric or establish a revised aggregate accuracy.

## Why upstream results answer a different question

[Upstream methodology](https://github.com/toon-format/toon/blob/main/benchmarks/README.md) explicitly measures reading formatted data, not generating TOON. [The evaluation prompt](https://github.com/toon-format/toon/blob/main/benchmarks/src/evaluate.ts) requests only a short value answer, using a format primer and fenced input.

The current [upstream results](https://github.com/toon-format/toon#benchmarks) report 244 questions across 13 datasets and four models; 42.6% fewer tokens compares TOON with pretty JSON. Its average token figures of 2474 versus compact JSON's 2892 imply roughly 14.5% fewer, rather than 42.6%. These are upstream measurements, not this Qwen run. Repeated questions on the same dataset and repeated models do not create independent datasets.

The upstream corruption track preserves TOON's declared length after removing rows. This legitimately measures the value of length metadata, but it is not a pure equal-information syntax comparison. A follow-up should report both native formats and matched expected-count/schema metadata.

## Proposed next experiments

1. Reproduce all 244 upstream questions on this Qwen deployment, preserving short-answer output; JSON pretty, compact JSON, TOON, and CSV only on the flat subset. Pin source revision and compare against the reference encoder.
2. Add 500 independently generated record datasets spanning row/column counts, flat records, nested uniform records, keyed maps and irregular objects. Deterministic lookup/filter/aggregation gold; fixed simple output. Synthetic and application-shaped, not an external leaderboard.
3. Add real-table comprehension: TableBench fact checking and data analysis, plus WikiTableQuestions. Split by table, report task types separately, retain official scoring and add a preregistered answer-normalization sensitivity metric.
4. Test realistic database/API tool results: records, logs, inventory and nested orders with fixed JSON final output. Measure correctness and total tokens at equal data and separately at equal input budgets.
5. Test context pressure with independently generated tables at increasing sizes, exact retrieval questions and fixed context/output reserves. A curated LongBench v2 structured-data subset can complement this only where its original data can be losslessly reserialized.

Use case/table-level paired intervals, independently resampled source datasets, multiple generation seeds on a predefined subset, matched output contracts, and separate input-token and total-token metrics. Do not select only TOON-winning shapes or pool corruption detection with ordinary QA.
