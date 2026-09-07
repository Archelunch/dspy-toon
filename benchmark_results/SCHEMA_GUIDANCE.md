# Schema-guided prompt experiments

Three paired iterations, 1,080 Qwen requests across 270 unique cases. Each round
compares its candidate to the same original name-free adapter, with matched
inputs, schemas, generation seeds and reasoning settings. The old adapter is
rerun concurrently rather than reused from a historical inference run. No JSON
or Chat adapters were rerun; no parser repairs, scalar coercion or retry fallback
were introduced.

## Outcome

The selected candidate is iteration 3. It retains missing nested constraints,
uses field-specific reminders, selects syntax rules from the signature, and
explicitly quotes string items/cells to prevent embedded commas from splitting
values. Its results are promising for structural reliability, with remaining
quality tradeoffs on table questions.

- [Iteration 1](qwen_schema_guidance_20260907/REPORT.md): parsing 161/200 to 175/200; total tokens +2.9%.
- [Iteration 2](qwen_schema_guidance_v2_20260907/REPORT.md): parsing 173/200 to 183/200; total tokens +1.7%.
- [Iteration 3](qwen_schema_guidance_v3_20260907/REPORT.md): parsing 109/140 to 128/140; total tokens -7.7%.

On iteration 3's external tasks alone, parsing rose from
84/100 to 90/100; total tokens changed
-6.5%. This separates the external tasks from the large synthetic
constraint benefit.

## What the iterations showed

The first candidate added too much repeated schema text. OCR prompts grew by
about 325 tokens and its extraction score declined. The second candidate
removed those repetitions, but OCR still regressed: the non-reasoning score
difference was -11.7 percentage points, with an unadjusted paired 95% interval
of [-23.6, -2.1].

Trace inspection found three newly failing OCR cases where unquoted commas
inside strings created extra primitive-array items or table columns. The third
candidate explicitly names commas and requires quotes around string items and
cells. On its fresh OCR sample, parsing improved from 65% to 75% without
reasoning and 80% to 85% with reasoning. The corresponding task-score intervals
still include zero, so the evidence for OCR answer-quality improvement is weak.

The strongest repeated gain was in the synthetic nested/keyed constraint group
with reasoning. Exact decoded payload equality improved from 4/20 to 20/20 in
round 1, from 10/20 to 19/20 in round 2, and from 10/20 to 20/20 in round 3.
These are three different generated case sets from the same small set of data
patterns, not evidence covering all application data.

Iteration 3's original TableBench score fell from 8/10 to 6/10 without reasoning
and 10/10 to 9/10 with reasoning. A subsequent trace audit found that two of the
three newly failing answers differed only in capitalization of the correct
entity name. Keeping the original scores and separately applying case-insensitive
scoring to both arms gives 8/10 to 7/10 without reasoning and 10/10 to 10/10 with
reasoning. All table responses parsed successfully. The remaining new error was
an incorrect locks-per-mile comparison without reasoning, not malformed data.
See the iteration-3 report for the audit and sensitivity method. Text extraction
was essentially unchanged and BBEH reasoning accuracy was unchanged. These small
samples do not establish a universal accuracy improvement.

## Scope and limits

Iterations were selected adaptively using preceding results. The final round
used new cases but only 10 cases each for text, tables and BBEH, and 20 each for
OCR and synthetic constraints. Do not pool the three candidate versions or
select the best score from each round as if they described one adapter. Further
confirmation should use a larger predefined table/OCR sample and additional
model seeds before claiming a general quality gain.

All inference used the same model ID and a concurrent baseline. Matching seeds
does not make sampled generation deterministic. Token counts are vLLM response
usage, including reasoning; all failures and truncations are retained. Aggregate
token differences depend on this workload mix and can be dominated by long BBEH
reasoning calls.

The final code also renders nullable alternatives inside nested arrays. An
offline check verified that this edge-case correction leaves every saved
iteration-3 request unchanged. The parser AST is identical to the baseline.
No additional test files or checker tools were added; the existing 71 tests pass.
