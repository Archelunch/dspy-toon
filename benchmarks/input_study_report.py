"""Build a readable report from completed frozen study artifacts."""

import csv
import json
from pathlib import Path

OUT = Path("benchmark_results/qwen_input_study")


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    pairs = json.loads((OUT / "paired.json").read_text())
    rows = [json.loads(s) for s in (OUT / "scored.jsonl").read_text().splitlines()]
    lookup = {(r["track"], r["slice"], r["thinking"], r["adapter"]): r for r in metrics}
    lines = [
        "# TOON input comprehension: upstream reproduction and tool-result study",
        "",
        "Completed **4,082 generation requests** on the user-provided Qwen/Qwen3.8-27B-FP8 deployment. "
        "This is a separate experiment from the earlier DSPy output-generation comparison. "
        "The two tracks measure reading serialized data with fixed output requirements.",
        "",
        "Upstream: **244 questions sharing 13 source datasets**, 1,682 calls. "
        "Tools: **500 independent synthetic datasets**, 2,000 primary calls plus 400 repeated-seed calls "
        "on 100 selected datasets. "
        "Repeated calls are not additional independent cases.",
        "",
        "Read [the interpretation and parser audit](INTERPRETATION.md) before using a winner claim. "
        "It explains the fence effect, upstream wrapper sensitivity, and a repeat-subset selection limitation.",
        "",
        "## Upstream reproduction",
        "",
        "Pinned [source revision](https://github.com/toon-format/toon/tree/f151a5d830d001bc244395b891183cba37e0d935). "
        "Unchanged upstream question/data generators, reference encoder, evaluation prompt and answer normalizer. "
        "Qwen settings differ from upstream provider defaults: temperature 0.6, top_p 0.95, 32K output "
        "cap, thinking off/on. "
        "No YAML/XML arms. No claim of reproducing the original multi-model averages.",
        "",
        "Percent correct; **thinking off / on**. Ordinary comprehension excludes structure-awareness and "
        "validation questions.",
        "",
        "| Population | Questions per format | Pretty JSON | Compact JSON | TOON |",
        "|---|---:|---:|---:|---:|",
    ]
    for subset, label in [("all", "Full upstream suite"), ("ordinary", "Ordinary comprehension")]:
        vals = []
        for arm in ["json-pretty", "json-compact", "toon"]:
            vals.append(" / ".join(f"{100 * lookup['upstream', subset, t, arm]['score']:.1f}" for t in [False, True]))
        n = lookup["upstream", subset, False, "toon"]["n"]
        lines.append(f"| {label} | {n} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "![Upstream accuracy](figures/01_upstream_accuracy.png)",
        "",
        "### Matched flat subset (CSV is supported)",
        "",
        "| Format | Questions per mode | Score off / on | Prompt tokens off / on |",
        "|---|---:|---:|---:|",
    ]
    for arm in ["json-pretty", "json-compact", "toon", "csv"]:
        a, b = [lookup["upstream", "flat", t, arm] for t in [False, True]]
        lines.append(
            f"| {arm} | {a['n']} | {100 * a['score']:.1f} / {100 * b['score']:.1f} | "
            f"{a['prompt_tokens']:,} / {b['prompt_tokens']:,} |"
        )
    lines += [
        "",
        "### Upstream question categories",
        "",
        "| Category | n per mode | Compact JSON off / on | TOON off / on |",
        "|---|---:|---:|---:|",
    ]
    for cat in ["field-retrieval", "aggregation", "filtering", "structure-awareness", "structural-validation"]:
        a = lookup["upstream", "category:" + cat, False, "toon"]
        vals = [
            " / ".join(f"{100 * lookup['upstream', 'category:' + cat, t, arm]['score']:.1f}" for t in [False, True])
            for arm in ["json-compact", "toon"]
        ]
        lines.append(f"| {cat} | {a['n']} | " + " | ".join(vals) + " |")
    lines += [
        "",
        "**Validation is only five questions per mode.** Upstream post-encode corruption preserves TOON’s "
        "expected length/width metadata. "
        "JSON does not receive matching expected-count metadata, so this track measures that built-in "
        "information advantage as well as format comprehension. "
        "Do not pool it into a broad accuracy claim without showing the ordinary-QA result.",
        "",
        "### Paired differences on ordinary comprehension",
        "",
        "TOON minus baseline. Dataset-cluster bootstrap resamples the source datasets; question bootstrap "
        "treats the fixed catalog’s questions as units. "
        "Few source datasets limit generalization. Intervals are exploratory 95% intervals from 10,000 "
        "resamples, without multiplicity correction.",
        "",
        "| Thinking | Baseline | Matched n | Gap, pp [question CI] | Source-cluster CI | Prompt change | "
        "Total-token change |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for t in [False, True]:
        for arm in ["json-compact", "json-pretty", "csv"]:
            a = next(
                r
                for r in pairs
                if r["track"] == "upstream"
                and r["slice"] == "ordinary"
                and r["thinking"] == t
                and r["control"] == arm
                and not r["clustered"]
            )
            b = next(
                r
                for r in pairs
                if r["track"] == "upstream"
                and r["slice"] == "ordinary"
                and r["thinking"] == t
                and r["control"] == arm
                and r["clustered"]
            )
            lines.append(
                f"| {'on' if t else 'off'} | {arm} | {a['n']} | {100 * a['score_delta']:+.1f} "
                f"[{100 * a['ci_low']:+.1f}, {100 * a['ci_high']:+.1f}] | [{100 * b['ci_low']:+.1f}, "
                f"{100 * b['ci_high']:+.1f}] | {100 * (a['prompt_token_ratio'] - 1):+.1f}% | "
                f"{100 * (a['total_token_ratio'] - 1):+.1f}% |"
            )
    lines += [
        "",
        "## Tool-result input comparison",
        "",
        "Inventory, nested customer records, keyed service maps, semi-uniform logs, and orders containing "
        "nested item arrays. "
        "Both inputs request the same JSON result output. These are controlled synthetic "
        "application-shaped records, not real production API traffic. "
        "100 independent datasets per family; 25 at each of 10/30/100/300 rows. Five balanced query types. "
        "Gold answers are computed directly from source data. All 500 TOON encodings are byte-identical "
        "to the pinned reference implementation.",
        "",
        "Primary seed only. Semantic correctness requires standard JSON parsing and a result scalar "
        "matching the gold after stripped case-insensitive string comparison. "
        "Numeric scalars may earn semantic credit; strict validity additionally requires exactly one "
        "result key with a string value. "
        "Outer-fence removal is reported separately as a sensitivity metric.",
        "",
        "| Family | n per mode | Compact JSON score off / on | TOON score off / on |",
        "|---|---:|---:|---:|",
    ]
    for family in ["inventory", "customers", "services", "logs", "orders"]:
        vals = [
            " / ".join(f"{100 * lookup['tools', 'dataset:' + family, t, arm]['score']:.1f}" for t in [False, True])
            for arm in ["json-compact", "toon"]
        ]
        lines.append(f"| {family} | 100 | " + " | ".join(vals) + " |")
    lines += [
        "",
        "![Tool accuracy](figures/02_tool_accuracy.png)",
        "",
        "| Thinking | Family | TOON − JSON, pp [95% CI] | Prompt change | Total-token change [95% CI] |",
        "|---|---|---:|---:|---:|",
    ]
    for t in [False, True]:
        for family in ["all", "inventory", "customers", "services", "logs", "orders"]:
            a = next(r for r in pairs if r["track"] == "tools" and r["slice"] == family and r["thinking"] == t)
            lines.append(
                f"| {'on' if t else 'off'} | {family} | {100 * a['score_delta']:+.1f} "
                f"[{100 * a['ci_low']:+.1f}, {100 * a['ci_high']:+.1f}] | "
                f"{100 * (a['prompt_token_ratio'] - 1):+.1f}% | {100 * (a['total_token_ratio'] - 1):+.1f}% "
                f"[{100 * (a['token_ci_low'] - 1):+.1f}, {100 * (a['token_ci_high'] - 1):+.1f}] |"
            )
    lines += [
        "",
        "![Tool tokens](figures/03_tool_tokens.png)",
        "",
        "![Size and accuracy](figures/04_size_accuracy.png)",
        "",
        "### Semantic accuracy and output contract",
        "",
        "| Thinking | Input format | n | Semantic score | Strict JSON outputs | Score after optional "
        "fence removal | Truncated |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for t in [False, True]:
        for arm in ["json-compact", "toon"]:
            r = lookup["tools", "all", t, arm]
            lines.append(
                f"| {'on' if t else 'off'} | {arm} | {r['n']} | {100 * r['score']:.1f}% | "
                f"{r['strict_valid']} | {100 * r['fence_score']:.1f}% | {r['truncated']} |"
            )
    lines += [
        "",
        "### Seed sensitivity on the predefined 100-dataset subset",
        "",
        "| Thinking | Input | Seed 1 accuracy | Seed 2 accuracy | Correctness flips / 100 |",
        "|---|---|---:|---:|---:|",
    ]
    for r in csv.DictReader((OUT / "seed_sensitivity.csv").open()):
        lines.append(
            f"| {'on' if r['thinking'] == 'True' else 'off'} | {r['adapter']} | "
            f"{100 * float(r['first_score']):.1f}% | {100 * float(r['second_score']):.1f}% | "
            f"{r['correctness_flips']} |"
        )
    lines += [
        "",
        "The repeated subset is not pooled into the primary 500-case score. Correctness flips measure "
        "sensitivity to the second seed and uncontrolled batching; they do not identify its internal cause.",
        "",
        "## Accounting and limits",
        "",
        f"- Completed calls: {len(rows):,}; transport errors: {sum(r['transport_error'] for r in rows)}; "
        f"token-limit finishes: {sum(r['truncated'] for r in rows)}.",
        "- Reported total tokens across all calls, including repeat seeds and failures: "
        f"{sum(r['total_tokens'] for r in rows):,}.",
        "- Completion tokens include reasoning; provider reasoning-token breakdown is unavailable. Counts "
        "include format primers and other instructions.",
        "- Latency is end-to-end at concurrency 50, with uncontrolled server load and caching. It is not "
        "an isolated format speed comparison.",
        "- One server-reported model deployment; checkpoint files were not inspected. Public-source "
        "training contamination is not ruled out.",
        "- The synthetic generator varies records but uses a small fixed set of question templates. Its "
        "independent datasets do not establish coverage of arbitrary business workflows.",
        "- This is input serialization research. It does not establish the reliability of generating TOON "
        "output, or a causal quality gain from the TOON 4.1 upgrade.",
        "",
        "## Reproduce and inspect",
        "",
        "- [Frozen protocol](PROTOCOL.md), [manifest](manifest.json), [encoder "
        "verification](encoder_verification.json), [analysis freeze](analysis_freeze.json).",
        "- [Metrics CSV](metrics.csv), [paired comparisons](paired.csv), [seed "
        "sensitivity](seed_sensitivity.csv), [scored traces](scored.jsonl).",
        "- [Requests](requests.jsonl), [raw responses](responses.jsonl), [upstream "
        "cases](upstream_cases.json), [tool cases](tool_cases.json).",
        "- Upstream source is preserved in upstream_source.tar; the locked dependency graph is inside "
        "that archive. Five figures are available as editable SVG and 220-DPI PNG.",
        "",
        "```sh",
        ".venv/bin/python -m benchmarks.input_study_analysis",
        ".venv/bin/python -m benchmarks.input_study_report",
        "uv run --no-project --with matplotlib==3.10.7 python benchmarks/input_study_plots.py",
        "```",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("Wrote REPORT.md")


if __name__ == "__main__":
    main()
