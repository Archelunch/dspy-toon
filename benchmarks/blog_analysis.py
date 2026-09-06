"""Export paired estimates and raw-format diagnostics for the expanded blog study."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from benchmarks import blog_comparison as study
from dspy_toon import decode

core = study.core


def reject_constant(value: str) -> None:
    raise ValueError(f"Non-JSON constant: {value}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if rows:
        with path.open("w") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def diagnostics(folder: Path) -> list[dict[str, Any]]:
    cases = core.read_jsonl(folder / "cases.jsonl")
    raw = {r["job_id"]: r for r in core.read_jsonl(folder / "responses.jsonl")}
    rows = core.read_jsonl(folder / "scored.jsonl")
    for row in rows:
        case = cases[row["case_index"]]
        response = raw[row["job_id"]].get("response", {})
        content = (response.get("choices") or [{}])[0].get("message", {}).get("content") or ""
        output_format = row["adapter"].split("_")[-1]
        row.update(raw_syntax_valid=False, raw_schema_valid=False, raw_error="")
        try:
            value = json.loads(content, parse_constant=reject_constant) if output_format == "json" else decode(content)
            row["raw_syntax_valid"] = True
            if not isinstance(value, dict) or set(value) != {"result"}:
                raise ValueError("Expected exactly one top-level result field")
            core.Draft7Validator(case["schema"]).validate(value["result"])
            row["raw_schema_valid"] = True
        except Exception as error:
            row["raw_error"] = str(error).splitlines()[0][:300]
        row["parser_recovery_or_coercion"] = row["schema_valid"] and not row["raw_schema_valid"]
        row["input_chars"] = len(json.dumps(case["inputs"], ensure_ascii=False))
        table = case["inputs"].get("table", [])
        row["table_rows"] = len(table)
        row["table_columns"] = len(table[0]) if table else 0
        if row.get("error"):
            category = "transport"
        elif row["finish_reason"] == "length":
            category = "token_limit"
        elif row["schema_valid"]:
            category = "accepted"
        elif "Expected " in row["raw_error"] and ("values" in row["raw_error"] or "rows" in row["raw_error"]):
            category = "array_count_or_width"
        elif row["raw_syntax_valid"]:
            category = "schema_or_wrapper"
        else:
            category = "other_syntax"
        row["failure_category"] = category
    core.save_jsonl(folder / "diagnostics.jsonl", rows)
    return rows


def summarize(rows: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    groups = defaultdict(list)
    for row in rows:
        groups[(row["suite"], row["adapter"], row["thinking"])].append(row)
    summary = []
    for (suite, arm, thinking), group in sorted(groups.items()):
        scores = np.array([r["score"] for r in group])
        samples = rng.integers(0, len(group), (10000, len(group)))
        ci = np.quantile(scores[samples].mean(axis=1), [0.025, 0.975])
        usage = [r["usage"] for r in group]
        total = sum(u.get("total_tokens", 0) for u in usage)
        exact = sum(r["exact_match"] for r in group)
        summary.append(
            {
                "suite": suite,
                "arm": arm,
                "thinking": thinking,
                "n": len(group),
                "score": float(scores.mean()),
                "score_ci_low": float(ci[0]),
                "score_ci_high": float(ci[1]),
                "exact_correct": exact,
                "schema_valid": sum(r["schema_valid"] for r in group),
                "raw_syntax_valid": sum(r["raw_syntax_valid"] for r in group),
                "raw_schema_valid": sum(r["raw_schema_valid"] for r in group),
                "recovery_or_coercion": sum(r["parser_recovery_or_coercion"] for r in group),
                "truncated": sum(r["finish_reason"] == "length" for r in group),
                "http_errors": sum(bool(r.get("error")) for r in group),
                "unknown_usage": sum(not u for u in usage),
                "prompt_tokens": sum(u.get("prompt_tokens", 0) for u in usage),
                "completion_tokens": sum(u.get("completion_tokens", 0) for u in usage),
                "total_tokens": total,
                "mean_total_tokens": total / len(group),
                "tokens_per_exact_correct": total / exact if exact else None,
                "mean_latency_s": float(np.mean([r["latency_s"] for r in group])),
                "p50_latency_s": float(np.median([r["latency_s"] for r in group])),
                "p95_latency_s": float(np.quantile([r["latency_s"] for r in group], 0.95)),
                "token_f1": float(np.mean([r.get("token_f1", 0) for r in group])) if suite.startswith("sob_") else None,
            }
        )
    return summary


def contrasts(rows: list[dict[str, Any]], rng: np.random.Generator) -> list[dict[str, Any]]:
    indexed = defaultdict(dict)
    for row in rows:
        indexed[(row["suite"], row["adapter"], row["thinking"])][row["case_id"]] = row
    output = []
    for suite in study.COUNTS:
        for thinking in [False, True]:
            comparisons = [("actual_adapter", "toon", "json")]
            if suite in study.ABLATION_SUITES:
                comparisons += [
                    ("input_effect_json_output", "toon_json", "json_json"),
                    ("input_effect_toon_output", "toon_toon", "json_toon"),
                    ("output_effect_json_input", "json_toon", "json_json"),
                    ("output_effect_toon_input", "toon_toon", "toon_json"),
                ]
            for label, arm_a, arm_b in comparisons:
                a, b = indexed[suite, arm_a, thinking], indexed[suite, arm_b, thinking]
                if a.keys() != b.keys():
                    raise ValueError(f"Incomplete pairs: {suite} {arm_a} {arm_b}")
                keys = sorted(a)
                sample = rng.integers(0, len(keys), (10000, len(keys)))
                diff = np.array([a[k]["score"] - b[k]["score"] for k in keys])
                validity = np.array([float(a[k]["schema_valid"]) - float(b[k]["schema_valid"]) for k in keys])
                token_a = np.array([a[k]["usage"].get("total_tokens", 0) for k in keys])
                token_b = np.array([b[k]["usage"].get("total_tokens", 0) for k in keys])
                boot = diff[sample].mean(axis=1)
                ci = np.quantile(boot, [0.025, 0.975, 0.0025, 0.9975])
                ratios = token_a[sample].sum(axis=1) / token_b[sample].sum(axis=1)
                ratio_ci = np.quantile(ratios, [0.025, 0.975])
                valid_ci = np.quantile(validity[sample].mean(axis=1), [0.025, 0.975])
                output.append(
                    {
                        "suite": suite,
                        "thinking": thinking,
                        "comparison": label,
                        "arm_a": arm_a,
                        "arm_b": arm_b,
                        "n": len(keys),
                        "score_delta": float(diff.mean()),
                        "ci95_low": float(ci[0]),
                        "ci95_high": float(ci[1]),
                        "ci995_low": float(ci[2]),
                        "ci995_high": float(ci[3]),
                        "wins": int((diff > 1e-9).sum()),
                        "ties": int((np.abs(diff) <= 1e-9).sum()),
                        "losses": int((diff < -1e-9).sum()),
                        "validity_delta": float(validity.mean()),
                        "validity_ci_low": float(valid_ci[0]),
                        "validity_ci_high": float(valid_ci[1]),
                        "total_token_ratio": float(token_a.sum() / token_b.sum()),
                        "token_ratio_ci_low": float(ratio_ci[0]),
                        "token_ratio_ci_high": float(ratio_ci[1]),
                    }
                )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/qwen_blog_expanded"))
    args = parser.parse_args()
    manifest = json.loads((args.output / "manifest.json").read_text())
    raw = core.read_jsonl(args.output / "responses.jsonl")
    if len(raw) != manifest["planned_calls"] or len({r["job_id"] for r in raw}) != len(raw):
        raise SystemExit(f"Full run incomplete: {len(raw)}/{manifest['planned_calls']}")
    rows = diagnostics(args.output)
    if len(rows) != len(raw):
        raise SystemExit("Run the scoring command for all responses first")
    rng = np.random.default_rng(20260906)
    summary = summarize(rows, rng)
    pairs = contrasts(rows, rng)
    for name, values in [("metrics", summary), ("paired", pairs)]:
        (args.output / f"{name}.json").write_text(json.dumps(values, indent=2))
        write_csv(args.output / f"{name}.csv", values)
    print(
        json.dumps(
            {
                "rows": len(rows),
                "cells": len(summary),
                "paired_contrasts": len(pairs),
                "failures": dict(Counter(r["failure_category"] for r in rows)),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
