"""Larger disjoint-case DSPy comparison and input/output serialization ablation."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from benchmarks import qwen_adapter_comparison as core
from dspy_toon import encode

SEED = 20260906
COUNTS = {"sob_text": 100, "sob_image": 100, "sob_audio": 94, "tablebench_numeric": 150, "bbeh_mini": 100}
ABLATION_SUITES = {"sob_image", "tablebench_numeric"}


def collect_cases() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old = core.read_jsonl(Path("benchmark_results/qwen_20260905/cases.jsonl"))
    templates = {c["suite"]: c for c in old}
    rng = random.Random(SEED)
    cases, exclusions = [], []
    for modality, file in [("text", "data_test"), ("image", "image_train"), ("audio", "audio_train")]:
        suite = "sob_" + modality
        used_contexts = {c["inputs"]["context"] for c in old if c["suite"] == suite}
        rows = core.pq.read_table(f"/private/tmp/dspy-bench-hf/sob_{file}-00000-of-00001.parquet").to_pylist()
        rng.shuffle(rows)
        seen_ids, seen_contexts = set(), set(used_contexts)
        selected = []
        for row in rows:
            if row["context"] in seen_contexts or row["record_id"] in seen_ids:
                continue
            case = {
                "suite": suite,
                "id": row["record_id"],
                "modality": modality,
                "inputs": {"context": row["context"], "question": row["question"]},
                "schema": json.loads(row["json_schema"]),
                "gold": json.loads(row["ground_truth"]),
                "schema_complexity": row["schema_complexity"],
                "instructions": templates[suite]["instructions"],
            }
            try:
                core.Draft7Validator.check_schema(case["schema"])
                core.Draft7Validator(case["schema"]).validate(case["gold"])
                sig = core.signature_for(case)
                core.TypeAdapter(sig.output_fields["result"].annotation).validate_python(case["gold"])
            except Exception as error:
                exclusions.append({"suite": suite, "id": case["id"], "reason": str(error)})
                continue
            selected.append(case)
            seen_ids.add(row["record_id"])
            seen_contexts.add(row["context"])
            if len(selected) == COUNTS[suite]:
                break
        cases.extend(selected)
    rows = core.read_jsonl(Path("/private/tmp/dspy-bench-hf/TableBench_TableBench.jsonl"))
    prior_tables = {json.dumps(c["inputs"]["table"], sort_keys=True) for c in old if c["suite"] == "tablebench_numeric"}
    rng.shuffle(rows)
    selected = []
    for row in rows:
        if row["qtype"] != "NumericalReasoning":
            continue
        table = row["table"]
        if isinstance(table, str):
            table = json.loads(table)
        columns = table["columns"]
        if len(set(columns)) != len(columns) or any(len(r) != len(columns) for r in table["data"]):
            exclusions.append({"suite": "tablebench_numeric", "id": row["id"], "reason": "Ambiguous table columns"})
            continue
        values = [dict(zip(columns, r, strict=True)) for r in table["data"]]
        key = json.dumps(values, sort_keys=True)
        if key in prior_tables:
            continue
        prior_tables.add(key)
        selected.append(
            {
                "suite": "tablebench_numeric",
                "id": row["id"],
                "inputs": {"table": values, "question": row["question"]},
                "schema": {"type": "string"},
                "gold": row["answer"],
                "subtype": row["qsubtype"],
                "instructions": templates["tablebench_numeric"]["instructions"],
            }
        )
        if len(selected) == COUNTS["tablebench_numeric"]:
            break
    cases.extend(selected)
    rows = json.loads(Path("/private/tmp/dspy-bench-bbeh/bbeh/mini/data.json").read_text())["examples"]
    used = {c["inputs"]["question"] for c in old if c["suite"] == "bbeh_mini"}
    indexes = [i for i, row in enumerate(rows) if row["input"] not in used]
    for index in rng.sample(indexes, COUNTS["bbeh_mini"]):
        row = rows[index]
        cases.append(
            {
                "suite": "bbeh_mini",
                "id": str(index),
                "inputs": {"question": row["input"]},
                "schema": {"type": "string"},
                "gold": row["target"],
                "instructions": templates["bbeh_mini"]["instructions"],
            }
        )
    for suite, count in COUNTS.items():
        actual = sum(c["suite"] == suite for c in cases)
        if actual != count:
            raise ValueError(f"Insufficient eligible cases in {suite}: {actual}/{count}")
    return cases, exclusions


def factorial_messages(case: dict[str, Any], input_format: str, output_format: str) -> list[dict[str, str]]:
    """Hold output instructions fixed while varying only input encoding and its label."""
    signature = core.signature_for(case)
    adapter = core.adapter_for(output_format)
    structure = adapter.format_field_structure(signature)
    if output_format == "json":
        anchor = "Outputs will be a JSON object"
        if anchor not in structure:
            raise ValueError("DSPy JSON output instruction boundary changed")
        structure = anchor + structure.split(anchor, 1)[1]
    system = "\n\n".join(
        [
            adapter.format_field_description(signature),
            f"The user message contains the input fields serialized as a {input_format.upper()} object.",
            structure,
            signature.instructions,
        ]
    )
    content = (
        encode(case["inputs"])
        if input_format == "toon"
        else json.dumps(case["inputs"], ensure_ascii=False, separators=(",", ":"))
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": content}]


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit("Use a new output directory to preserve frozen runs")
    cases, exclusions = collect_cases()
    args.output.mkdir(parents=True)
    core.save_jsonl(args.output / "cases.jsonl", cases)
    core.save_jsonl(args.output / "eligibility_exclusions.jsonl", exclusions)
    requests = []
    for index, case in enumerate(cases):
        signature = core.signature_for(case)
        arms = {a: core.adapter_for(a).format(signature, [], case["inputs"]) for a in ["json", "toon"]}
        if case["suite"] in ABLATION_SUITES:
            arms.update(
                {f"{i}_{o}": factorial_messages(case, i, o) for i in ["json", "toon"] for o in ["json", "toon"]}
            )
        for arm, messages in arms.items():
            for thinking in [False, True]:
                requests.append(
                    {
                        "job_id": f"{case['suite']}:{case['id']}:{arm}:{int(thinking)}",
                        "case_index": index,
                        "suite": case["suite"],
                        "case_id": case["id"],
                        "adapter": arm,
                        "thinking": thinking,
                        "payload": {
                            "model": core.MODEL,
                            "messages": messages,
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "max_tokens": 32768,
                            "seed": SEED + index,
                            "chat_template_kwargs": {"enable_thinking": thinking},
                        },
                    }
                )
    random.Random(SEED).shuffle(requests)
    core.save_jsonl(args.output / "requests.jsonl", requests)
    sources = [Path(__file__), Path(core.__file__), *Path("src/dspy_toon").glob("*.py")]
    for source in sources:
        relative = source.relative_to(Path.cwd()) if source.is_absolute() else source
        dest = args.output / "source_snapshot" / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "model": core.MODEL,
        "dspy": core.dspy.__version__,
        "seed": SEED,
        "counts": COUNTS,
        "planned_calls": len(requests),
        "concurrency": 50,
        "max_tokens": 32768,
        "temperature": 0.6,
        "top_p": 0.95,
        "cases_sha256": core.digest((args.output / "cases.jsonl").read_bytes()),
        "requests_sha256": core.digest((args.output / "requests.jsonl").read_bytes()),
        "code_sha256": {str(p): core.digest(p.read_bytes()) for p in sources},
        "source_provenance": "../qwen_20260905/manifest.json and scorer_sources.json",
        "design": "Actual json/toon adapters on all cases; normalized 2x2 input/output ablation on OCR and tables.",
        "exclusions": len(exclusions),
        "prior_case_overlap": 0,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"cases": len(cases), "calls": len(requests), "exclusions": len(exclusions)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "score"])
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/qwen_blog_expanded"))
    args = parser.parse_args()
    args.concurrency, args.preflight = 50, False
    args.base_url = "http://192.168.36.11:8007/v1"
    args.sob = Path("benchmark_results/qwen_20260905/scorers/sob")
    args.bbeh = Path("benchmark_results/qwen_20260905/scorers/bbeh")
    args.tablebench = Path("benchmark_results/qwen_20260905/scorers/tablebench")
    for i in ["json", "toon"]:
        for o in ["json", "toon"]:
            core.ADAPTERS[f"{i}_{o}"] = core.ADAPTERS[o]
    if args.action == "prepare":
        prepare(args)
    elif args.action == "run":
        asyncio.run(core.run(args))
    else:
        core.score(args)


if __name__ == "__main__":
    main()
