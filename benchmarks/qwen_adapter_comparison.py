"""Paired, one-request adapter benchmark against an OpenAI-compatible vLLM server.

Use prepare to freeze cases/prompts, run to execute the frozen requests, and
score to evaluate stored responses without making more model calls. See the
run's protocol.md for source revisions and the limits of this adapted pilot.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import hashlib
import importlib.util
import io
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

# Load NumPy through Arrow before DSPy installs its optional-dependency lazy loader.
# isort: off
import pyarrow.parquet as pq
import dspy
import httpx

# isort: on
from jsonschema import Draft7Validator
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, create_model

from benchmarks.baml_adapter import BAMLAdapter
from dspy_toon import ToonAdapter

ADAPTERS = {"toon": ToonAdapter, "json": dspy.JSONAdapter, "baml": BAMLAdapter, "chat": dspy.ChatAdapter}
MODEL = "Qwen/Qwen3.8-27B-FP8"
SEED = 42


def digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def schema_type(schema: dict[str, Any], name: str) -> Any:
    """Translate the basic types used by SOB, retaining descriptions/constraints."""
    if "enum" in schema:
        return Literal[tuple(schema["enum"])]
    kind = schema.get("type")
    if kind == "object":
        if not schema.get("properties"):
            additional = schema.get("additionalProperties", {})
            return dict[str, schema_type(additional, name + "Value") if isinstance(additional, dict) else Any]
        fields = {}
        for key, child in schema["properties"].items():
            if not key.isidentifier() or key.startswith("_"):
                raise ValueError(f"Unsupported model field name: {key!r}")
            annotation = schema_type(child, name + key.title())
            required = key in schema.get("required", [])
            constraints = {
                dest: child[src]
                for src, dest in (
                    ("minimum", "ge"),
                    ("maximum", "le"),
                    ("minItems", "min_length"),
                    ("maxItems", "max_length"),
                    ("minLength", "min_length"),
                    ("maxLength", "max_length"),
                )
                if src in child
            }
            fields[key] = (
                annotation,
                Field(... if required else None, description=child.get("description"), **constraints),
            )
        return create_model(
            name,
            __config__=ConfigDict(extra="forbid" if schema.get("additionalProperties") is False else "allow"),
            **fields,
        )
    if kind == "array":
        return list[schema_type(schema["items"], name + "Item")]
    if kind in ("string", "integer", "number", "boolean", "null"):
        return {"string": str, "integer": int, "number": float, "boolean": bool, "null": type(None)}[kind]
    if not schema:
        return Any
    raise ValueError(f"Unsupported schema type: {kind}")


def signature_for(case: dict[str, Any]) -> type[dspy.Signature]:
    output = schema_type(case["schema"], "Result")
    fields = {key: (list[dict[str, str]] if key == "table" else str, dspy.InputField()) for key in case["inputs"]}
    fields["result"] = (output, dspy.OutputField(desc="The requested answer, following all field descriptions."))
    return dspy.Signature(fields, instructions=case["instructions"])


def adapter_for(name: str) -> Any:
    return ADAPTERS[name](use_json_adapter_fallback=False) if name == "chat" else ADAPTERS[name]()


def prepare(args: argparse.Namespace) -> None:
    if (args.output / "manifest.json").exists():
        raise SystemExit("A frozen manifest already exists; use a different output directory")
    cases = []
    sources = []
    rng = random.Random(SEED)
    for modality, file in [("text", "data_test"), ("image", "image_train"), ("audio", "audio_train")]:
        path = args.data / f"sob_{file}-00000-of-00001.parquet"
        rows = pq.read_table(path).to_pylist()
        # Stratify text/image by schema difficulty; audio by context-length quartile.
        groups = defaultdict(list)
        lengths = sorted(len(r["context"]) for r in rows)
        for row in rows:
            group = (
                sum(len(row["context"]) > lengths[int(len(lengths) * q / 4)] for q in (1, 2, 3))
                if modality == "audio"
                else row["schema_complexity"]
            )
            groups[str(group)].append(row)
        for group in groups.values():
            rng.shuffle(group)
        selected = []
        while len(selected) < args.samples:
            for key in sorted(groups):
                if groups[key] and len(selected) < args.samples:
                    selected.append(groups[key].pop())
        for row in selected:
            schema = json.loads(row["json_schema"])
            gold = json.loads(row["ground_truth"])
            Draft7Validator.check_schema(schema)
            # Dataset defects remain recorded; never select cases by model performance.
            gold_errors = [e.message for e in Draft7Validator(schema).iter_errors(gold)]
            cases.append(
                {
                    "suite": "sob_" + modality,
                    "id": row["record_id"],
                    "modality": modality,
                    "inputs": {"context": row["context"], "question": row["question"]},
                    "schema": schema,
                    "gold": gold,
                    "schema_complexity": row["schema_complexity"],
                    "gold_schema_errors": gold_errors,
                    "instructions": "Answer the question using only the provided context. Extract the requested "
                    "information faithfully. "
                    "Follow the result field's schema and descriptions. If the context does not "
                    "contain a requested answer, "
                    "return null rather than guess. Return the final result using the adapter's output format.",
                }
            )
        sources.append(
            {
                "name": "SOB " + modality,
                "file": path.name,
                "sha256": digest(path.read_bytes()),
                "population": len(rows),
                "groups": {k: len(v) for k, v in groups.items()},
            }
        )
    path = args.data / "TableBench_TableBench.jsonl"
    rows = [r for r in read_jsonl(path) if r["qtype"] == "NumericalReasoning"]
    for row in rng.sample(rows, args.samples):
        table = row["table"]
        if isinstance(table, str):
            table = json.loads(table)
        columns = table["columns"]
        if len(set(columns)) != len(columns) or any(len(r) != len(columns) for r in table["data"]):
            raise ValueError("Table has duplicate columns or inconsistent widths")
        cases.append(
            {
                "suite": "tablebench_numeric",
                "id": row["id"],
                "inputs": {
                    "table": [dict(zip(columns, r, strict=True)) for r in table["data"]],
                    "question": row["question"],
                },
                "schema": {"type": "string"},
                "gold": row["answer"],
                "subtype": row["qsubtype"],
                "instructions": "Answer the question using only the provided table. Return only the concise final"
                " answer as the result field. "
                "For multiple answers, use the order asked for and separate them with commas. Do "
                "not include explanations, units "
                "unless asked for, or a 'Final Answer:' prefix in the result. Preserve requested "
                "percentages and precision.",
            }
        )
    sources.append(
        {
            "name": "TableBench NumericalReasoning",
            "file": path.name,
            "sha256": digest(path.read_bytes()),
            "population": len(rows),
        }
    )
    path = args.bbeh / "bbeh/mini/data.json"
    rows = json.loads(path.read_text())["examples"]
    for idx in rng.sample(range(len(rows)), args.samples):
        row = rows[idx]
        cases.append(
            {
                "suite": "bbeh_mini",
                "id": str(idx),
                "inputs": {"question": row["input"]},
                "schema": {"type": "string"},
                "gold": row["target"],
                "instructions": "Solve the task accurately. Return the final answer alone inside the result "
                "field, in the notation requested "
                "by the task. Use the adapter's field format; do not include reasoning or an "
                "answer-prefix sentence in the result.",
            }
        )
    sources.append(
        {"name": "BBEH mini", "file": str(path), "sha256": digest(path.read_bytes()), "population": len(rows)}
    )
    args.output.mkdir(parents=True, exist_ok=True)
    save_jsonl(args.output / "cases.jsonl", cases)
    requests = []
    for index, case in enumerate(cases):
        signature = signature_for(case)
        for name in ADAPTERS:
            adapter = adapter_for(name)
            messages = adapter.format(signature, [], case["inputs"])
            for thinking in (False, True):
                requests.append(
                    {
                        "job_id": f"{case['suite']}:{case['id']}:{name}:{int(thinking)}",
                        "case_index": index,
                        "suite": case["suite"],
                        "case_id": case["id"],
                        "adapter": name,
                        "thinking": thinking,
                        "payload": {
                            "model": MODEL,
                            "messages": messages,
                            "temperature": 0.6,
                            "top_p": 0.95,
                            "max_tokens": args.max_tokens,
                            "seed": SEED + index,
                            "chat_template_kwargs": {"enable_thinking": thinking},
                        },
                    }
                )
    rng.shuffle(requests)
    save_jsonl(args.output / "requests.jsonl", requests)
    code = [Path(__file__), Path("benchmarks/baml_adapter.py"), *Path("src/dspy_toon").glob("*.py")]
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(),
        "model": MODEL,
        "dspy": dspy.__version__,
        "seed": SEED,
        "samples_per_suite": args.samples,
        "planned_calls": len(requests),
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "temperature": 0.6,
        "top_p": 0.95,
        "sources": sources,
        "cases_sha256": digest((args.output / "cases.jsonl").read_bytes()),
        "requests_sha256": digest((args.output / "requests.jsonl").read_bytes()),
        "code_sha256": {str(p): digest(p.read_bytes()) for p in code},
        "method": (
            "Adapter.format -> identical unconstrained vLLM transport -> Adapter.parse -> "
            "common validation/scoring; no retries or fallback model calls"
        ),
        "baml": "Repository BAML-inspired DSPy JSONAdapter subclass, not BoundaryML BAML runtime",
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(
        json.dumps(
            {
                "cases": len(cases),
                "calls": len(requests),
                "gold_schema_defects": sum(bool(c.get("gold_schema_errors")) for c in cases),
            }
        ),
        flush=True,
    )


async def run(args: argparse.Namespace) -> None:
    manifest = json.loads((args.output / "manifest.json").read_text())
    requests_path = args.output / "requests.jsonl"
    if digest(requests_path.read_bytes()) != manifest["requests_sha256"]:
        raise SystemExit("Frozen requests have changed")
    for file, expected in manifest["code_sha256"].items():
        if digest(Path(file).read_bytes()) != expected:
            raise SystemExit(f"Frozen code changed: {file}")
    rows = read_jsonl(requests_path)
    result_path = args.output / ("preflight.jsonl" if args.preflight else "responses.jsonl")
    if args.preflight:
        seen = set()
        selected = []
        for row in rows:
            key = (row["suite"], row["adapter"], row["thinking"])
            if key not in seen:
                selected.append(row)
                seen.add(key)
        rows = selected
    done = {r["job_id"] for r in read_jsonl(result_path)} if result_path.exists() else set()
    rows = [r for r in rows if r["job_id"] not in done]
    queue = asyncio.Queue()
    for row in rows:
        queue.put_nowait(row)
    timeout = httpx.Timeout(600, connect=10)
    failures = 0
    stop = asyncio.Event()
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:

        async def worker() -> None:
            nonlocal failures
            while not queue.empty() and not stop.is_set():
                row = queue.get_nowait()
                start = time.perf_counter()
                result = {k: v for k, v in row.items() if k != "payload"}
                result["started"] = datetime.now(timezone.utc).isoformat()
                try:
                    response = await client.post(args.base_url.rstrip("/") + "/chat/completions", json=row["payload"])
                    result.update(latency_s=time.perf_counter() - start, status_code=response.status_code)
                    result["response"] = response.json()
                    response.raise_for_status()
                    if not result["response"].get("choices"):
                        raise ValueError("No choices in server response")
                    failures = 0
                except Exception as error:
                    failures += 1
                    result.update(error=f"{type(error).__name__}: {error}", latency_s=time.perf_counter() - start)
                    if failures >= 3:
                        stop.set()
                with result_path.open("a") as stream:
                    stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                done.add(row["job_id"])
                if len(done) % 10 == 0 or result.get("error"):
                    print(
                        f"{len(done)} completed; {queue.qsize()} queued; "
                        f"latest={row['suite']}/{row['adapter']}/thinking={row['thinking']} "
                        f"{result['latency_s']:.1f}s",
                        flush=True,
                    )
                queue.task_done()

        await asyncio.gather(*(worker() for _ in range(args.concurrency)))
    print(f"Finished: {len(done)} recorded; {queue.qsize()} unrun", flush=True)


def load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    with contextlib.redirect_stdout(io.StringIO()):
        spec.loader.exec_module(module)
    return module


def score(args: argparse.Namespace) -> None:
    sob = load_module("sob_metrics", args.sob / "evaluate.py")
    bbeh = load_module("bbeh_metrics", args.bbeh / "bbeh/evaluate.py")
    tablebench = load_module("tablebench_metrics", args.tablebench / "metrics/custom_em_metric.py")
    cases = read_jsonl(args.output / "cases.jsonl")
    source = args.output / ("preflight.jsonl" if args.preflight else "responses.jsonl")
    scored = []
    for row in read_jsonl(source):
        case = cases[row["case_index"]]
        record = {k: v for k, v in row.items() if k != "response"}
        raw = row.get("response", {})
        choice = (raw.get("choices") or [{}])[0]
        message = choice.get("message", {})
        content = message.get("content") or ""
        reasoning = message.get("reasoning") or message.get("reasoning_content") or ""
        record.update(
            usage=raw.get("usage", {}),
            finish_reason=choice.get("finish_reason"),
            reasoning_chars=len(reasoning),
            final_chars=len(content),
            system_fingerprint=raw.get("system_fingerprint"),
            parse_success=False,
            schema_valid=False,
            score=0.0,
            exact_match=0.0,
        )
        if not row.get("error"):
            try:
                signature = signature_for(case)
                parsed = adapter_for(row["adapter"]).parse(signature, content)
                value = TypeAdapter(signature.output_fields["result"].annotation).validate_python(parsed["result"])
                value = value.model_dump(mode="json", exclude_unset=True) if isinstance(value, BaseModel) else value
                record.update(parse_success=True, prediction=value)
                Draft7Validator(case["schema"]).validate(value)
                record["schema_valid"] = True
                if case["suite"].startswith("sob_"):
                    metrics = sob.evaluate_record(
                        {
                            "metadata": {"record_id": case["id"], "schema_complexity": case["schema_complexity"]},
                            "input": {"json_schema": case["schema"]},
                            "output": {"ground_truth": case["gold"], "candidate_response": value},
                        },
                        case["modality"],
                    ).row
                    record.update(
                        score=metrics["leaf_value_em"],
                        exact_match=metrics["strict_json_em"],
                        token_f1=metrics["value_token_f1"],
                        sob_metrics=metrics,
                    )
                elif case["suite"] == "bbeh_mini":
                    record["score"] = record["exact_match"] = float(bbeh.evaluate_correctness(value, case["gold"]))
                else:
                    record["score"] = tablebench.compute_em([case["gold"]], [value])
                    record["exact_match"] = float(record["score"] >= 1 - 1e-10)
            except Exception as error:
                record["parse_or_score_error"] = f"{type(error).__name__}: {error}"
        scored.append(record)
    prefix = "preflight_" if args.preflight else ""
    save_jsonl(args.output / (prefix + "scored.jsonl"), scored)
    groups = defaultdict(list)
    for row in scored:
        groups[(row["suite"], row["adapter"], row["thinking"])].append(row)
    summary = []
    for (suite, adapter, thinking), group in sorted(groups.items()):
        summary.append(
            {
                "suite": suite,
                "adapter": adapter,
                "thinking": thinking,
                "n": len(group),
                "score": statistics.mean(r["score"] for r in group),
                "exact_match": statistics.mean(r["exact_match"] for r in group),
                "parse_success": sum(r["parse_success"] for r in group),
                "schema_valid": sum(r["schema_valid"] for r in group),
                "truncated": sum(r["finish_reason"] == "length" for r in group),
                "http_errors": sum(bool(r.get("error")) for r in group),
                "prompt_tokens": sum(r["usage"].get("prompt_tokens", 0) for r in group),
                "completion_tokens": sum(r["usage"].get("completion_tokens", 0) for r in group),
                "reasoning_responses": sum(r["reasoning_chars"] > 0 for r in group),
                "latency_median_s": statistics.median(r["latency_s"] for r in group),
                "latency_mean_s": statistics.mean(r["latency_s"] for r in group),
            }
        )
    (args.output / (prefix + "summary.json")).write_text(json.dumps(summary, indent=2))
    print(
        json.dumps(
            {
                "scored": len(scored),
                "parse_failures": sum(not r["parse_success"] for r in scored),
                "truncated": sum(r["finish_reason"] == "length" for r in scored),
            }
        ),
        flush=True,
    )
    for r in summary:
        print(
            f"{r['suite']:19} {r['adapter']:5} thinking={r['thinking']} n={r['n']:2} "
            f"score={r['score']:.3f} parsed={r['parse_success']} "
            f"tokens={r['prompt_tokens'] + r['completion_tokens']} latency={r['latency_mean_s']:.1f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "score"])
    parser.add_argument("--output", type=Path, default=Path("benchmark_results/qwen_20260905"))
    parser.add_argument("--data", type=Path, default=Path("/private/tmp/dspy-bench-hf"))
    parser.add_argument("--sob", type=Path, default=Path("/private/tmp/dspy-bench-sob"))
    parser.add_argument("--bbeh", type=Path, default=Path("/private/tmp/dspy-bench-bbeh"))
    parser.add_argument("--tablebench", type=Path, default=Path("/private/tmp/dspy-bench-tablebench"))
    parser.add_argument("--base-url", default="http://192.168.36.11:8007/v1")
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.action == "run":
        asyncio.run(run(args))
    else:
        {"prepare": prepare, "score": score}[args.action](args)


if __name__ == "__main__":
    main()
