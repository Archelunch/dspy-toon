"""Pinned upstream input-comprehension reproduction and independent tool-result workloads."""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

from benchmarks import qwen_adapter_comparison as core
from dspy_toon import encode

ROOT = Path("/private/tmp/dspy-toon-upstream")
OUT = Path("benchmark_results/qwen_input_study")
SEED = 20260907


def tool_cases():
    cases = []
    for family in ["inventory", "customers", "services", "logs", "orders"]:
        for i in range(100):
            rng = random.Random(f"{SEED}:{family}:{i}")
            count = [10, 30, 100, 300][i % 4]
            rows = []
            for j in range(count):
                rid = f"R{j:04d}"
                if family == "inventory":
                    r = dict(
                        id=rid,
                        warehouse=rng.choice(["east", "west", "north"]),
                        quantity=rng.randrange(100),
                        reorder_level=20,
                        active=rng.choice([True, False]),
                        unit_price=rng.randrange(1, 100),
                    )
                elif family == "customers":
                    r = dict(
                        id=rid,
                        profile=dict(
                            city=rng.choice(["Berlin", "Paris", "Belgrade"]), tier=rng.choice(["basic", "pro"])
                        ),
                        plan=dict(seats=rng.randrange(1, 30), monthly_price=rng.randrange(10, 100)),
                        active=rng.choice([True, False]),
                    )
                elif family == "services":
                    r = dict(
                        region=rng.choice(["eu", "us", "ap"]),
                        replicas=rng.randrange(1, 10),
                        error_count=rng.randrange(20),
                        enabled=rng.choice([True, False]),
                    )
                elif family == "logs":
                    r = dict(
                        id=rid,
                        service=rng.choice(["api", "worker", "auth"]),
                        status=rng.choice([200, 400, 500]),
                        latency_ms=rng.randrange(1, 500),
                    )
                    if j % 2:
                        r["error"] = dict(code=rng.choice(["TIMEOUT", "INVALID"]), retryable=rng.choice([True, False]))
                else:
                    r = dict(
                        id=rid,
                        customer=dict(region=rng.choice(["eu", "us", "ap"])),
                        status=rng.choice(["paid", "pending", "cancelled"]),
                        items=[
                            dict(sku=f"S{k}", quantity=rng.randrange(1, 8), unit_price=rng.randrange(1, 40))
                            for k in range(rng.randrange(1, 5))
                        ],
                    )
                rows.append(r)
            target = rng.randrange(count)
            op = (i // 4) % 5
            if family == "inventory":
                options = [
                    (f"What is quantity for id R{target:04d}?", rows[target]["quantity"]),
                    ("How many records have active=true?", sum(r["active"] for r in rows)),
                    ("What is the sum of quantity across all records?", sum(r["quantity"] for r in rows)),
                    (
                        "How many records have warehouse=east and quantity below reorder_level?",
                        sum(r["warehouse"] == "east" and r["quantity"] < r["reorder_level"] for r in rows),
                    ),
                    (f"What is warehouse for id R{target:04d}?", rows[target]["warehouse"]),
                ]
                data = {"inventory": rows}
            elif family == "customers":
                options = [
                    (f"What is plan.seats for id R{target:04d}?", rows[target]["plan"]["seats"]),
                    ("How many customers have profile.tier=pro?", sum(r["profile"]["tier"] == "pro" for r in rows)),
                    ("What is the sum of plan.seats across all customers?", sum(r["plan"]["seats"] for r in rows)),
                    (
                        "How many customers have active=true and profile.city=Berlin?",
                        sum(r["active"] and r["profile"]["city"] == "Berlin" for r in rows),
                    ),
                    (f"What is profile.city for id R{target:04d}?", rows[target]["profile"]["city"]),
                ]
                data = {"customers": rows}
            elif family == "services":
                options = [
                    (f"What is replicas for service R{target:04d}?", rows[target]["replicas"]),
                    ("How many services have enabled=true?", sum(r["enabled"] for r in rows)),
                    ("What is the sum of error_count across all services?", sum(r["error_count"] for r in rows)),
                    (
                        "How many services have region=eu and error_count greater than 10?",
                        sum(r["region"] == "eu" and r["error_count"] > 10 for r in rows),
                    ),
                    (f"What is region for service R{target:04d}?", rows[target]["region"]),
                ]
                data = {"services": {f"R{j:04d}": r for j, r in enumerate(rows)}}
            elif family == "logs":
                options = [
                    (f"What is latency_ms for id R{target:04d}?", rows[target]["latency_ms"]),
                    ("How many logs have status=500?", sum(r["status"] == 500 for r in rows)),
                    ("What is the sum of latency_ms across all logs?", sum(r["latency_ms"] for r in rows)),
                    (
                        "How many logs have service=api and status=500?",
                        sum(r["service"] == "api" and r["status"] == 500 for r in rows),
                    ),
                    (f"What is service for id R{target:04d}?", rows[target]["service"]),
                ]
                data = {"logs": rows}
            else:
                options = [
                    (f"How many items are in order R{target:04d}?", len(rows[target]["items"])),
                    ("How many orders have status=paid?", sum(r["status"] == "paid" for r in rows)),
                    (
                        "What is the sum of item quantity across every item in every order?",
                        sum(t["quantity"] for r in rows for t in r["items"]),
                    ),
                    (
                        "How many orders have customer.region=eu and status=paid?",
                        sum(r["customer"]["region"] == "eu" and r["status"] == "paid" for r in rows),
                    ),
                    (f"What is customer.region for order R{target:04d}?", rows[target]["customer"]["region"]),
                ]
                data = {"orders": rows}
            question, gold = options[op]
            cases.append(
                dict(
                    id=f"{family}-{i:03}",
                    dataset=family,
                    data=data,
                    rows=count,
                    category=["lookup-number", "count", "sum", "filter", "lookup-string"][op],
                    question=question,
                    gold=str(gold),
                    toon=encode(data),
                )
            )
    return cases


def prepare():
    upstream = json.loads((OUT / "upstream_cases.json").read_text())
    tools = tool_cases()
    (OUT / "tool_cases.json").write_text(json.dumps(tools, ensure_ascii=False))
    subprocess.run(
        [
            "node",
            "benchmarks/input_study_bridge.mjs",
            "verify",
            str(ROOT),
            str(OUT / "tool_cases.json"),
            str(OUT / "encoder_verification.json"),
        ],
        check=True,
    )
    requests = []
    for index, c in enumerate(upstream):
        for arm, f in c["formats"].items():
            prompt = f"""{f["primer"]}

Given the following data in {arm} format:

```{f["fence"]}
{f["text"]}
```

Question: {c["prompt"]}
Answer format requirements:
- Provide only the value itself, no explanation
- For numbers: output digits only (no commas, currency symbols, or units)
- For dates/field names: use the exact string from the data
- For lists: output comma-separated values with no spaces

Answer:"""
            for thinking in [False, True]:
                requests.append(job(c, index, "upstream", arm, thinking, 0, [{"role": "user", "content": prompt}]))
    for index, c in enumerate(tools):
        for arm in ["json-compact", "toon"]:
            body = c["toon"] if arm == "toon" else json.dumps(c["data"], separators=(",", ":"), ensure_ascii=False)
            system = (
                "Answer the question using only the supplied tool-result data. "
                'Return a JSON object with exactly one field, "result", containing the concise answer as a string. '
                "For counts or sums use digits without units or thousands separators. Do not include explanations.\n"
                + (
                    "TOON: Arrays declare length and fields; rows follow header order. "
                    "Nested header groups flatten nested objects; keyed maps have key[N:]{fields} headers."
                    if arm == "toon"
                    else "JSON: Objects and arrays with explicit keys."
                )
            )
            msgs = [
                {"role": "system", "content": system},
                {"role": "user", "content": f"Tool result ({arm}):\n{body}\n\nQuestion: {c['question']}"},
            ]
            # Preselected 20% repeated with another generation seed; dataset remains identical.
            for repeat in [0, 1] if index % 5 == 0 else [0]:
                for thinking in [False, True]:
                    requests.append(job(c, index, "tools", arm, thinking, repeat, msgs))
    random.Random(SEED).shuffle(requests)
    core.save_jsonl(OUT / "requests.jsonl", requests)
    files = [
        Path(__file__),
        Path("benchmarks/input_study_bridge.mjs"),
        Path(core.__file__),
        *Path("src/dspy_toon").glob("*.py"),
    ]
    files += list((ROOT / "benchmarks/src").rglob("*.ts")) + list((ROOT / "packages/toon/src").rglob("*.ts"))
    manifest = dict(
        planned_calls=len(requests),
        requests_sha256=core.digest((OUT / "requests.jsonl").read_bytes()),
        code_sha256={str(p): core.digest(p.read_bytes()) for p in files},
        upstream_commit=subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip(),
        upstream_cases=len(upstream),
        independent_tool_datasets=len(tools),
        concurrency=50,
        repeated_tool_datasets=100,
        seed=SEED,
        cases_sha256={n: core.digest((OUT / n).read_bytes()) for n in ["upstream_cases.json", "tool_cases.json"]},
    )
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(Counter(r["suite"] for r in requests)), len(requests), flush=True)


def job(case, index, track, arm, thinking, repeat, messages):
    return dict(
        job_id=f"{track}:{case['id']}:{arm}:{int(thinking)}:{repeat}",
        case_index=index,
        case_id=case["id"],
        suite=track,
        dataset=case["dataset"],
        adapter=arm,
        thinking=thinking,
        repeat=repeat,
        payload=dict(
            model=core.MODEL,
            messages=messages,
            temperature=0.6,
            top_p=0.95,
            max_tokens=32768,
            seed=SEED + index + repeat * 10000,
            chat_template_kwargs=dict(enable_thinking=thinking),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        asyncio.run(
            core.run(
                argparse.Namespace(output=OUT, preflight=False, concurrency=50, base_url="http://192.168.36.11:8007/v1")
            )
        )
