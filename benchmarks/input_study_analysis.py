"""Score the frozen input study and report paired, dataset-aware comparisons."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import numpy as np

OUT = Path("benchmark_results/qwen_input_study")


def write_csv(name, rows):
    if not rows:
        return
    with (OUT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def score():
    upstream = json.loads((OUT / "upstream_cases.json").read_text())
    tools = json.loads((OUT / "tool_cases.json").read_text())
    responses = [json.loads(s) for s in (OUT / "responses.jsonl").read_text().splitlines()]
    manifest = json.loads((OUT / "manifest.json").read_text())
    if len(responses) != manifest["planned_calls"] or len({r["job_id"] for r in responses}) != len(responses):
        raise ValueError("Study is incomplete or contains duplicate responses")
    pending, rows = [], []
    for r in responses:
        result = r.get("response", {})
        choice = (result.get("choices") or [{}])[0]
        content = choice.get("message", {}).get("content") or ""
        usage = result.get("usage") or {}
        track = r["suite"]
        c = (upstream if track == "upstream" else tools)[r["case_index"]]
        row = {
            k: r[k]
            for k in ["job_id", "case_index", "case_id", "dataset", "adapter", "thinking", "repeat", "latency_s"]
        }
        row.update(
            track=track,
            category=c["type"] if track == "upstream" else c["category"],
            rows=c.get("rows"),
            transport_error=bool(r.get("error")),
            truncated=choice.get("finish_reason") == "length",
            content=content,
            reasoning_chars=len(choice.get("message", {}).get("reasoning") or ""),
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", 0),
            strict_valid=False,
            score=0,
            fence_score=0,
        )
        if track == "upstream":
            pending.append(
                dict(
                    job_id=r["job_id"],
                    actual=content.strip(),
                    expected=c["groundTruth"],
                    kind=c.get("answerType", "string"),
                    options=c.get("normalizationOptions", {}),
                )
            )
        else:

            def assess(text):
                try:
                    value = json.loads(text)
                    scalar = value.get("result") if isinstance(value, dict) else None
                    valid = isinstance(value, dict) and set(value) == {"result"} and isinstance(scalar, str)
                    correct = (
                        isinstance(scalar, (str, int, float)) and str(scalar).strip().casefold() == c["gold"].casefold()
                    )
                    return valid, correct
                except (ValueError, TypeError):
                    return False, False

            row["strict_valid"], row["score"] = assess(content)
            unfenced = content.strip()
            if unfenced.startswith("```") and unfenced.endswith("```"):
                unfenced = "\n".join(unfenced.splitlines()[1:-1])
            row["fence_score"] = assess(unfenced)[1]
        rows.append(row)
    (OUT / "normalizer_input.json").write_text(json.dumps(pending))
    subprocess.run(
        [
            "node",
            "benchmarks/input_study_bridge.mjs",
            "score",
            "/private/tmp/dspy-toon-upstream",
            str(OUT / "normalizer_input.json"),
            str(OUT / "normalizer_output.json"),
        ],
        check=True,
    )
    normalized = {r["job_id"]: r for r in json.loads((OUT / "normalizer_output.json").read_text())}
    for r in rows:
        if r["track"] == "upstream":
            r["score"] = normalized[r["job_id"]]["match"] and not r["transport_error"]
            r["fence_score"] = r["score"]
    (OUT / "scored.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    return rows


def summarize(rows):
    return dict(
        n=len(rows),
        score=np.mean([r["score"] for r in rows]),
        strict_valid=sum(r["strict_valid"] for r in rows),
        fence_score=np.mean([r["fence_score"] for r in rows]),
        truncated=sum(r["truncated"] for r in rows),
        errors=sum(r["transport_error"] for r in rows),
        prompt_tokens=sum(r["prompt_tokens"] for r in rows),
        completion_tokens=sum(r["completion_tokens"] for r in rows),
        total_tokens=sum(r["total_tokens"] for r in rows),
        latency_p50=np.median([r["latency_s"] for r in rows]),
        latency_p95=np.percentile([r["latency_s"] for r in rows], 95),
    )


def contrast(rows, control, cluster=False):
    pairs = {(r["case_id"], r["adapter"]): r for r in rows}
    a = [r for r in rows if r["adapter"] == "toon" and (r["case_id"], control) in pairs]
    b = [pairs[r["case_id"], control] for r in a]
    if not a:
        return None
    delta = np.array([float(x["score"]) - float(y["score"]) for x, y in zip(a, b, strict=True)])
    token_a = np.array([r["total_tokens"] for r in a], dtype=float)
    token_b = np.array([r["total_tokens"] for r in b], dtype=float)
    rng = np.random.default_rng(20260907)
    groups = sorted({r["dataset"] for r in a}) if cluster else list(range(len(a)))
    ix = [np.array([i for i, r in enumerate(a) if r["dataset"] == g]) for g in groups] if cluster else None
    boot = []
    ratios = []
    for _ in range(10000):
        draw = rng.integers(0, len(groups), len(groups))
        indexes = np.concatenate([ix[k] for k in draw]) if cluster else draw
        boot.append(delta[indexes].mean())
        ratios.append(token_a[indexes].sum() / max(1, token_b[indexes].sum()))
    ci = np.percentile(boot, [2.5, 97.5])
    tc = np.percentile(ratios, [2.5, 97.5])
    return dict(
        control=control,
        n=len(a),
        clusters=len(groups),
        clustered=cluster,
        score_delta=delta.mean(),
        ci_low=ci[0],
        ci_high=ci[1],
        toon_wins=int((delta > 0).sum()),
        ties=int((delta == 0).sum()),
        json_wins=int((delta < 0).sum()),
        total_token_ratio=token_a.sum() / max(1, token_b.sum()),
        token_ci_low=tc[0],
        token_ci_high=tc[1],
        prompt_token_ratio=sum(r["prompt_tokens"] for r in a) / max(1, sum(r["prompt_tokens"] for r in b)),
    )


def main():
    rows = score()
    primary = [r for r in rows if r["repeat"] == 0]
    metrics = []
    for track in ["upstream", "tools"]:
        track_rows = [r for r in primary if r["track"] == track]
        slices = {"all": track_rows}
        if track == "upstream":
            slices["ordinary"] = [
                r for r in track_rows if r["category"] not in ["structure-awareness", "structural-validation"]
            ]
            slices["flat"] = [
                r for r in track_rows if r["case_id"] in {x["case_id"] for x in track_rows if x["adapter"] == "csv"}
            ]
        for field in ["dataset", "category"] + (["rows"] if track == "tools" else []):
            for value in sorted({r[field] for r in track_rows}, key=str):
                slices[f"{field}:{value}"] = [r for r in track_rows if r[field] == value]
        for slice_name, subset in slices.items():
            for thinking in [False, True]:
                for arm in sorted({r["adapter"] for r in subset}):
                    cell = [r for r in subset if r["thinking"] == thinking and r["adapter"] == arm]
                    if cell:
                        metrics.append(
                            dict(track=track, slice=slice_name, thinking=thinking, adapter=arm, **summarize(cell))
                        )
    write_csv("metrics.csv", metrics)
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pairs = []
    for track in ["upstream", "tools"]:
        base = [r for r in primary if r["track"] == track]
        slices = {"all": base}
        if track == "upstream":
            slices["ordinary"] = [
                r for r in base if r["category"] not in ["structure-awareness", "structural-validation"]
            ]
        else:
            slices.update({d: [r for r in base if r["dataset"] == d] for d in sorted({r["dataset"] for r in base})})
        for label, subset in slices.items():
            for thinking in [False, True]:
                for control in ["json-compact", "json-pretty", "csv"] if track == "upstream" else ["json-compact"]:
                    cell = [r for r in subset if r["thinking"] == thinking]
                    for cluster in [False, True] if track == "upstream" else [False]:
                        result = contrast(cell, control, cluster)
                        if result:
                            pairs.append(dict(track=track, slice=label, thinking=thinking, **result))
    write_csv("paired.csv", pairs)
    (OUT / "paired.json").write_text(json.dumps(pairs, indent=2))
    repeats = []
    lookup = {(r["case_id"], r["adapter"], r["thinking"]): r for r in primary if r["track"] == "tools"}
    for thinking in [False, True]:
        for arm in ["json-compact", "toon"]:
            second = [
                r
                for r in rows
                if r["track"] == "tools" and r["repeat"] == 1 and r["thinking"] == thinking and r["adapter"] == arm
            ]
            first = [lookup[r["case_id"], arm, thinking] for r in second]
            repeats.append(
                dict(
                    adapter=arm,
                    thinking=thinking,
                    n=len(second),
                    first_score=np.mean([r["score"] for r in first]),
                    second_score=np.mean([r["score"] for r in second]),
                    correctness_flips=sum(a["score"] != b["score"] for a, b in zip(first, second, strict=True)),
                )
            )
    write_csv("seed_sensitivity.csv", repeats)
    print("Scored", len(rows), "requests;", sum(r["transport_error"] for r in rows), "transport errors")
    print(json.dumps([r for r in metrics if r["slice"] == "all"], indent=2))


if __name__ == "__main__":
    main()
