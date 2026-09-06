"""Render blog-ready SVG/PNG charts from exported metrics; makes no model calls.

uv run --no-project --with matplotlib python benchmarks/blog_plots.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FOLDER = Path("benchmark_results/qwen_blog_expanded")
COLORS = {
    "json": "#2563eb",
    "toon": "#d97706",
    "json_json": "#2563eb",
    "toon_json": "#7c3aed",
    "json_toon": "#0d9488",
    "toon_toon": "#d97706",
}
LABELS = {
    "sob_text": "Text extraction",
    "sob_image": "OCR-text extraction",
    "sob_audio": "Meeting transcripts",
    "tablebench_numeric": "Table reasoning",
    "bbeh_mini": "Hard reasoning",
}
SUITES = list(LABELS)
ARMS = ["json_json", "toon_json", "json_toon", "toon_toon"]
ARM_LABELS = {
    "json_json": "JSON → JSON",
    "toon_json": "TOON → JSON",
    "json_toon": "JSON → TOON",
    "toon_toon": "TOON → TOON",
}


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(FOLDER / "figures" / f"{name}.svg", bbox_inches="tight")
    fig.savefig(FOLDER / "figures" / f"{name}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = json.loads((FOLDER / "metrics.json").read_text())
    lookup = {(r["suite"], r["arm"], r["thinking"]): r for r in rows}
    (FOLDER / "figures").mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    x = np.arange(len(SUITES))
    for ax, thinking in zip(axes, [False, True], strict=True):
        for offset, arm in [(-0.19, "json"), (0.19, "toon")]:
            values = [lookup[s, arm, thinking] for s in SUITES]
            means = np.array([100 * r["score"] for r in values])
            errors = np.array(
                [
                    [100 * r["score"] - 100 * r["score_ci_low"] for r in values],
                    [100 * r["score_ci_high"] - 100 * r["score"] for r in values],
                ]
            )
            ax.bar(
                x + offset,
                means,
                width=0.36,
                color=COLORS[arm],
                label=arm.upper(),
                yerr=errors,
                capsize=3,
                error_kw={"linewidth": 1},
            )
        ax.set_title("Native thinking " + ("on" if thinking else "off"))
        ax.set_xticks(x, [LABELS[s].replace(" ", "\n", 1) for s in SUITES])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Task-specific score (%)")
    axes[1].legend(frameon=False)
    fig.suptitle("TOON vs JSON in DSPy: larger, disjoint-case evaluation", fontsize=14)
    fig.supxlabel("n = 100 / 100 / 94 / 150 / 100 per adapter and mode · 32K cap · 95% bootstrap intervals", fontsize=9)
    save(fig, "01_actual_accuracy")

    paired = json.loads((FOLDER / "paired.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, layout="constrained")
    for ax, thinking in zip(axes, [False, True], strict=True):
        vals = [
            next(
                r
                for r in paired
                if r["suite"] == s and r["thinking"] == thinking and r["comparison"] == "actual_adapter"
            )
            for s in SUITES
        ]
        means = np.array([100 * r["score_delta"] for r in vals])
        err = np.array(
            [
                [means[i] - 100 * r["ci95_low"] for i, r in enumerate(vals)],
                [100 * r["ci95_high"] - means[i] for i, r in enumerate(vals)],
            ]
        )
        ax.errorbar(means, x, xerr=err, fmt="o", color=COLORS["toon"], capsize=4)
        ax.axvline(0, color="#64748b", linewidth=1)
        ax.set_yticks(x, [LABELS[s] for s in SUITES])
        ax.invert_yaxis()
        ax.set_title("Thinking " + ("on" if thinking else "off"))
        ax.set_xlabel("TOON − JSON (percentage points)")
        ax.grid(axis="x", alpha=0.18)
    fig.suptitle("Paired differences: small gaps need uncertainty", fontsize=14)
    fig.supxlabel("95% case-bootstrap intervals · descriptive, without multiple-comparison correction", fontsize=9)
    save(fig, "02_paired_accuracy")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    for ax, thinking in zip(axes, [False, True], strict=True):
        ratios = []
        for s in SUITES:
            a, b = lookup[s, "toon", thinking], lookup[s, "json", thinking]
            ratios.append(100 * (a["total_tokens"] / b["total_tokens"] - 1))
        ax.barh(x, ratios, color=["#0d9488" if v < 0 else "#d97706" for v in ratios])
        ax.set_yticks(x, [LABELS[s] for s in SUITES])
        ax.invert_yaxis()
        ax.axvline(0, color="#64748b", linewidth=1)
        ax.set_title("Thinking " + ("on" if thinking else "off"))
        ax.set_xlabel("Total token change vs JSON (%)")
        for i, v in enumerate(ratios):
            ax.text(v + (0.8 if v >= 0 else -0.8), i, f"{v:+.1f}%", ha="left" if v >= 0 else "right", va="center")
        ax.margins(x=0.28)
    fig.suptitle("Input compression is not the same as total token savings", fontsize=14)
    fig.supxlabel("All calls included, including failures · completion tokens include native reasoning", fontsize=9)
    save(fig, "03_total_tokens")

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey="row", layout="constrained")
    for row, suite in enumerate(["tablebench_numeric", "sob_image"]):
        for col, thinking in enumerate([False, True]):
            ax = axes[row, col]
            for arm in ARMS:
                r = lookup[suite, arm, thinking]
                ax.scatter(r["mean_total_tokens"], 100 * r["score"], s=85, color=COLORS[arm], label=ARM_LABELS[arm])
                ax.annotate(
                    ARM_LABELS[arm],
                    (r["mean_total_tokens"], 100 * r["score"]),
                    xytext=(5, -16) if arm == "json_json" else (5, 6),
                    textcoords="offset points",
                    fontsize=8,
                )
            ax.set_title(LABELS[suite] + " · thinking " + ("on" if thinking else "off"))
            ax.set_xlabel("Mean total tokens / request")
            ax.set_ylabel("Task score (%)")
            ax.grid(alpha=0.18)
            ax.margins(0.25)
    fig.suptitle("Separate input encoding from output format", fontsize=14)
    fig.supxlabel(
        "Point estimates; paired intervals in report · compact JSON baseline · 150 table and 100 OCR cases per cell",
        fontsize=9,
    )
    save(fig, "04_factorial_quality_cost")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True, layout="constrained")
    for ax, thinking in zip(axes, [False, True], strict=True):
        for offset, arm in [(-0.19, "json"), (0.19, "toon")]:
            vals = [lookup[s, arm, thinking] for s in SUITES]
            raw = np.array([100 * r["raw_schema_valid"] / r["n"] for r in vals])
            accepted = np.array([100 * r["schema_valid"] / r["n"] for r in vals])
            ax.bar(x + offset, raw, width=0.36, color=COLORS[arm], alpha=0.45, label=arm.upper() + " raw schema")
            ax.scatter(
                x + offset, accepted, color=COLORS[arm], marker="D", s=25, label=arm.upper() + " adapter accepted"
            )
        ax.set_xticks(x, [LABELS[s].replace(" ", "\n", 1) for s in SUITES])
        ax.set_ylim(0, 105)
        ax.set_title("Thinking " + ("on" if thinking else "off"))
        ax.grid(axis="y", alpha=0.18)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Valid outputs (%)")
    axes[1].legend(frameon=False, fontsize=8, loc="lower left")
    fig.suptitle("Unfenced responses versus adapter-accepted output", fontsize=14)
    fig.supxlabel(
        "Raw: unfenced syntax + wrapper + original schema, without repair/coercion\n"
        "80 of 88 actual-JSON recoveries needed only Markdown fence removal",
        fontsize=9,
    )
    save(fig, "05_raw_vs_accepted")

    diagnostics = [json.loads(x) for x in (FOLDER / "diagnostics.jsonl").read_text().splitlines()]
    table = {}
    for r in diagnostics:
        if r["suite"] == "tablebench_numeric" and not r["thinking"]:
            table[r["case_id"], r["adapter"]] = r
    fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
    for arm_a, arm_b, label, color in [
        ("toon", "json", "Actual adapters", COLORS["toon"]),
        ("toon_json", "json_json", "Normalized inputs, JSON output", COLORS["toon_json"]),
    ]:
        vals = [r for (k, a), r in table.items() if a == arm_a]
        cells = [r["table_rows"] * r["table_columns"] for r in vals]
        savings = [
            100 * (1 - r["usage"]["prompt_tokens"] / table[r["case_id"], arm_b]["usage"]["prompt_tokens"]) for r in vals
        ]
        ax.scatter(cells, savings, s=20, alpha=0.6, color=color, label=label)
    ax.set_xscale("log")
    ax.set_xlabel("Input table cells (rows × columns), log scale")
    ax.set_ylabel("Prompt-token saving vs matched JSON (%)")
    ax.axhline(0, color="#64748b", linewidth=1)
    ax.grid(alpha=0.15)
    ax.legend(frameon=False)
    ax.set_title("Where table-input compression pays off")
    fig.supxlabel(
        "150 distinct tables · measured server prompt-token usage · includes instructions and format labels", fontsize=9
    )
    save(fig, "06_table_size_compression")
    print("Saved six charts as SVG and 220-DPI PNG")


if __name__ == "__main__":
    main()
