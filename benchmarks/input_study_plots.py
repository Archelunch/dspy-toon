"""Publication assets for the frozen input-comprehension study (no inference)."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path("benchmark_results/qwen_input_study")
COLORS = {"json-pretty": "#64748b", "json-compact": "#2563eb", "toon": "#d97706", "csv": "#0d9488"}
LABELS = {"json-pretty": "Pretty JSON", "json-compact": "Compact JSON", "toon": "TOON", "csv": "CSV"}


def main():
    metrics = json.loads((OUT / "metrics.json").read_text())
    lookup = {(r["track"], r["slice"], r["thinking"], r["adapter"]): r for r in metrics}
    paired = json.loads((OUT / "paired.json").read_text())
    (OUT / "figures").mkdir(exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )

    def save(fig, name):
        for ext in ["svg", "png"]:
            fig.savefig(OUT / "figures" / f"{name}.{ext}", dpi=220, bbox_inches="tight")
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, thinking in zip(axes, [False, True], strict=True):
        for offset, arm in zip([-0.25, 0, 0.25], ["json-pretty", "json-compact", "toon"], strict=True):
            cells = [lookup["upstream", s, thinking, arm] for s in ["ordinary", "all"]]
            values = [100 * r["score"] for r in cells]
            bars = ax.bar(np.arange(2) + offset, values, 0.23, color=COLORS[arm], label=LABELS[arm])
            ax.bar_label(bars, fmt="%.1f", fontsize=8, padding=3)
        ax.set_xticks([0, 1], ["Ordinary comprehension\n203 questions", "Full upstream suite\n244 questions"])
        ax.set_ylim(0, 110)
        ax.set_title("Thinking " + ("on" if thinking else "off"))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Correct answers (%)")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Upstream TOON benchmark reproduced on Qwen3.8-27B", fontsize=14)
    fig.supxlabel(
        "Same short-answer output · same questions per format · point estimates; paired intervals in report", fontsize=9
    )
    save(fig, "01_upstream_accuracy")
    families = ["inventory", "customers", "services", "logs", "orders"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, thinking in zip(axes, [False, True], strict=True):
        for offset, arm in [(-0.2, "json-compact"), (0.2, "toon")]:
            vals = [100 * lookup["tools", "dataset:" + f, thinking, arm]["score"] for f in families]
            bars = ax.bar(np.arange(5) + offset, vals, 0.37, color=COLORS[arm], label=LABELS[arm])
            ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3)
        ax.set_xticks(
            np.arange(5), ["Inventory", "Nested\ncustomers", "Keyed\nservices", "Mixed\nlogs", "Nested\norders"]
        )
        ax.set_ylim(0, 110)
        ax.set_title("Thinking " + ("on" if thinking else "off"))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Correct answers before fence handling (%)")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("Tool-result answers before JSON fence handling", fontsize=14)
    fig.supxlabel("100 independent synthetic datasets per family · primary seed only · errors included", fontsize=9)
    save(fig, "02_tool_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True, layout="constrained")
    for ax, thinking in zip(axes, [False, True], strict=True):
        rows = [
            next(r for r in paired if r["track"] == "tools" and r["slice"] == f and r["thinking"] == thinking)
            for f in families
        ]
        for offset, key, color, label in [
            (-0.18, "prompt_token_ratio", "#7c3aed", "Prompt tokens"),
            (0.18, "total_token_ratio", "#d97706", "Total tokens"),
        ]:
            vals = [100 * (r[key] - 1) for r in rows]
            bars = ax.barh(np.arange(5) + offset, vals, 0.33, color=color, label=label)
            ax.bar_label(bars, fmt="%+.1f%%", fontsize=8, padding=3)
        ax.set_yticks(np.arange(5), families)
        ax.set_title("Thinking " + ("on" if thinking else "off"))
        ax.axvline(0, color="#64748b", lw=1)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.margins(x=0.2)
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False)
    fig.suptitle("TOON token change relative to compact JSON", fontsize=14)
    fig.supxlabel(
        "Negative = fewer tokens · total includes reasoning and failed answers · fixed JSON output", fontsize=9
    )
    save(fig, "03_tool_tokens")
    parsed = [json.loads(line) for line in (OUT / "parser_sensitivity.jsonl").read_text().splitlines()]
    parsed = [r for r in parsed if r["repeat"] == 0]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, thinking in zip(axes, [False, True], strict=True):
        for arm in ["json-compact", "toon"]:
            sizes = [10, 30, 100, 300]
            ys = [
                100
                * np.mean(
                    [
                        r["dspy_score"]
                        for r in parsed
                        if r["rows"] == n and r["thinking"] == thinking and r["adapter"] == arm
                    ]
                )
                for n in sizes
            ]
            ax.plot(sizes, ys, marker="o", color=COLORS[arm], label=LABELS[arm])
        ax.set_xscale("log")
        ax.set_xticks(sizes, [str(n) for n in sizes])
        ax.set_ylim(0, 105)
        ax.set_title("Thinking " + ("on" if thinking else "off"))
        ax.set_xlabel("Records per tool-result dataset")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Semantically correct answers (%)")
    axes[1].legend(frameon=False)
    fig.suptitle("Answer quality by input size, after DSPy JSON parsing", fontsize=14)
    fig.supxlabel(
        "125 independent synthetic datasets per size · five families and balanced task types · primary seed", fontsize=9
    )
    save(fig, "04_size_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, layout="constrained")
    for ax, thinking in zip(axes, [False, True], strict=True):
        for offset, arm in [(-0.2, "json-compact"), (0.2, "toon")]:
            values = [
                100
                * np.mean(
                    [
                        r["dspy_score"]
                        for r in parsed
                        if r["dataset"] == family and r["thinking"] == thinking and r["adapter"] == arm
                    ]
                )
                for family in families
            ]
            bars = ax.bar(np.arange(5) + offset, values, 0.37, color=COLORS[arm], label=LABELS[arm])
            ax.bar_label(bars, fmt="%.0f", fontsize=8, padding=3)
        ax.set_xticks(
            np.arange(5), ["Inventory", "Nested\ncustomers", "Keyed\nservices", "Mixed\nlogs", "Nested\norders"]
        )
        ax.set_ylim(0, 110)
        ax.set_title("Thinking " + ("on" if thinking else "off"))
        ax.grid(axis="y", alpha=0.15)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Correct answers after DSPy JSON parsing (%)")
    axes[1].legend(frameon=False, loc="lower left")
    fig.suptitle("TOON input with JSON output: data shape matters", fontsize=14)
    fig.supxlabel(
        "Post-hoc parser sensitivity · 100 independent datasets per family · primary seed · same saved responses",
        fontsize=9,
    )
    save(fig, "05_dspy_tool_accuracy")
    print("Saved five figures as editable SVG and 220-DPI PNG")


if __name__ == "__main__":
    main()
