#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Math 24 (24s game) optimization benchmark comparing adapter performance with GEPA.

Evaluates how adapters affect prompt optimization on the Math Twenty Four dataset
(`nlile/24-game`) where a model must output whether a puzzle is solvable and
provide expressions that evaluate to 24 using the four given numbers.

Usage:
    python -m benchmarks.math24_optimization --model gemini/gemini-2.5-flash-lite
"""

from __future__ import annotations

import argparse
import json
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy

from benchmarks.baml_adapter import BAMLAdapter
from datasets import load_dataset
from dspy_toon import ToonAdapter

# =============================================================================
# Answer Signature
# =============================================================================


class SolveMath24(dspy.Signature):
    """Generate Math 24 solutions and solvability flag."""

    numbers: list[int] = dspy.InputField(desc="Four numbers that may form 24")
    solutions: list[str] = dspy.OutputField(desc="Valid expressions that evaluate to 24")
    solvable: bool = dspy.OutputField(desc="Whether any solution exists")


# =============================================================================
# Dataset Loading
# =============================================================================


def load_math24_splits(
    n_total: int = 100,
    solvable_ratio: float = 0.8,
    train_ratio: float = 0.6,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load a balanced Math 24 sample and split into train/val.

    - Sample `n_total` puzzles with `solvable_ratio` portion solvable
    - Split sampled data into train/val by `train_ratio`
    """

    ds = load_dataset("nlile/24-game")
    data = list(ds["train"])

    solvable_items = [item for item in data if item.get("solvable", False)]
    unsolvable_items = [item for item in data if not item.get("solvable", False)]

    random.seed(seed)
    n_solvable_target = min(int(n_total * solvable_ratio), len(solvable_items))
    n_unsolvable_target = n_total - n_solvable_target

    if n_unsolvable_target > len(unsolvable_items):
        n_unsolvable_target = len(unsolvable_items)
        n_solvable_target = min(n_total - n_unsolvable_target, len(solvable_items))

    sampled = random.sample(solvable_items, n_solvable_target) + random.sample(unsolvable_items, n_unsolvable_target)
    random.shuffle(sampled)

    train_cut = max(1, min(len(sampled) - 1, int(len(sampled) * train_ratio)))
    train_data = sampled[:train_cut]
    val_data = sampled[train_cut:]

    def convert_to_examples(items: list[dict[str, Any]]) -> list[dspy.Example]:
        examples = []
        for item in items:
            example = dspy.Example(
                numbers=item.get("numbers", []),
                solutions=item.get("solutions", []),
                solvable=bool(item.get("solvable", False)),
                amt=item.get("amt"),
                solved_rate=item.get("solved_rate"),
                mean_time=item.get("mean_time"),
                std_time=item.get("std_time"),
            ).with_inputs("numbers")
            examples.append(example)
        return examples

    return convert_to_examples(train_data), convert_to_examples(val_data)


# =============================================================================
# Metrics
# =============================================================================


def _normalize_solution(expr: str) -> str:
    """Normalize mathematical expressions for comparison by standardizing operators and removing whitespace."""
    normalized = expr.replace("×", "*").replace("X", "*").replace("x", "*")
    normalized = re.sub(r"\s+", "", normalized)
    return normalized


def _parse_solutions(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(s) for s in raw if str(s).strip()]
    if isinstance(raw, str):
        parts = re.split(r"[\n;,]+", raw)
        return [p.strip() for p in parts if p.strip()]
    return [str(raw)]


def _extract_prediction_fields(prediction: dspy.Prediction | Any) -> tuple[list[str], bool]:
    solutions = []
    solvable = False

    if hasattr(prediction, "solutions"):
        solutions = getattr(prediction, "solutions")
    elif hasattr(prediction, "answer") and hasattr(prediction.answer, "solutions"):
        solutions = getattr(prediction.answer, "solutions")

    if hasattr(prediction, "solvable"):
        solvable = getattr(prediction, "solvable")
    elif hasattr(prediction, "answer") and hasattr(prediction.answer, "solvable"):
        solvable = getattr(prediction.answer, "solvable")

    # Handle string boolean cases
    if isinstance(solvable, str):
        solvable = solvable.strip().lower() in {"true", "yes", "1"}

    return _parse_solutions(solutions), bool(solvable)


def _math24_score(example: dspy.Example, prediction: dspy.Prediction | Any) -> tuple[float, str]:
    gold_solutions = [s for s in example.solutions or []]
    gold_set = {_normalize_solution(s) for s in gold_solutions}
    pred_solutions, pred_solvable = _extract_prediction_fields(prediction)
    pred_set = [_normalize_solution(s) for s in pred_solutions]

    if not example.solvable:
        if not pred_solvable:
            return 1.0, "Correctly marked puzzle as unsolvable."
        return 0.0, "Puzzle is unsolvable but prediction marked it solvable."

    if not pred_solvable:
        return 0.0, "Puzzle is solvable but prediction marked it unsolvable."

    if not gold_set:
        return 0.0, "Gold solutions missing; cannot evaluate."

    correct = sum(1 for s in pred_set if s in gold_set)
    incorrect = len(pred_set) - correct

    if incorrect == 0 and correct == len(gold_set):
        return 1.0, "All solutions correct."

    if incorrect == 0 and correct > 0:
        score = correct / len(gold_set)
        missing = len(gold_set) - correct
        return score, f"All provided solutions correct but missing {missing}."

    score = max((correct - incorrect) / len(gold_set), 0.0)
    missing = len(gold_set) - correct
    feedback = f"{correct} correct, {incorrect} incorrect, {missing} missing."
    return score, feedback


def math24_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> float:
    """Metric used for evaluation; returns score in [0, 1]."""
    score, _ = _math24_score(example, prediction)
    return score


def math24_metric_with_feedback(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace: list | None = None,
) -> dspy.Prediction:
    """Metric with feedback for GEPA optimization."""
    score, feedback = _math24_score(example, prediction)
    return dspy.Prediction(score=score, feedback=feedback)


# =============================================================================
# Results Storage
# =============================================================================


@dataclass
class EvaluationResult:
    """Results of a single evaluation run."""

    accuracy: float = 0.0
    total_correct: int = 0
    total_questions: int = 0


@dataclass
class AdapterOptimizationResult:
    """Results for a single adapter's optimization."""

    adapter_name: str
    baseline: EvaluationResult = field(default_factory=EvaluationResult)
    optimized: EvaluationResult = field(default_factory=EvaluationResult)
    accuracy_boost: float = 0.0
    optimization_time_seconds: float = 0.0
    optimized_program_path: str | None = None


@dataclass
class OptimizationBenchmarkResults:
    """Complete optimization benchmark results."""

    model: str
    reflection_model: str
    timestamp: str
    n_train: int
    n_val: int
    adapter_results: list[AdapterOptimizationResult] = field(default_factory=list)


# =============================================================================
# Benchmark Runner
# =============================================================================


def get_adapters() -> dict[str, Any]:
    """Get all adapters to benchmark."""
    return {
        "ToonAdapter": ToonAdapter(),
        "ChatAdapter": dspy.ChatAdapter(),
        "JSONAdapter": dspy.JSONAdapter(),
        "BAMLAdapter": BAMLAdapter(),
    }


def run_optimization_benchmark(
    model: str = "gemini/gemini-2.5-flash-lite",
    reflection_model: str = "gemini/gemini-2.5-pro",
    n_total: int = 100,
    solvable_ratio: float = 0.8,
    train_ratio: float = 0.6,
    num_threads: int = 4,
    seed: int = 42,
    output_dir: str = "benchmark_results",
) -> OptimizationBenchmarkResults:
    """Run Math 24 optimization benchmark with GEPA for all adapters."""
    print(f"\n{'=' * 70}")
    print("MATH 24 OPTIMIZATION BENCHMARK (GEPA)")
    print(f"Model: {model}")
    print(f"Reflection Model: {reflection_model}")
    print("=" * 70)

    print("\nLoading Math 24 dataset...")
    trainset, valset = load_math24_splits(
        n_total=n_total,
        solvable_ratio=solvable_ratio,
        train_ratio=train_ratio,
        seed=seed,
    )
    print(f"Train set: {len(trainset)} examples")
    print(f"Val set: {len(valset)} examples")

    lm = dspy.LM(model, temperature=0.0, cache=True)
    reflection_lm = dspy.LM(reflection_model, temperature=1.0, max_tokens=8000)

    adapters = get_adapters()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = OptimizationBenchmarkResults(
        model=model,
        reflection_model=reflection_model,
        timestamp=datetime.now().isoformat(),
        n_train=len(trainset),
        n_val=len(valset),
    )

    for adapter_name, adapter in adapters.items():
        print(f"\n{'=' * 60}")
        print(f"Optimizing {adapter_name}...")
        print("=" * 60)

        adapter_result = AdapterOptimizationResult(adapter_name=adapter_name)

        dspy.configure(lm=lm, adapter=adapter)
        baseline_program = dspy.Predict(SolveMath24)

        evaluator = dspy.Evaluate(
            devset=valset,
            metric=math24_metric,
            num_threads=num_threads,
            display_progress=True,
        )

        print(f"\n  Evaluating baseline {adapter_name}...")
        baseline_eval = evaluator(baseline_program)
        adapter_result.baseline = EvaluationResult(
            accuracy=baseline_eval.score,
            total_correct=int(baseline_eval.score * len(valset) / 100),
            total_questions=len(valset),
        )
        print(f"  Baseline score: {adapter_result.baseline.accuracy:.2f}%")

        print(f"\n  Running GEPA optimization for {adapter_name}...")
        start_time = datetime.now()

        try:
            optimizer = dspy.GEPA(
                metric=math24_metric_with_feedback,
                auto="light",
                num_threads=num_threads,
                track_stats=True,
                reflection_lm=reflection_lm,
            )

            optimized_program = optimizer.compile(
                student=dspy.Predict(SolveMath24),
                trainset=trainset,
                valset=valset,
            )

            adapter_result.optimization_time_seconds = (datetime.now() - start_time).total_seconds()
            print(f"  Optimization completed in {adapter_result.optimization_time_seconds:.1f}s")

            program_file = output_path / f"optimized_{adapter_name}_math24_{timestamp}.json"
            optimized_program.save(str(program_file))
            adapter_result.optimized_program_path = str(program_file)
            print(f"  Saved optimized program to: {program_file}")

            print(f"\n  Evaluating optimized {adapter_name}...")
            optimized_eval = evaluator(optimized_program)
            adapter_result.optimized = EvaluationResult(
                accuracy=optimized_eval.score,
                total_correct=int(optimized_eval.score * len(valset) / 100),
                total_questions=len(valset),
            )
            print(f"  Optimized score: {adapter_result.optimized.accuracy:.2f}%")

        except Exception as e:
            print(f"  GEPA optimization failed: {e}")
            adapter_result.optimization_time_seconds = (datetime.now() - start_time).total_seconds()
            adapter_result.optimized = adapter_result.baseline

        adapter_result.accuracy_boost = adapter_result.optimized.accuracy - adapter_result.baseline.accuracy
        print(f"\n  Score boost: {adapter_result.accuracy_boost:+.2f}%")

        results.adapter_results.append(adapter_result)

    return results


def print_results(results: OptimizationBenchmarkResults) -> None:
    """Print optimization benchmark results summary."""
    print("\n" + "=" * 90)
    print("OPTIMIZATION BENCHMARK SUMMARY")
    print("=" * 90)
    print(f"Model: {results.model}")
    print(f"Reflection Model: {results.reflection_model}")
    print(f"Train/Val: {results.n_train}/{results.n_val}")

    sorted_results = sorted(results.adapter_results, key=lambda x: x.accuracy_boost, reverse=True)

    print(f"\n{'Adapter':<15} {'Baseline':>12} {'Optimized':>12} {'Boost':>12} {'Opt Time':>12}")
    print("-" * 65)

    best_boost = sorted_results[0].accuracy_boost if sorted_results else 0
    best_optimized = max((r.optimized.accuracy for r in sorted_results), default=0)

    for r in sorted_results:
        boost_marker = " *" if r.accuracy_boost == best_boost else ""
        opt_marker = " ^" if r.optimized.accuracy == best_optimized else ""
        print(
            f"{r.adapter_name:<15} "
            f"{r.baseline.accuracy:>11.2f}% "
            f"{r.optimized.accuracy:>11.2f}%{opt_marker} "
            f"{r.accuracy_boost:>+11.2f}%{boost_marker} "
            f"{r.optimization_time_seconds:>10.1f}s"
        )

    print("-" * 65)
    print("* = Best boost, ^ = Best optimized score")

    print("\n" + "=" * 90)
    print("DETAILED COMPARISON")
    print("=" * 90)

    for r in sorted_results:
        print(f"\n{r.adapter_name}:")
        print(f"  Baseline:  {r.baseline.accuracy:.2f}% ({r.baseline.total_correct}/{r.baseline.total_questions})")
        print(f"  Optimized: {r.optimized.accuracy:.2f}% ({r.optimized.total_correct}/{r.optimized.total_questions})")
        print(f"  Boost:     {r.accuracy_boost:+.2f}%")
        print(f"  Optimization time: {r.optimization_time_seconds:.1f}s")
        if r.optimized_program_path:
            print(f"  Saved to: {r.optimized_program_path}")


def save_results(results: OptimizationBenchmarkResults, output_dir: str = "benchmark_results") -> None:
    """Save results to JSON file."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    adapters_data: list[dict[str, Any]] = []
    data: dict[str, Any] = {
        "model": results.model,
        "reflection_model": results.reflection_model,
        "timestamp": results.timestamp,
        "n_train": results.n_train,
        "n_val": results.n_val,
        "adapters": adapters_data,
    }

    for r in results.adapter_results:
        adapters_data.append(
            {
                "name": r.adapter_name,
                "baseline": {
                    "accuracy": r.baseline.accuracy,
                    "total_correct": r.baseline.total_correct,
                    "total_questions": r.baseline.total_questions,
                },
                "optimized": {
                    "accuracy": r.optimized.accuracy,
                    "total_correct": r.optimized.total_correct,
                    "total_questions": r.optimized.total_questions,
                },
                "accuracy_boost": r.accuracy_boost,
                "optimization_time_seconds": r.optimization_time_seconds,
                "optimized_program_path": r.optimized_program_path,
            }
        )

    summary_file = output_path / f"math24_optimization_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Math 24 optimization benchmark with GEPA")
    parser.add_argument(
        "--model",
        default="gemini/gemini-flash-latest",
        help="Model to optimize",
    )
    parser.add_argument(
        "--reflection-model",
        default="gemini/gemini-3-pro-preview",
        help="Model for GEPA reflection",
    )
    parser.add_argument(
        "--n-total",
        type=int,
        default=100,
        help="Total number of sampled puzzles",
    )
    parser.add_argument(
        "--solvable-ratio",
        type=float,
        default=0.8,
        help="Target fraction of solvable puzzles in the sample",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.6,
        help="Fraction of sampled data used for training (rest for validation)",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=4,
        help="Number of threads for GEPA evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    results = run_optimization_benchmark(
        model=args.model,
        reflection_model=args.reflection_model,
        n_total=args.n_total,
        solvable_ratio=args.solvable_ratio,
        train_ratio=args.train_ratio,
        num_threads=args.num_threads,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print_results(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
