#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""MMLU-Pro optimization benchmark comparing adapter performance with GEPA.

Tests how different adapters affect prompt optimization using GEPA optimizer.
Compares baseline vs optimized accuracy across ToonAdapter, JSONAdapter,
ChatAdapter, and BAMLAdapter.

Usage:
    python -m benchmarks.mmlu_pro_optimization --model gemini/gemini-2.5-flash-lite
"""

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import dspy
from datasets import load_dataset
from pydantic import BaseModel

from benchmarks.baml_adapter import BAMLAdapter
from dspy_toon import ToonAdapter

# =============================================================================
# Answer Model and Signature
# =============================================================================


class MCQAnswer(BaseModel):
    answer: Literal["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]


class AnswerMCQ(dspy.Signature):
    """Answer a multiple choice question by selecting the correct option."""

    question: str = dspy.InputField()
    options: str = dspy.InputField()
    answer: MCQAnswer = dspy.OutputField()


# =============================================================================
# Dataset Loading
# =============================================================================


def load_mmlu_pro_splits(
    n_train: int = 100,
    n_val: int = 70,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load MMLU-Pro with separate train and validation splits.

    Args:
        n_train: Number of samples from test split for training
        n_val: Number of samples from validation split for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (trainset, valset) as lists of dspy.Example objects
    """
    random.seed(seed)
    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    def convert_to_examples(items: list[dict[str, Any]]) -> list[dspy.Example]:
        examples = []
        for item in items:
            options_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(item["options"]))
            example = dspy.Example(
                question=item["question"],
                options=options_str,
                answer=item["answer"],
                category=item["category"],
                question_id=item["question_id"],
            ).with_inputs("question", "options")
            examples.append(example)
        return examples

    # Train set: stratified sample from test split
    test_data = list(ds["test"])
    categories_test = defaultdict(list)
    for item in test_data:
        categories_test[item["category"]].append(item)

    n_categories = len(categories_test)
    samples_per_category = n_train // n_categories
    remaining = n_train % n_categories

    train_samples = []
    for i, (category, items) in enumerate(sorted(categories_test.items())):
        n_sample = samples_per_category + (1 if i < remaining else 0)
        n_sample = min(n_sample, len(items))
        train_samples.extend(random.sample(items, n_sample))

    trainset = convert_to_examples(train_samples)

    # Validation set: from validation split
    val_data = list(ds["validation"])
    val_samples = val_data[:n_val] if len(val_data) <= n_val else random.sample(val_data, n_val)
    valset = convert_to_examples(val_samples)

    return trainset, valset


# =============================================================================
# Metrics
# =============================================================================


def accuracy_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Check if predicted answer matches ground truth."""
    try:
        pred_answer = prediction.answer.answer if hasattr(prediction.answer, "answer") else prediction.answer
        return pred_answer == example.answer
    except Exception:
        return False


def accuracy_metric_with_feedback(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace: list | None = None,
) -> dspy.Prediction:
    """Metric with feedback for GEPA optimization.

    GEPA requires 5 arguments: (gold, pred, trace, pred_name, pred_trace).
    Returns dspy.Prediction with score and feedback.
    """
    try:
        pred_answer = prediction.answer.answer if hasattr(prediction.answer, "answer") else prediction.answer
        if pred_answer == example.answer:
            return dspy.Prediction(score=1.0, feedback=f"Correct! The answer is {example.answer}.")
        else:
            return dspy.Prediction(
                score=0.0,
                feedback=f"Incorrect. Expected {example.answer}, got {pred_answer}. "
                f"Review the question and options carefully.",
            )
    except Exception as e:
        return dspy.Prediction(score=0.0, feedback=f"Error parsing answer: {e}")


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
        "ChatAdapter": dspy.ChatAdapter(),
        "JSONAdapter": dspy.JSONAdapter(),
        "BAMLAdapter": BAMLAdapter(),
        "ToonAdapter": ToonAdapter(),
    }


def run_optimization_benchmark(
    model: str = "gemini/gemini-2.5-flash-lite",
    reflection_model: str = "gemini/gemini-2.5-pro",
    n_train: int = 100,
    n_val: int = 70,
    num_threads: int = 4,
    seed: int = 42,
    output_dir: str = "benchmark_results",
) -> OptimizationBenchmarkResults:
    """Run MMLU-Pro optimization benchmark with GEPA for all adapters."""
    print(f"\n{'=' * 70}")
    print("MMLU-PRO OPTIMIZATION BENCHMARK (GEPA)")
    print(f"Model: {model}")
    print(f"Reflection Model: {reflection_model}")
    print("=" * 70)

    # Load dataset
    print("\nLoading MMLU-Pro dataset...")
    trainset, valset = load_mmlu_pro_splits(n_train, n_val, seed)
    print(f"Train set: {len(trainset)} examples (from test split)")
    print(f"Val set: {len(valset)} examples (from validation split)")

    # Initialize LMs
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
        n_train=n_train,
        n_val=n_val,
    )

    for adapter_name, adapter in adapters.items():
        print(f"\n{'=' * 60}")
        print(f"Optimizing {adapter_name}...")
        print("=" * 60)

        adapter_result = AdapterOptimizationResult(adapter_name=adapter_name)

        # Create baseline program
        dspy.configure(lm=lm, adapter=adapter)
        baseline_program = dspy.Predict(AnswerMCQ)

        # Create evaluator
        evaluator = dspy.Evaluate(
            devset=valset,
            metric=accuracy_metric,
            num_threads=num_threads,
            display_progress=True,
        )

        # Evaluate baseline
        print(f"\n  Evaluating baseline {adapter_name}...")
        baseline_eval = evaluator(baseline_program)
        adapter_result.baseline = EvaluationResult(
            accuracy=baseline_eval.score,
            total_correct=int(baseline_eval.score * len(valset) / 100),
            total_questions=len(valset),
        )
        print(f"  Baseline accuracy: {adapter_result.baseline.accuracy:.2f}%")

        # Optimize with GEPA
        print(f"\n  Running GEPA optimization for {adapter_name}...")
        start_time = datetime.now()

        try:
            optimizer = dspy.GEPA(
                metric=accuracy_metric_with_feedback,
                auto="light",
                num_threads=num_threads,
                track_stats=True,
                reflection_lm=reflection_lm,
            )

            optimized_program = optimizer.compile(
                student=dspy.Predict(AnswerMCQ),
                trainset=trainset,
                valset=valset,
            )

            adapter_result.optimization_time_seconds = (datetime.now() - start_time).total_seconds()
            print(f"  Optimization completed in {adapter_result.optimization_time_seconds:.1f}s")

            # Save optimized program
            program_file = output_path / f"optimized_{adapter_name}_{timestamp}.json"
            optimized_program.save(str(program_file))
            adapter_result.optimized_program_path = str(program_file)
            print(f"  Saved optimized program to: {program_file}")

            # Evaluate optimized program
            print(f"\n  Evaluating optimized {adapter_name}...")
            optimized_eval = evaluator(optimized_program)
            adapter_result.optimized = EvaluationResult(
                accuracy=optimized_eval.score,
                total_correct=int(optimized_eval.score * len(valset) / 100),
                total_questions=len(valset),
            )
            print(f"  Optimized accuracy: {adapter_result.optimized.accuracy:.2f}%")

        except Exception as e:
            print(f"  GEPA optimization failed: {e}")
            adapter_result.optimization_time_seconds = (datetime.now() - start_time).total_seconds()
            adapter_result.optimized = adapter_result.baseline

        # Calculate boost
        adapter_result.accuracy_boost = adapter_result.optimized.accuracy - adapter_result.baseline.accuracy
        print(f"\n  Accuracy boost: {adapter_result.accuracy_boost:+.2f}%")

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

    # Sort by boost descending
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
    print("* = Best boost, ^ = Best optimized accuracy")

    # Detailed comparison table
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

    summary_file = output_path / f"mmlu_pro_optimization_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MMLU-Pro optimization benchmark with GEPA")
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-lite",
        help="Model to optimize",
    )
    parser.add_argument(
        "--reflection-model",
        default="gemini/gemini-2.5-pro",
        help="Model for GEPA reflection",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=100,
        help="Number of training samples (from test split)",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=70,
        help="Number of validation samples (from validation split)",
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
        n_train=args.n_train,
        n_val=args.n_val,
        num_threads=args.num_threads,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print_results(results)
    save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
