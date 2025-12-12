#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Table analysis optimization benchmark comparing adapter performance with GEPA.

Tests how different adapters affect prompt optimization using GEPA optimizer
for structured table metadata extraction. Compares baseline vs optimized accuracy
across ToonAdapter, JSONAdapter, ChatAdapter, and BAMLAdapter.

Usage:
    python -m benchmarks.table_analysis_optimization --model gemini/gemini-2.5-flash-lite
"""

import argparse
import csv
import json
import random
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import dspy
from pydantic import BaseModel

from benchmarks.baml_adapter import BAMLAdapter
from dspy_toon import ToonAdapter

# =============================================================================
# Answer Model and Signature
# =============================================================================


class TableMetadata(BaseModel):
    """Structured metadata extracted from a CSV table."""

    num_rows: int
    num_columns: int
    column_types: dict[str, str]  # "str", "int", or "float"
    column_max: dict[str, float | None]
    column_min: dict[str, float | None]
    identifier_first: str | None
    identifier_last: str | None
    identifier_shortest: str | None


class AnalyzeTable(dspy.Signature):
    """Extract structured metadata from a CSV table.

    You are given a CSV-like string representation of a table (with header row, no index).
    Extract a structure object following the provided response format class. Do not guess:
    if a value does not exist or is not applicable, return null.
    Count rows excluding the header. Infer each column type as 'str', 'int', or 'float'.
    For string columns, set min/max to null. If the 'Identifier' column is missing,
    set all Identifier-related fields to null. For null/None entries, set string columns to '',
    and numerical to None.
    Return only the structured object.
    """

    table: str = dspy.InputField(desc="CSV-like string representation of a table with header row")
    metadata: TableMetadata = dspy.OutputField(desc="Extracted table metadata")


# =============================================================================
# Dataset Loading
# =============================================================================


def load_table_analysis_dataset(
    csv_path: str | Path,
    n_train: int = 34,
    n_val: int = 66,
    seed: int = 42,
) -> tuple[list[dspy.Example], list[dspy.Example]]:
    """Load table analysis dataset with separate train and validation splits.

    Args:
        csv_path: Path to the CSV file containing table data
        n_train: Number of samples for training
        n_val: Number of samples for validation
        seed: Random seed for reproducibility

    Returns:
        Tuple of (trainset, valset) as lists of dspy.Example objects
    """
    random.seed(seed)
    csv_path = Path(csv_path)

    # Load all data
    all_examples = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            table = row["table"]
            ground_truth_str = row["ground_truth"]
            ground_truth = json.loads(ground_truth_str)

            example = dspy.Example(
                table=table,
                metadata=ground_truth,
            ).with_inputs("table")
            all_examples.append(example)

    # Shuffle and split
    random.shuffle(all_examples)

    # Ensure we don't exceed available data
    total_available = len(all_examples)

    # Adjust split if needed to ensure we have validation examples
    n_train = int(total_available * 0.34)
    n_val = total_available - n_train

    trainset = all_examples[:n_train]
    valset = all_examples[n_train : n_train + n_val]

    return trainset, valset


# =============================================================================
# Metrics
# =============================================================================


def _normalize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Normalize metadata for comparison (handle type consistency, etc.)."""
    normalized = metadata.copy()

    # Normalize column_max/min: ensure None for string columns
    if "column_types" in normalized and "column_max" in normalized:
        for col_name, col_type in normalized["column_types"].items():
            if col_type == "str":
                normalized["column_max"][col_name] = None
                normalized["column_min"][col_name] = None

    # Note: identifier fields can be None (no Identifier column) or "" (empty Identifier column)
    # Both are valid and should be compared as-is

    return normalized


def _metadata_match(pred: dict[str, Any], gold: dict[str, Any]) -> bool:
    """Check if predicted metadata matches ground truth."""
    pred_norm = _normalize_metadata(pred)
    gold_norm = _normalize_metadata(gold)

    # Compare all fields
    for key in gold_norm:
        if key not in pred_norm:
            return False

        if key in ["column_types", "column_max", "column_min"]:
            # Compare dictionaries - ensure all keys match
            if set(pred_norm[key].keys()) != set(gold_norm[key].keys()):
                return False
            # Compare values for each key
            for col_key in gold_norm[key]:
                if pred_norm[key].get(col_key) != gold_norm[key][col_key]:
                    return False
        else:
            # Compare directly
            if pred_norm[key] != gold_norm[key]:
                return False

    return True


def accuracy_metric(example: dspy.Example, prediction: dspy.Prediction, trace=None) -> bool:
    """Check if predicted metadata matches ground truth."""
    try:
        pred_metadata = prediction.metadata
        if isinstance(pred_metadata, BaseModel):
            pred_dict = pred_metadata.model_dump()
        else:
            pred_dict = pred_metadata

        gold_dict = example.metadata
        return _metadata_match(pred_dict, gold_dict)
    except Exception:
        return False


def accuracy_metric_with_feedback(
    example: dspy.Example,
    prediction: dspy.Prediction,
    trace=None,
    pred_name: str | None = None,
    pred_trace: list | None = None,
) -> dspy.Prediction:
    try:
        pred_metadata = prediction.metadata
        if isinstance(pred_metadata, BaseModel):
            pred_dict = pred_metadata.model_dump()
        else:
            pred_dict = pred_metadata

        gold_dict = example.metadata
        is_correct = _metadata_match(pred_dict, gold_dict)

        if is_correct:
            return dspy.Prediction(
                score=1.0,
                feedback="Correct! All metadata fields match the expected values.",
            )
        else:
            # Find differences for feedback
            differences = []
            pred_norm = _normalize_metadata(pred_dict)
            gold_norm = _normalize_metadata(gold_dict)

            for key in gold_norm:
                if key not in pred_norm:
                    differences.append(f"Missing field: {key}")
                elif pred_norm[key] != gold_norm[key]:
                    differences.append(f"{key}: expected {gold_norm[key]}, got {pred_norm[key]}")

            return dspy.Prediction(
                score=0.0,
                feedback=f"Incorrect. Differences: {', '.join(differences)}. "
                f"Review the table structure and extraction rules carefully.",
            )
    except Exception as e:
        return dspy.Prediction(score=0.0, feedback=f"Error parsing metadata: {e}")


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
    dataset_path: str
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
    dataset_path: str = "datasets/data_table_analysis.csv",
    model: str = "gemini/gemini-2.5-flash-lite",
    reflection_model: str = "gemini/gemini-2.5-pro",
    n_train: int = 34,
    n_val: int = 66,
    num_threads: int = 4,
    seed: int = 42,
    output_dir: str = "benchmark_results",
) -> OptimizationBenchmarkResults:
    """Run table analysis optimization benchmark with GEPA for all adapters."""
    print(f"\n{'=' * 70}")
    print("TABLE ANALYSIS OPTIMIZATION BENCHMARK (GEPA)")
    print(f"Model: {model}")
    print(f"Reflection Model: {reflection_model}")
    print("=" * 70)

    # Load dataset
    print(f"\nLoading table analysis dataset from {dataset_path}...")
    trainset, valset = load_table_analysis_dataset(dataset_path, n_train, n_val, seed)
    print(f"Train set: {len(trainset)} examples")
    print(f"Val set: {len(valset)} examples")

    # Initialize LMs
    lm = dspy.LM(model, temperature=0.0, cache=True, max_tokens=8000)
    reflection_lm = dspy.LM(reflection_model, temperature=1.0, max_tokens=8000)

    adapters = get_adapters()
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results = OptimizationBenchmarkResults(
        model=model,
        reflection_model=reflection_model,
        timestamp=datetime.now().isoformat(),
        dataset_path=str(dataset_path),
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
        baseline_program = dspy.Predict(AnalyzeTable)

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
                student=dspy.Predict(AnalyzeTable),
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
            import traceback

            traceback.print_exc()
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
    print(f"Dataset: {results.dataset_path}")
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
        "dataset_path": results.dataset_path,
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

    summary_file = output_path / f"table_analysis_optimization_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to: {summary_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Table analysis optimization benchmark with GEPA")
    parser.add_argument(
        "--dataset-path",
        default="datasets/data_table_analysis.csv",
        help="Path to the table analysis CSV dataset",
    )
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
        default=70,
        help="Number of training samples",
    )
    parser.add_argument(
        "--n-val",
        type=int,
        default=30,
        help="Number of validation samples",
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
        dataset_path=args.dataset_path,
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
