#!/usr/bin/env python3
"""Evaluate BAMLAdapter on table analysis with different DSPy programs.

Runs the table analysis benchmark on three DSPy programs:
- Unoptimized baseline (fresh Predict(AnalyzeTable) with BAMLAdapter)
- Optimized program saved for BAMLAdapter
- Optimized program saved for ToonAdapter (run with BAMLAdapter)

Example:
    python -m benchmarks.baml_adapter_program_eval \
      --dataset-path adapters/dspy-toon/datasets/data_table_analysis.csv \
      --model gemini/gemini-2.5-flash-lite
"""

from __future__ import annotations

import argparse
from typing import Any

import dspy

from benchmarks.baml_adapter import BAMLAdapter
from benchmarks.table_analysis_optimization import (
    AnalyzeTable,
    EvaluationResult,
    accuracy_metric,
    load_table_analysis_dataset,
)


def evaluate_program(program: Any, valset: list[dspy.Example], num_threads: int) -> EvaluationResult:
    """Run evaluation on the validation set and return metrics."""
    evaluator = dspy.Evaluate(
        devset=valset,
        metric=accuracy_metric,
        num_threads=num_threads,
        display_progress=True,
    )
    result = evaluator(program)
    return EvaluationResult(
        accuracy=result.score,
        total_correct=int(result.score * len(valset) / 100),
        total_questions=len(valset),
    )


def load_program(path: str | None, lm: dspy.LM, adapter: Any) -> Any:
    """Load a saved DSPy program by recreating and calling .load(), then rebind LM/adapter."""
    if not path:
        return None
    program = dspy.Predict(AnalyzeTable)
    program.load(path)
    # Rebind to ensure we don't keep the adapter embedded in the saved program
    program.lm = lm
    program.adapter = adapter
    print(f"Loaded program from {path} with BAML adapter rebound")
    return program


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BAMLAdapter with provided DSPy programs")
    parser.add_argument(
        "--dataset-path",
        default="datasets/data_table_analysis.csv",
        help="Path to the table analysis CSV dataset",
    )
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-lite",
        help="Model to use for evaluation",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=70,
        help="Number of training samples (used only for splitting)",
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
        help="Number of threads for evaluation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for dataset split",
    )
    parser.add_argument(
        "--baml-program",
        default="benchmark_optimisation_table_results/optimized_BAMLAdapter_20251210_103010.json",
        help="Path to optimized BAMLAdapter program JSON",
    )
    parser.add_argument(
        "--toon-program",
        default="benchmark_optimisation_table_results/optimized_ToonAdapter_20251210_103010.json",
        help="Path to optimized ToonAdapter program JSON (evaluated with BAMLAdapter)",
    )
    args = parser.parse_args()

    # Load dataset
    trainset, valset = load_table_analysis_dataset(
        csv_path=args.dataset_path,
        n_train=args.n_train,
        n_val=args.n_val,
        seed=args.seed,
    )
    print(f"Loaded dataset with {len(trainset)} train and {len(valset)} val examples")

    # Configure LM and adapter once; reused for all programs
    lm = dspy.LM(args.model, temperature=0.0, cache=True, max_tokens=8000)
    baml_adapter = BAMLAdapter()

    programs = [
        ("unoptimized_baseline", None),
        ("optimized_BAMLAdapter", args.baml_program),
        ("optimized_ToonAdapter", args.toon_program),
    ]

    results: list[tuple[str, EvaluationResult]] = []

    for name, path in programs:
        print("\n" + "=" * 70)
        print(f"Evaluating program: {name}")
        print("=" * 70)

        # Configure adapter + LM for each run to be explicit
        dspy.configure(lm=lm, adapter=baml_adapter)

        program = load_program(path, lm=lm, adapter=baml_adapter)
        if program is None:
            program = dspy.Predict(AnalyzeTable)
            print("Using fresh Predict(AnalyzeTable) as unoptimized baseline")

        eval_result = evaluate_program(program, valset, args.num_threads)
        results.append((name, eval_result))

        print(f"Accuracy: {eval_result.accuracy:.2f}% ({eval_result.total_correct}/{eval_result.total_questions})")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY (BAMLAdapter)")
    print("=" * 90)
    for name, res in results:
        print(f"{name:<25} {res.accuracy:>7.2f}% ({res.total_correct}/{res.total_questions})")


if __name__ == "__main__":
    main()

