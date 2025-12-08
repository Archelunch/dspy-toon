#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""MMLU-Pro accuracy benchmark for adapter comparison.

Compares ToonAdapter against JSONAdapter, ChatAdapter, and BAMLAdapter
on the MMLU-Pro dataset measuring accuracy and token usage.

Usage:
    python -m benchmarks.mmlu_pro_accuracy --model gemini/gemini-2.5-flash-lite
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


def load_mmlu_pro_stratified(n_validation: int = 70, n_test: int = 130, seed: int = 42) -> list[dspy.Example]:
    """Load MMLU-Pro with stratified sampling across categories.

    Args:
        n_validation: Number of samples from validation split
        n_test: Number of samples from test split
        seed: Random seed for reproducibility

    Returns:
        List of dspy.Example objects
    """
    random.seed(seed)
    ds = load_dataset("TIGER-Lab/MMLU-Pro")

    examples = []

    # Process validation split
    val_data = list(ds["validation"])
    categories_val = defaultdict(list)
    for item in val_data:
        categories_val[item["category"]].append(item)

    # Sample from validation (take all if <= n_validation since val only has 70)
    val_samples = val_data[:n_validation] if len(val_data) <= n_validation else random.sample(val_data, n_validation)
    examples.extend(val_samples)

    # Process test split with stratified sampling
    test_data = list(ds["test"])
    categories_test = defaultdict(list)
    for item in test_data:
        categories_test[item["category"]].append(item)

    # Calculate samples per category for test
    n_categories = len(categories_test)
    samples_per_category = n_test // n_categories
    remaining = n_test % n_categories

    test_samples = []
    for i, (category, items) in enumerate(sorted(categories_test.items())):
        n_sample = samples_per_category + (1 if i < remaining else 0)
        n_sample = min(n_sample, len(items))
        test_samples.extend(random.sample(items, n_sample))

    examples.extend(test_samples)

    # Convert to dspy.Example format
    dspy_examples = []
    for item in examples:
        # Format options as string
        options_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(item["options"]))

        example = dspy.Example(
            question=item["question"],
            options=options_str,
            answer=item["answer"],  # Ground truth letter (A-J)
            category=item["category"],
            question_id=item["question_id"],
        ).with_inputs("question", "options")

        dspy_examples.append(example)

    return dspy_examples


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


# =============================================================================
# Results Storage
# =============================================================================


@dataclass
class PromptLog:
    """Log of a single prediction with prompt details."""

    question_id: str
    category: str
    question: str
    options: str
    ground_truth: str
    predicted: str | None
    correct: bool
    input_tokens: int
    output_tokens: int
    error: str | None = None
    raw_messages: list[dict[str, Any]] | None = None


@dataclass
class AdapterResult:
    """Results for a single adapter."""

    adapter_name: str
    accuracy: float = 0.0
    total_correct: int = 0
    total_questions: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_tokens: int = 0
    logs: list[PromptLog] = field(default_factory=list)
    category_accuracy: dict[str, float] = field(default_factory=dict)


@dataclass
class BenchmarkResults:
    """Complete benchmark results."""

    model: str
    timestamp: str
    n_questions: int
    adapter_results: list[AdapterResult] = field(default_factory=list)


# =============================================================================
# Benchmark Runner
# =============================================================================


def get_adapters() -> dict[str, Any]:
    """Get all adapters to benchmark."""
    return {
        "ToonAdapter": ToonAdapter(),
        "BAMLAdapter": BAMLAdapter(),
        "JSONAdapter": dspy.JSONAdapter(),
        "ChatAdapter": dspy.ChatAdapter(),
    }


def run_benchmark(
    model: str,
    n_validation: int = 70,
    n_test: int = 130,
    seed: int = 42,
) -> BenchmarkResults:
    """Run MMLU-Pro accuracy benchmark with all adapters."""
    print(f"\n{'=' * 70}")
    print("MMLU-PRO ACCURACY BENCHMARK")
    print(f"Model: {model}")
    print("=" * 70)

    # Load dataset
    print("\nLoading MMLU-Pro dataset...")
    examples = load_mmlu_pro_stratified(n_validation, n_test, seed)
    print(f"Loaded {len(examples)} questions")

    # Count categories
    category_counts: dict[str, int] = defaultdict(int)
    for ex in examples:
        category_counts[ex.category] += 1
    print(f"Categories: {dict(category_counts)}")

    # Initialize LM
    lm = dspy.LM(model, temperature=0.0, cache=True)
    adapters = get_adapters()

    results = BenchmarkResults(
        model=model,
        timestamp=datetime.now().isoformat(),
        n_questions=len(examples),
    )

    # Run each adapter
    for adapter_name, adapter in adapters.items():
        print(f"\n{'=' * 60}")
        print(f"Running {adapter_name}...")
        print("=" * 60)

        adapter_result = AdapterResult(adapter_name=adapter_name, total_questions=len(examples))
        category_correct: dict[str, int] = defaultdict(int)
        category_total: dict[str, int] = defaultdict(int)

        dspy.configure(lm=lm, adapter=adapter, track_usage=True)
        predictor = dspy.Predict(AnswerMCQ)

        for i, example in enumerate(examples):
            log = PromptLog(
                question_id=example.question_id,
                category=example.category,
                question=example.question,
                options=example.options,
                ground_truth=example.answer,
                predicted=None,
                correct=False,
                input_tokens=0,
                output_tokens=0,
            )

            try:
                prediction = predictor(question=example.question, options=example.options)

                # Extract answer
                pred_answer = (
                    prediction.answer.answer if hasattr(prediction.answer, "answer") else str(prediction.answer)
                )
                log.predicted = pred_answer
                log.correct = pred_answer == example.answer

                # Get token usage
                usage = prediction.get_lm_usage()
                if usage:
                    for lm_usage in usage.values():
                        log.input_tokens += lm_usage.get("prompt_tokens", 0)
                        log.output_tokens += lm_usage.get("completion_tokens", 0)

                # Capture raw messages from LM history
                if lm.history:
                    log.raw_messages = lm.history[-1].get("messages", []) + lm.history[-1].get("outputs", [])

                if log.correct:
                    adapter_result.total_correct += 1
                    category_correct[example.category] += 1

            except Exception as e:
                log.error = str(e)
                log.predicted = None

            category_total[example.category] += 1
            adapter_result.total_input_tokens += log.input_tokens
            adapter_result.total_output_tokens += log.output_tokens
            adapter_result.logs.append(log)

            # Progress
            if (i + 1) % 20 == 0 or i == len(examples) - 1:
                current_acc = adapter_result.total_correct / (i + 1) * 100
                print(f"  Progress: {i + 1}/{len(examples)} | Accuracy: {current_acc:.1f}%")

        # Calculate final metrics
        adapter_result.accuracy = adapter_result.total_correct / adapter_result.total_questions * 100
        adapter_result.total_tokens = adapter_result.total_input_tokens + adapter_result.total_output_tokens

        # Per-category accuracy
        for cat in category_total:
            if category_total[cat] > 0:
                adapter_result.category_accuracy[cat] = category_correct[cat] / category_total[cat] * 100

        results.adapter_results.append(adapter_result)

        print(f"\n{adapter_name} Results:")
        print(f"  Accuracy: {adapter_result.accuracy:.2f}%")
        print(f"  Tokens - Input: {adapter_result.total_input_tokens}, Output: {adapter_result.total_output_tokens}")

    return results


def print_results(results: BenchmarkResults) -> None:
    """Print benchmark results summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Sort by accuracy descending
    sorted_results = sorted(results.adapter_results, key=lambda x: x.accuracy, reverse=True)

    print(f"\n{'Adapter':<20} {'Accuracy':>10} {'Correct':>10} {'Input Tok':>12} {'Output Tok':>12} {'Total Tok':>12}")
    print("-" * 78)

    best_accuracy = sorted_results[0].accuracy if sorted_results else 0
    min_tokens = min((r.total_tokens for r in sorted_results), default=0)

    for r in sorted_results:
        acc_marker = " *" if r.accuracy == best_accuracy else ""
        tok_marker = " ^" if r.total_tokens == min_tokens else ""
        print(
            f"{r.adapter_name:<20} {r.accuracy:>9.2f}%{acc_marker} "
            f"{r.total_correct:>10} {r.total_input_tokens:>12} "
            f"{r.total_output_tokens:>12} {r.total_tokens:>12}{tok_marker}"
        )

    print("-" * 78)
    print("* = Best accuracy, ^ = Lowest token usage")

    # Per-category breakdown for best adapter
    if sorted_results:
        best = sorted_results[0]
        print(f"\nCategory breakdown for {best.adapter_name}:")
        for cat, acc in sorted(best.category_accuracy.items(), key=lambda x: x[1], reverse=True):
            print(f"  {cat:<25} {acc:>6.1f}%")


def save_results(results: BenchmarkResults, output_dir: str = "benchmark_results") -> None:
    """Save results to JSON files for inspection."""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save summary
    adapters_data: list[dict[str, Any]] = []
    for r in results.adapter_results:
        adapters_data.append(
            {
                "name": r.adapter_name,
                "accuracy": r.accuracy,
                "total_correct": r.total_correct,
                "total_questions": r.total_questions,
                "total_input_tokens": r.total_input_tokens,
                "total_output_tokens": r.total_output_tokens,
                "total_tokens": r.total_tokens,
                "category_accuracy": r.category_accuracy,
            }
        )

    summary = {
        "model": results.model,
        "timestamp": results.timestamp,
        "n_questions": results.n_questions,
        "adapters": adapters_data,
    }

    summary_file = output_path / f"mmlu_pro_summary_{timestamp}.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to: {summary_file}")

    # Save detailed logs per adapter
    for r in results.adapter_results:
        logs_data = []
        for log in r.logs:
            logs_data.append(
                {
                    "question_id": log.question_id,
                    "category": log.category,
                    "question": log.question,
                    "options": log.options,
                    "ground_truth": log.ground_truth,
                    "predicted": log.predicted,
                    "correct": log.correct,
                    "input_tokens": log.input_tokens,
                    "output_tokens": log.output_tokens,
                    "error": log.error,
                    "raw_messages": log.raw_messages,
                }
            )

        adapter_file = output_path / f"mmlu_pro_{r.adapter_name}_{timestamp}.json"
        with open(adapter_file, "w") as f:
            json.dump(logs_data, f, indent=2)
        print(f"Logs for {r.adapter_name} saved to: {adapter_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="MMLU-Pro accuracy benchmark")
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-lite",
        help="Model to use for benchmark",
    )
    parser.add_argument(
        "--n-validation",
        type=int,
        default=70,
        help="Number of validation samples",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=130,
        help="Number of test samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save detailed results to files",
    )
    parser.add_argument(
        "--output-dir",
        default="benchmark_results",
        help="Output directory for results",
    )
    args = parser.parse_args()

    results = run_benchmark(
        model=args.model,
        n_validation=args.n_validation,
        n_test=args.n_test,
        seed=args.seed,
    )
    print_results(results)

    if args.save:
        save_results(results, args.output_dir)


if __name__ == "__main__":
    main()
