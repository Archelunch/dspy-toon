#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Adapter comparison benchmark.

Compares ToonAdapter against JSONAdapter, ChatAdapter, and BAMLAdapter using
real LLM calls with DSPy's usage tracking.

Usage:
    python -m benchmarks.adapter_comparison --model gemini/gemini-2.5-flash-lite
"""

import argparse
from dataclasses import dataclass, field
from typing import Any, Literal

import dspy
from pydantic import BaseModel, Field

from benchmarks.baml_adapter import BAMLAdapter
from dspy_toon import ToonAdapter

# =============================================================================
# Test Models
# =============================================================================


class Person(BaseModel):
    """Person extracted from text."""

    name: str = Field(description="Full name")
    age: int = Field(description="Age in years")
    occupation: str = Field(description="Job or profession")


class Product(BaseModel):
    """Product information."""

    name: str = Field(description="Product name")
    price: float = Field(description="Price in USD")
    category: str = Field(description="Product category")


class Address(BaseModel):
    """Address information."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    country: Literal["US", "CA", "UK", "DE"] = Field(description="Country code")


class PersonWithAddress(BaseModel):
    """Person with nested address."""

    name: str = Field(description="Full name")
    email: str = Field(description="Email address")
    address: Address = Field(description="Home address")


# =============================================================================
# DSPy Signatures
# =============================================================================


class ExtractPerson(dspy.Signature):
    """Extract person information from text."""

    text: str = dspy.InputField()
    person: Person = dspy.OutputField()


class ExtractPeople(dspy.Signature):
    """Extract all people mentioned in the text."""

    text: str = dspy.InputField()
    people: list[Person] = dspy.OutputField()


class ExtractProducts(dspy.Signature):
    """Extract all products from the catalog."""

    text: str = dspy.InputField()
    products: list[Product] = dspy.OutputField()


class ExtractPersonWithAddress(dspy.Signature):
    """Extract person with address from text."""

    text: str = dspy.InputField()
    person: PersonWithAddress = dspy.OutputField()


# =============================================================================
# Test Cases
# =============================================================================

TEST_CASES = [
    {
        "name": "single_person",
        "description": "Extract single person",
        "input_text": "Alice Johnson is a 35-year-old software engineer.",
        "signature": ExtractPerson,
    },
    {
        "name": "list_3_people",
        "description": "Extract list of 3 people",
        "input_text": """
        Alice (35) is a software engineer.
        Bob (28) works as a designer.
        Carol (42) is a project manager.
        """,
        "signature": ExtractPeople,
    },
    {
        "name": "list_10_people",
        "description": "Extract list of 10 people",
        "input_text": """
        Team roster:
        Alice (25) - Engineer
        Bob (32) - Designer
        Carol (28) - Manager
        David (45) - Director
        Eve (30) - Analyst
        Frank (27) - Developer
        Grace (35) - Architect
        Henry (40) - Consultant
        Ivy (29) - Researcher
        Jack (33) - Engineer
        """,
        "signature": ExtractPeople,
    },
    {
        "name": "list_5_products",
        "description": "Extract list of 5 products",
        "input_text": """
        Product catalog:
        - iPhone 15 Pro: $999, Electronics
        - MacBook Air: $1299, Electronics
        - AirPods Pro: $249, Electronics
        - iPad Mini: $499, Electronics
        - Apple Watch: $399, Electronics
        """,
        "signature": ExtractProducts,
    },
    {
        "name": "nested_address",
        "description": "Extract person with nested address",
        "input_text": """
        Contact John Smith at john@example.com.
        He lives at 123 Main Street, Boston, US.
        """,
        "signature": ExtractPersonWithAddress,
    },
]


# =============================================================================
# Benchmark Results
# =============================================================================


@dataclass
class UsageMetrics:
    """Token usage metrics from real LLM call."""

    adapter_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    success: bool = False
    error: str | None = None


@dataclass
class TestResult:
    """Results for a single test case."""

    test_name: str
    description: str
    metrics: list[UsageMetrics] = field(default_factory=list)


def get_adapters() -> dict[str, Any]:
    """Get all adapters to benchmark."""
    return {
        "ToonAdapter": ToonAdapter(),
        "BAMLAdapter": BAMLAdapter(),
        "JSONAdapter": dspy.JSONAdapter(),
        "ChatAdapter": dspy.ChatAdapter(),
    }


def run_benchmark(model: str) -> list[TestResult]:
    """Run benchmark with real LLM calls and usage tracking."""
    print(f"\n{'=' * 70}")
    print("ADAPTER TOKEN USAGE BENCHMARK")
    print(f"Model: {model}")
    print("=" * 70)

    lm = dspy.LM(model, temperature=0.0, cache=False)
    adapters = {
        "ToonAdapter": ToonAdapter(),
        "BAMLAdapter": BAMLAdapter(),
        "JSONAdapter": dspy.JSONAdapter(),
        "ChatAdapter": dspy.ChatAdapter(),
    }

    results: list[TestResult] = []

    for test_case in TEST_CASES:
        print(f"\n📊 {test_case['name']}: {test_case['description']}")
        print("-" * 60)

        signature = test_case["signature"]

        test_result = TestResult(
            test_name=test_case["name"],
            description=test_case["description"],
        )

        for adapter_name, adapter in adapters.items():
            metrics = UsageMetrics(adapter_name=adapter_name)

            try:
                dspy.configure(lm=lm, adapter=adapter, track_usage=True)
                predictor = dspy.Predict(signature)
                prediction = predictor(text=test_case["input_text"])

                # Get usage from prediction
                usage = prediction.get_lm_usage()
                if usage:
                    # Sum up usage across all LM calls
                    for lm_usage in usage.values():
                        metrics.input_tokens += lm_usage.get("prompt_tokens", 0)
                        metrics.output_tokens += lm_usage.get("completion_tokens", 0)
                    metrics.total_tokens = metrics.input_tokens + metrics.output_tokens
                    metrics.success = True

                print(
                    f"{adapter_name} "
                    f"in: {metrics.input_tokens:>5}  "
                    f"out: {metrics.output_tokens:>5}  "
                    f"total: {metrics.total_tokens:>5}"
                )

            except Exception as e:
                metrics.error = str(e)
                print(f"  ❌ {adapter_name:<15} Error: {str(e)[:40]}...")

            test_result.metrics.append(metrics)

        results.append(test_result)

    return results


def print_results(results: list[TestResult]) -> None:
    """Print benchmark results summary."""
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)

    # Aggregate totals
    totals: dict[str, dict[str, int]] = {}
    for result in results:
        for metrics in result.metrics:
            if metrics.adapter_name not in totals:
                totals[metrics.adapter_name] = {"input": 0, "output": 0, "total": 0}
            if metrics.success:
                totals[metrics.adapter_name]["input"] += metrics.input_tokens
                totals[metrics.adapter_name]["output"] += metrics.output_tokens
                totals[metrics.adapter_name]["total"] += metrics.total_tokens

    # Print per-test results
    for result in results:
        print(f"\n📊 {result.test_name}")
        print(f"{'Adapter':<20} {'Input':>8} {'Output':>8} {'Total':>8}")
        print("-" * 50)

        sorted_metrics = sorted(result.metrics, key=lambda x: x.total_tokens if x.success else 9999)
        min_total = min((m.total_tokens for m in sorted_metrics if m.success), default=0)

        for m in sorted_metrics:
            if m.success:
                marker = " 🏆" if m.total_tokens == min_total else ""
                print(f"{m.adapter_name:<20} {m.input_tokens:>8} {m.output_tokens:>8} {m.total_tokens:>8}{marker}")
            else:
                print(f"{m.adapter_name:<20} {'ERROR':>8}")

    # Print totals
    print("\n" + "=" * 70)
    print("TOTAL TOKEN USAGE (all tests)")
    print("=" * 70)

    print(f"\n{'Adapter':<20} {'Input':>10} {'Output':>10} {'Total':>10}")
    print("-" * 55)

    sorted_totals = sorted(totals.items(), key=lambda x: x[1]["total"])
    min_total = sorted_totals[0][1]["total"] if sorted_totals else 0
    json_total = totals.get("JSONAdapter", {}).get("total", 0)

    for adapter, t in sorted_totals:
        marker = " 🏆" if t["total"] == min_total else ""
        print(f"{adapter:<20} {t['input']:>10} {t['output']:>10} {t['total']:>10}{marker}")

    print("-" * 55)

    # Winner
    if sorted_totals:
        winner_name, winner_totals = sorted_totals[0]
        if json_total > 0:
            savings = (json_total - winner_totals["total"]) / json_total * 100
            print(f"\n🏆 Winner: {winner_name} ({savings:+.1f}% vs JSONAdapter)")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Adapter comparison benchmark")
    parser.add_argument(
        "--model",
        default="gemini/gemini-2.5-flash-lite",
        help="Model to use for benchmark",
    )
    args = parser.parse_args()

    results = run_benchmark(args.model)
    print_results(results)


if __name__ == "__main__":
    main()
