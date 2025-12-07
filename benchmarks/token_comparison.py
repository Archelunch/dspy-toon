#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Token usage comparison between TOON, JSON, and other formats.

This benchmark compares token usage across different serialization formats
used for LLM structured outputs.

Usage:
    python -m benchmarks.token_comparison
    # Or with specific options:
    python -m benchmarks.token_comparison --model gpt-4o
"""

import argparse
import json
from dataclasses import dataclass
from typing import Any

from dspy_toon import encode as toon_encode

# =============================================================================
# Token Counting
# =============================================================================


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Count tokens using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for tokenizer selection

    Returns:
        Number of tokens
    """
    try:
        import tiktoken

        # Map model names to encodings
        if "gpt-4" in model or "gpt-3.5" in model:
            encoding = tiktoken.encoding_for_model(model)
        else:
            # Default to cl100k_base for most modern models
            encoding = tiktoken.get_encoding("cl100k_base")

        return len(encoding.encode(text))
    except ImportError:
        # Fallback: rough estimate (4 chars per token average)
        return len(text) // 4


# =============================================================================
# Format Encoders
# =============================================================================


def encode_json(data: Any, indent: int | None = None) -> str:
    """Encode data as JSON."""
    return json.dumps(data, indent=indent, ensure_ascii=False)


def encode_json_compact(data: Any) -> str:
    """Encode data as compact JSON (no whitespace)."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False)


def encode_toon(data: Any) -> str:
    """Encode data as TOON."""
    return toon_encode(data)


# =============================================================================
# Benchmark Data
# =============================================================================

BENCHMARK_DATASETS = {
    "simple_object": {
        "description": "Simple flat object with various types",
        "data": {
            "name": "Alice Johnson",
            "age": 30,
            "email": "alice@example.com",
            "active": True,
            "score": 95.5,
        },
    },
    "nested_object": {
        "description": "Object with nested structures",
        "data": {
            "user": {
                "name": "Bob Smith",
                "profile": {
                    "bio": "Software engineer",
                    "location": "San Francisco",
                    "links": {
                        "github": "https://github.com/bob",
                        "twitter": "https://twitter.com/bob",
                    },
                },
            },
            "settings": {
                "theme": "dark",
                "notifications": True,
                "language": "en",
            },
        },
    },
    "user_list_small": {
        "description": "Small list of uniform user objects (5 users)",
        "data": [
            {"id": i, "name": f"User{i}", "email": f"user{i}@example.com", "active": i % 2 == 0} for i in range(1, 6)
        ],
    },
    "user_list_medium": {
        "description": "Medium list of uniform user objects (20 users)",
        "data": [
            {"id": i, "name": f"User{i}", "email": f"user{i}@example.com", "active": i % 2 == 0} for i in range(1, 21)
        ],
    },
    "user_list_large": {
        "description": "Large list of uniform user objects (100 users)",
        "data": [
            {"id": i, "name": f"User{i}", "email": f"user{i}@example.com", "active": i % 2 == 0} for i in range(1, 101)
        ],
    },
    "products_catalog": {
        "description": "Product catalog with pricing",
        "data": [
            {
                "sku": f"SKU-{i:04d}",
                "name": f"Product {i}",
                "price": 9.99 + i * 5,
                "quantity": 100 - i,
                "category": ["Electronics", "Clothing", "Home"][i % 3],
            }
            for i in range(1, 51)
        ],
    },
    "api_response": {
        "description": "Typical API response structure",
        "data": {
            "status": "success",
            "code": 200,
            "data": {
                "results": [
                    {"id": 1, "title": "First Item", "score": 0.95},
                    {"id": 2, "title": "Second Item", "score": 0.87},
                    {"id": 3, "title": "Third Item", "score": 0.82},
                ],
                "pagination": {
                    "page": 1,
                    "per_page": 10,
                    "total": 150,
                    "total_pages": 15,
                },
            },
            "meta": {
                "request_id": "abc123",
                "timestamp": "2025-01-15T10:30:00Z",
            },
        },
    },
    "mixed_array": {
        "description": "Array with mixed types (non-tabular)",
        "data": [
            {"type": "user", "name": "Alice"},
            {"type": "product", "sku": "P001", "price": 29.99},
            "plain string",
            42,
            True,
            {"type": "nested", "data": {"key": "value"}},
        ],
    },
}


# =============================================================================
# Benchmark Results
# =============================================================================


@dataclass
class FormatResult:
    """Results for a single format."""

    name: str
    encoded: str
    tokens: int
    chars: int


@dataclass
class BenchmarkResult:
    """Results for a single benchmark."""

    dataset_name: str
    description: str
    formats: list[FormatResult]

    @property
    def best_format(self) -> FormatResult:
        """Get the format with fewest tokens."""
        return min(self.formats, key=lambda f: f.tokens)

    def get_savings(self, baseline: str = "JSON") -> dict[str, float]:
        """Calculate token savings compared to baseline."""
        baseline_result = next((f for f in self.formats if f.name == baseline), None)
        if not baseline_result:
            return {}

        return {
            f.name: ((baseline_result.tokens - f.tokens) / baseline_result.tokens * 100)
            for f in self.formats
            if f.name != baseline
        }


# =============================================================================
# Run Benchmarks
# =============================================================================


def run_benchmark(data: Any, model: str = "gpt-4o") -> list[FormatResult]:
    """Run benchmark on a single dataset.

    Args:
        data: Data to encode
        model: Model for token counting

    Returns:
        List of results for each format
    """
    results = []

    # JSON (pretty)
    json_pretty = encode_json(data, indent=2)
    results.append(
        FormatResult(
            name="JSON (pretty)",
            encoded=json_pretty,
            tokens=count_tokens(json_pretty, model),
            chars=len(json_pretty),
        )
    )

    # JSON (compact)
    json_compact = encode_json_compact(data)
    results.append(
        FormatResult(
            name="JSON",
            encoded=json_compact,
            tokens=count_tokens(json_compact, model),
            chars=len(json_compact),
        )
    )

    # TOON
    toon = encode_toon(data)
    results.append(
        FormatResult(
            name="TOON",
            encoded=toon,
            tokens=count_tokens(toon, model),
            chars=len(toon),
        )
    )

    return results


def run_all_benchmarks(model: str = "gpt-4o") -> list[BenchmarkResult]:
    """Run all benchmarks.

    Args:
        model: Model for token counting

    Returns:
        List of benchmark results
    """
    results = []

    for name, dataset in BENCHMARK_DATASETS.items():
        format_results = run_benchmark(dataset["data"], model)
        results.append(
            BenchmarkResult(
                dataset_name=name,
                description=dataset["description"],
                formats=format_results,
            )
        )

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("\n" + "=" * 80)
    print("TOKEN USAGE BENCHMARK: TOON vs JSON")
    print("=" * 80)

    for result in results:
        print(f"\n📊 {result.dataset_name}")
        print(f"   {result.description}")
        print("-" * 60)

        # Header
        print(f"{'Format':<20} {'Tokens':>10} {'Chars':>10} {'Savings':>12}")
        print("-" * 60)

        # Find baseline (JSON)
        baseline = next((f for f in result.formats if f.name == "JSON"), result.formats[0])
        baseline_tokens = baseline.tokens

        for fmt in result.formats:
            savings = (baseline_tokens - fmt.tokens) / baseline_tokens * 100 if baseline_tokens > 0 else 0
            savings_str = f"{savings:+.1f}%" if fmt.name != "JSON" else "-"
            print(f"{fmt.name:<20} {fmt.tokens:>10} {fmt.chars:>10} {savings_str:>12}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_json_tokens = sum(next((f.tokens for f in r.formats if f.name == "JSON"), 0) for r in results)
    total_toon_tokens = sum(next((f.tokens for f in r.formats if f.name == "TOON"), 0) for r in results)

    if total_json_tokens > 0:
        overall_savings = (total_json_tokens - total_toon_tokens) / total_json_tokens * 100
        print(f"\nTotal JSON tokens: {total_json_tokens}")
        print(f"Total TOON tokens: {total_toon_tokens}")
        print(f"Overall TOON savings: {overall_savings:.1f}%")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Token usage benchmark")
    parser.add_argument(
        "--model",
        default="gpt-4o",
        help="Model for token counting (default: gpt-4o)",
    )
    parser.add_argument(
        "--dataset",
        choices=list(BENCHMARK_DATASETS.keys()),
        help="Run only specific dataset",
    )
    parser.add_argument(
        "--show-encoded",
        action="store_true",
        help="Show encoded output for each format",
    )
    args = parser.parse_args()

    if args.dataset:
        dataset = BENCHMARK_DATASETS[args.dataset]
        format_results = run_benchmark(dataset["data"], args.model)
        results = [
            BenchmarkResult(
                dataset_name=args.dataset,
                description=dataset["description"],
                formats=format_results,
            )
        ]
    else:
        results = run_all_benchmarks(args.model)

    print_results(results)

    if args.show_encoded:
        print("\n" + "=" * 80)
        print("ENCODED SAMPLES")
        print("=" * 80)
        for result in results:
            print(f"\n📄 {result.dataset_name}")
            for fmt in result.formats:
                print(f"\n--- {fmt.name} ({fmt.tokens} tokens) ---")
                # Truncate long outputs
                if len(fmt.encoded) > 500:
                    print(fmt.encoded[:500] + "\n... (truncated)")
                else:
                    print(fmt.encoded)


if __name__ == "__main__":
    main()
