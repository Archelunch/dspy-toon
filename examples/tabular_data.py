#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Tabular data extraction example with ToonAdapter.

This example demonstrates ToonAdapter's efficiency with lists of
uniform objects, which are encoded in compact tabular format.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python examples/tabular_data.py
"""

from typing import Literal

import dspy
from pydantic import BaseModel

from dspy_toon import ToonAdapter, encode

# =============================================================================
# Models for Tabular Data
# =============================================================================


class ExtractedPerson(BaseModel):
    """Person extracted from text."""

    name: str
    age: int
    occupation: str


class ProductItem(BaseModel):
    """Product in a catalog."""

    id: int
    name: str
    price: float
    category: str
    in_stock: bool


class TransactionRecord(BaseModel):
    """Financial transaction record."""

    date: str
    description: str
    amount: float
    category: Literal["income", "expense", "transfer"]


class ExtractedEntity(BaseModel):
    """Named entity from text."""

    text: str
    entity_type: Literal["person", "organization", "location", "date", "money"]
    confidence: float


# =============================================================================
# DSPy Signatures
# =============================================================================


class ExtractPeople(dspy.Signature):
    """Extract all people mentioned in the text."""

    text: str = dspy.InputField(desc="Text containing multiple people")
    people: list[ExtractedPerson] = dspy.OutputField(desc="List of extracted people")


class ExtractProducts(dspy.Signature):
    """Extract product information from catalog text."""

    catalog: str = dspy.InputField(desc="Product catalog text")
    products: list[ProductItem] = dspy.OutputField(desc="List of products")


class ExtractEntities(dspy.Signature):
    """Extract all named entities from text."""

    text: str = dspy.InputField(desc="Text to analyze")
    entities: list[ExtractedEntity] = dspy.OutputField(desc="Named entities found")


# =============================================================================
# TOON Tabular Format Demonstration
# =============================================================================


def demonstrate_tabular_format():
    """Show how tabular data looks in TOON vs JSON."""
    print("\n" + "=" * 60)
    print("TOON TABULAR FORMAT DEMONSTRATION")
    print("=" * 60)

    # Sample tabular data - list of uniform objects
    users = [
        {"id": 1, "name": "Alice", "email": "alice@example.com", "active": True},
        {"id": 2, "name": "Bob", "email": "bob@example.com", "active": True},
        {"id": 3, "name": "Charlie", "email": "charlie@example.com", "active": False},
        {"id": 4, "name": "Diana", "email": "diana@example.com", "active": True},
        {"id": 5, "name": "Eve", "email": "eve@example.com", "active": False},
    ]

    import json

    json_output = json.dumps(users, indent=2)
    json_compact = json.dumps(users)
    toon_output = encode(users)

    print("\n📋 JSON (pretty):")
    print(json_output[:300] + "..." if len(json_output) > 300 else json_output)

    print("\n📋 JSON (compact):")
    print(json_compact[:200] + "..." if len(json_compact) > 200 else json_compact)

    print("\n📝 TOON (tabular format):")
    print(toon_output)

    # Token comparison
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        json_tokens = len(encoding.encode(json_compact))
        toon_tokens = len(encoding.encode(toon_output))
        savings = (json_tokens - toon_tokens) / json_tokens * 100

        print("\n📊 Token Comparison:")
        print(f"   JSON (compact): {json_tokens} tokens, {len(json_compact)} chars")
        print(f"   TOON (tabular): {toon_tokens} tokens, {len(toon_output)} chars")
        print(f"   Token savings: {savings:.1f}%")
        print(f"   Character savings: {(len(json_compact) - len(toon_output)) / len(json_compact) * 100:.1f}%")
    except ImportError:
        print("\n(Install tiktoken to see token comparison)")


def demonstrate_large_dataset():
    """Show token savings with larger datasets."""
    print("\n" + "=" * 60)
    print("LARGE DATASET TOKEN COMPARISON")
    print("=" * 60)

    import json

    # Generate larger dataset
    large_data = [
        {
            "id": i,
            "name": f"User_{i}",
            "email": f"user{i}@example.com",
            "department": ["Engineering", "Sales", "Marketing", "HR"][i % 4],
            "active": i % 3 != 0,
        }
        for i in range(1, 101)  # 100 users
    ]

    json_compact = json.dumps(large_data)
    toon_output = encode(large_data)

    print("\n📋 Dataset: 100 uniform user records")
    print("\n📊 Size Comparison:")
    print(f"   JSON: {len(json_compact)} characters")
    print(f"   TOON: {len(toon_output)} characters")
    print(f"   Reduction: {(len(json_compact) - len(toon_output)) / len(json_compact) * 100:.1f}%")

    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        json_tokens = len(encoding.encode(json_compact))
        toon_tokens = len(encoding.encode(toon_output))
        savings = (json_tokens - toon_tokens) / json_tokens * 100

        print(f"\n   JSON tokens: {json_tokens}")
        print(f"   TOON tokens: {toon_tokens}")
        print(f"   Token savings: {savings:.1f}%")

        # Cost estimation (example: $0.01 per 1K tokens)
        cost_per_1k = 0.01
        json_cost = (json_tokens / 1000) * cost_per_1k
        toon_cost = (toon_tokens / 1000) * cost_per_1k
        print("\n   Estimated cost (at $0.01/1K tokens):")
        print(f"   JSON: ${json_cost:.4f}")
        print(f"   TOON: ${toon_cost:.4f}")
        print(f"   Savings per request: ${json_cost - toon_cost:.4f}")
        print(f"   Savings per 10,000 requests: ${(json_cost - toon_cost) * 10000:.2f}")
    except ImportError:
        pass


def extract_people_example():
    """Demonstrate extracting multiple people."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Extract Multiple People")
    print("=" * 60)

    text = """
    The meeting was attended by several team members:
    - Alice Johnson, 32, Senior Engineer
    - Bob Smith, 45, Project Manager
    - Carol White, 28, UX Designer
    - David Brown, 38, DevOps Lead
    - Emma Davis, 30, QA Engineer
    """

    print(f"\n📄 Input Text:\n{text}")

    extractor = dspy.Predict(ExtractPeople)

    try:
        result = extractor(text=text)
        print(f"\n✅ Extracted {len(result.people)} people:")

        # Show as TOON tabular format
        people_data = [p.model_dump() for p in result.people]
        print("\n📝 Result in TOON format:")
        print(encode(people_data))
    except Exception as e:
        print(f"❌ Error: {e}")


def extract_entities_example():
    """Demonstrate entity extraction."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Named Entity Extraction")
    print("=" * 60)

    text = """
    Apple Inc. announced that CEO Tim Cook will present at the
    Worldwide Developers Conference in San Jose on June 5, 2024.
    The company reported revenue of $89.5 billion for the quarter.
    """

    print(f"\n📄 Input Text:\n{text}")

    extractor = dspy.Predict(ExtractEntities)

    try:
        result = extractor(text=text)
        print(f"\n✅ Extracted {len(result.entities)} entities:")

        # Show as TOON tabular format
        entities_data = [e.model_dump() for e in result.entities]
        print("\n📝 Result in TOON format:")
        print(encode(entities_data))
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run examples."""
    print("=" * 60)
    print("DSPy-TOON Tabular Data Examples")
    print("=" * 60)

    # Always show format demos (no LLM required)
    demonstrate_tabular_format()
    demonstrate_large_dataset()

    # Try to configure DSPy for LLM examples
    try:
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm, adapter=ToonAdapter())
        print("\n✅ DSPy configured with ToonAdapter")

        # Run LLM examples
        extract_people_example()
        extract_entities_example()
    except Exception as e:
        print(f"\n⚠️  Could not configure LM: {e}")
        print("   Skipping LLM examples")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
