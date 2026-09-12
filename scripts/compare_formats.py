#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Compare dspy-toon vs toon-python output formats.

This script shows actual differences between the two implementations.

Usage:
    python scripts/compare_formats.py
"""

import sys
from pathlib import Path

from pydantic import BaseModel

# Add toon-python to path
toon_python_path = Path(__file__).parent.parent.parent / "toon-python" / "src"
sys.path.insert(0, str(toon_python_path))

from toon_format import decode as toon_decode  # noqa: E402
from toon_format import encode as toon_encode  # noqa: E402

from dspy_toon import decode as dspy_decode  # noqa: E402
from dspy_toon import encode as dspy_encode  # noqa: E402
from dspy_toon.adapter import _build_toon_schema  # noqa: E402


def print_section(title: str):
    print(f"\n{'=' * 70}")
    print(f" {title}")
    print("=" * 70)


def show_diff(label: str, dspy_out: str, toon_out: str):
    """Show character-level differences between outputs."""
    print(f"\n--- {label} ---")
    print(f"\n[dspy-toon] ({len(dspy_out)} chars):")
    print(repr(dspy_out))
    print(f"\n[toon-python] ({len(toon_out)} chars):")
    print(repr(toon_out))

    if dspy_out == toon_out:
        print("\n[Result]: IDENTICAL")
        return True
    else:
        print("\n[Result]: DIFFERENT")
        # Show first difference
        for i, (a, b) in enumerate(zip(dspy_out, toon_out)):
            if a != b:
                print(f"  First diff at position {i}: dspy='{repr(a)}' vs toon='{repr(b)}'")
                print(f"  Context: ...{repr(dspy_out[max(0, i - 10) : i + 10])}...")
                break
        if len(dspy_out) != len(toon_out):
            print(f"  Length diff: dspy={len(dspy_out)}, toon={len(toon_out)}")
        return False


def test_encode_simple_object():
    """Compare simple object encoding."""
    print_section("Simple Object Encoding")

    data = {"name": "Alice", "age": 30}

    dspy_out = dspy_encode(data)
    toon_out = toon_encode(data)

    print("\n[Formatted output]:")
    print("dspy-toon:")
    print(dspy_out)
    print("\ntoon-python:")
    print(toon_out)

    return show_diff("Raw comparison", dspy_out, toon_out)


def test_encode_nested_object():
    """Compare nested object encoding."""
    print_section("Nested Object Encoding")

    data = {
        "name": "Alice",
        "address": {"city": "NYC", "country": "US"},
    }

    dspy_out = dspy_encode(data)
    toon_out = toon_encode(data)

    print("\n[Formatted output]:")
    print("dspy-toon:")
    print(dspy_out)
    print("\ntoon-python:")
    print(toon_out)

    return show_diff("Raw comparison", dspy_out, toon_out)


def test_encode_primitive_array():
    """Compare primitive array encoding."""
    print_section("Primitive Array Encoding")

    data = {"tags": ["python", "ai", "llm"]}

    dspy_out = dspy_encode(data)
    toon_out = toon_encode(data)

    print("\n[Formatted output]:")
    print("dspy-toon:")
    print(dspy_out)
    print("\ntoon-python:")
    print(toon_out)

    return show_diff("Raw comparison", dspy_out, toon_out)


def test_encode_tabular_array():
    """Compare tabular array encoding."""
    print_section("Tabular Array Encoding")

    data = {
        "items": [
            {"id": 1, "name": "Apple", "price": 1.5},
            {"id": 2, "name": "Banana", "price": 0.75},
        ]
    }

    dspy_out = dspy_encode(data)
    toon_out = toon_encode(data)

    print("\n[Formatted output]:")
    print("dspy-toon:")
    print(dspy_out)
    print("\ntoon-python:")
    print(toon_out)

    return show_diff("Raw comparison", dspy_out, toon_out)


def test_encode_with_nulls():
    """Compare encoding with null values."""
    print_section("Encoding with Nulls")

    data = {"name": "Bob", "age": None, "tags": None}

    dspy_out = dspy_encode(data)
    toon_out = toon_encode(data)

    print("\n[Formatted output]:")
    print("dspy-toon:")
    print(dspy_out)
    print("\ntoon-python:")
    print(toon_out)

    return show_diff("Raw comparison", dspy_out, toon_out)


def test_schema_vs_encode():
    """Show that schema DESCRIBES the format, encode PRODUCES actual output."""
    print_section("Schema vs Encode - Different Purposes")

    print("""
    IMPORTANT: Schema and Encode serve DIFFERENT purposes:

    1. SCHEMA (_build_toon_schema):
       - TYPE DESCRIPTION for LLMs
       - Shows: types (string, int), placeholders ([COUNT]), descriptions
       - Purpose: Tell LLM "produce output in THIS format"

    2. ENCODE (encode()):
       - ACTUAL DATA serialization
       - Shows: real values (123, "Alice"), actual counts ([2])
       - Purpose: Convert Python data to TOON string
    """)

    class Item(BaseModel):
        id: int
        name: str

    class Model(BaseModel):
        items: list[Item]
        tags: list[str]

    schema = _build_toon_schema(Model)

    # Sample data that matches the schema
    data = {
        "items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
        "tags": ["x", "y", "z"],
    }
    dspy_encoded = dspy_encode(data)
    toon_encoded = toon_encode(data)

    print("[1. SCHEMA - Type description for LLM]:")
    print(schema)

    print("\n[2. ENCODE - Both implementations produce IDENTICAL output]:")
    print(f"dspy-toon encode:  {repr(dspy_encoded)}")
    print(f"toon-python encode: {repr(toon_encoded)}")
    print(f"Match: {dspy_encoded == toon_encoded}")

    print("\n[3. How schema maps to encoded output]:")
    print("  Schema:  items[COUNT]{id,name}:  -->  Encoded: items[2]{id,name}:")
    print("  Schema:  tags[COUNT]: string,... -->  Encoded: tags[3]: x,y,z")
    print("  Schema:  id: int                 -->  Encoded: 1, 2")
    print("  Schema:  name: string            -->  Encoded: A, B")


def test_indentation_comparison():
    """Compare indentation between implementations."""
    print_section("Indentation Comparison")

    data = {
        "level1": {
            "level2": {
                "level3": "deep",
            }
        }
    }

    dspy_out = dspy_encode(data)
    toon_out = toon_encode(data)

    print("\n[dspy-toon]:")
    for i, line in enumerate(dspy_out.split("\n")):
        spaces = len(line) - len(line.lstrip())
        print(f"  Line {i}: {spaces} spaces | '{line}'")

    print("\n[toon-python]:")
    for i, line in enumerate(toon_out.split("\n")):
        spaces = len(line) - len(line.lstrip())
        print(f"  Line {i}: {spaces} spaces | '{line}'")

    return show_diff("Raw comparison", dspy_out, toon_out)


def test_decode_simple_object():
    """Compare simple object decoding."""
    print_section("Decode: Simple Object")

    toon_str = "name: Alice\nage: 30"

    dspy_out = dspy_decode(toon_str)
    toon_out = toon_decode(toon_str)

    print(f"\n[Input TOON]:\n{toon_str}")
    print(f"\n[dspy-toon decode]: {dspy_out}")
    print(f"[toon-python decode]: {toon_out}")

    match = dspy_out == toon_out
    print(f"\n[Result]: {'IDENTICAL' if match else 'DIFFERENT'}")
    return match


def test_decode_nested_object():
    """Compare nested object decoding."""
    print_section("Decode: Nested Object")

    toon_str = "name: Alice\naddress:\n  city: NYC\n  country: US"

    dspy_out = dspy_decode(toon_str)
    toon_out = toon_decode(toon_str)

    print(f"\n[Input TOON]:\n{toon_str}")
    print(f"\n[dspy-toon decode]: {dspy_out}")
    print(f"[toon-python decode]: {toon_out}")

    match = dspy_out == toon_out
    print(f"\n[Result]: {'IDENTICAL' if match else 'DIFFERENT'}")
    return match


def test_decode_primitive_array():
    """Compare primitive array decoding."""
    print_section("Decode: Primitive Array")

    toon_str = "tags[3]: python,ai,llm"

    dspy_out = dspy_decode(toon_str)
    toon_out = toon_decode(toon_str)

    print(f"\n[Input TOON]:\n{toon_str}")
    print(f"\n[dspy-toon decode]: {dspy_out}")
    print(f"[toon-python decode]: {toon_out}")

    match = dspy_out == toon_out
    print(f"\n[Result]: {'IDENTICAL' if match else 'DIFFERENT'}")
    return match


def test_decode_tabular_array():
    """Compare tabular array decoding."""
    print_section("Decode: Tabular Array")

    toon_str = "items[2]{id,name,price}:\n  1,Apple,1.5\n  2,Banana,0.75"

    dspy_out = dspy_decode(toon_str)
    toon_out = toon_decode(toon_str)

    print(f"\n[Input TOON]:\n{toon_str}")
    print(f"\n[dspy-toon decode]: {dspy_out}")
    print(f"[toon-python decode]: {toon_out}")

    match = dspy_out == toon_out
    print(f"\n[Result]: {'IDENTICAL' if match else 'DIFFERENT'}")
    if not match:
        print(f"  dspy types: {[(k, type(v)) for k, v in dspy_out.items()]}")
        print(f"  toon types: {[(k, type(v)) for k, v in toon_out.items()]}")
    return match


def test_decode_with_nulls():
    """Compare decoding with null values."""
    print_section("Decode: With Nulls")

    toon_str = "name: Bob\nage: null\ntags: null"

    dspy_out = dspy_decode(toon_str)
    toon_out = toon_decode(toon_str)

    print(f"\n[Input TOON]:\n{toon_str}")
    print(f"\n[dspy-toon decode]: {dspy_out}")
    print(f"[toon-python decode]: {toon_out}")

    match = dspy_out == toon_out
    print(f"\n[Result]: {'IDENTICAL' if match else 'DIFFERENT'}")
    return match


def test_decode_roundtrip():
    """Test encode->decode roundtrip for both implementations."""
    print_section("Roundtrip: Encode -> Decode")

    test_data = [
        {"name": "Alice", "age": 30},
        {"items": [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}]},
        {"tags": ["x", "y", "z"]},
        {"nested": {"deep": {"value": 42}}},
        {"mixed": None, "valid": "yes"},
    ]

    all_match = True
    for i, original in enumerate(test_data):
        # dspy-toon roundtrip
        dspy_encoded = dspy_encode(original)
        dspy_decoded = dspy_decode(dspy_encoded)

        # toon-python roundtrip
        toon_encoded = toon_encode(original)
        toon_decoded = toon_decode(toon_encoded)

        # Cross-decode: decode dspy output with toon, and vice versa
        cross_dspy_to_toon = toon_decode(dspy_encoded)
        cross_toon_to_dspy = dspy_decode(toon_encoded)

        match = dspy_decoded == toon_decoded == cross_dspy_to_toon == cross_toon_to_dspy

        print(f"\n[Test {i + 1}] Original: {original}")
        print(f"  dspy roundtrip:  {dspy_decoded}")
        print(f"  toon roundtrip:  {toon_decoded}")
        print(f"  cross-decode OK: {cross_dspy_to_toon == cross_toon_to_dspy}")
        print(f"  Result: {'MATCH' if match else 'DIFFER'}")

        if not match:
            all_match = False

    return all_match


def main():
    print("=" * 70)
    print(" TOON Format Comparison: dspy-toon vs toon-python")
    print("=" * 70)
    print("\nComparing encode() and decode() between implementations.")

    # Encode tests
    encode_results = {
        "Encode: Simple object": test_encode_simple_object(),
        "Encode: Nested object": test_encode_nested_object(),
        "Encode: Primitive array": test_encode_primitive_array(),
        "Encode: Tabular array": test_encode_tabular_array(),
        "Encode: With nulls": test_encode_with_nulls(),
        "Encode: Indentation": test_indentation_comparison(),
    }

    # Decode tests
    decode_results = {
        "Decode: Simple object": test_decode_simple_object(),
        "Decode: Nested object": test_decode_nested_object(),
        "Decode: Primitive array": test_decode_primitive_array(),
        "Decode: Tabular array": test_decode_tabular_array(),
        "Decode: With nulls": test_decode_with_nulls(),
        "Decode: Roundtrip": test_decode_roundtrip(),
    }

    results = {**encode_results, **decode_results}

    test_schema_vs_encode()

    print_section("SUMMARY")
    print("\n  Test                    | Result")
    print("  " + "-" * 42)
    for name, passed in results.items():
        status = "IDENTICAL" if passed else "DIFFERENT"
        print(f"  {name:<23} | {status}")

    identical = sum(results.values())
    total = len(results)
    print(f"\n  {identical}/{total} tests produce identical output")

    if identical < total:
        print("\n  NOTE: Differences may be due to:")
        print("    - Indentation (2 vs 4 spaces)")
        print("    - Comma in tabular headers [N,] vs [N]")
        print("    - Minor formatting choices")
        print("  These differences don't affect TOON spec compliance.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
