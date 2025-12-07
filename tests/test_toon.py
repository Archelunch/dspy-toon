# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Tests for TOON encoding and decoding."""

import pytest

from dspy_toon import ToonDecodeError, decode, encode


class TestEncode:
    """Tests for TOON encoding."""

    def test_encode_simple_object(self, simple_user_data):
        """Test encoding a simple object."""
        result = encode(simple_user_data)
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_encode_nested_object(self, nested_data):
        """Test encoding a nested object."""
        result = encode(nested_data)
        assert "user:" in result
        assert "name: Alice" in result
        assert "settings:" in result
        assert "theme: dark" in result

    def test_encode_primitive_array(self):
        """Test encoding an array of primitives."""
        data = [1, 2, 3, 4, 5]
        result = encode(data)
        assert "[5]:" in result
        assert "1,2,3,4,5" in result

    def test_encode_tabular_array(self, user_list_data):
        """Test encoding uniform objects as tabular format."""
        result = encode(user_list_data)
        # Should use tabular format [N,]{fields}:
        assert "[3,]" in result or "[3]" in result
        assert "id" in result
        assert "name" in result
        assert "age" in result

    def test_encode_mixed_array(self):
        """Test encoding a mixed array."""
        data = [{"name": "Alice"}, 42, "hello"]
        result = encode(data)
        assert "[3]:" in result
        assert "- name: Alice" in result
        assert "- 42" in result
        assert "- hello" in result

    def test_encode_empty_array(self):
        """Test encoding an empty array."""
        result = encode({"items": []})
        assert "items[0]:" in result

    def test_encode_string_quoting(self):
        """Test that strings are quoted only when necessary."""
        # Simple strings don't need quotes
        assert encode({"key": "hello"}) == "key: hello"

        # Strings with colons need quotes
        result = encode({"key": "hello: world"})
        assert '"hello: world"' in result

        # Boolean-like strings need quotes
        result = encode({"key": "true"})
        assert '"true"' in result

    def test_encode_special_values(self):
        """Test encoding special values."""
        data = {
            "null_val": None,
            "true_val": True,
            "false_val": False,
            "int_val": 42,
            "float_val": 3.14,
        }
        result = encode(data)
        assert "null_val: null" in result
        assert "true_val: true" in result
        assert "false_val: false" in result
        assert "int_val: 42" in result
        assert "float_val: 3.14" in result


class TestDecode:
    """Tests for TOON decoding."""

    def test_decode_simple_object(self):
        """Test decoding a simple object."""
        toon = "name: Alice\nage: 30"
        result = decode(toon)
        assert result == {"name": "Alice", "age": 30}

    def test_decode_nested_object(self):
        """Test decoding a nested object."""
        toon = "user:\n  name: Alice\n  settings:\n    theme: dark"
        result = decode(toon)
        assert result == {"user": {"name": "Alice", "settings": {"theme": "dark"}}}

    def test_decode_primitive_array(self):
        """Test decoding an inline primitive array."""
        toon = "[3]: 1,2,3"
        result = decode(toon)
        assert result == [1, 2, 3]

    def test_decode_tabular_array(self):
        """Test decoding a tabular array."""
        toon = "[2,]{id,name}:\n  1,Alice\n  2,Bob"
        result = decode(toon)
        assert result == [
            {"id": 1, "name": "Alice"},
            {"id": 2, "name": "Bob"},
        ]

    def test_decode_empty_input(self):
        """Test decoding empty input returns empty object."""
        assert decode("") == {}
        assert decode("   ") == {}

    def test_decode_quoted_strings(self):
        """Test decoding quoted strings."""
        toon = 'key: "hello: world"'
        result = decode(toon)
        assert result == {"key": "hello: world"}

    def test_decode_escape_sequences(self):
        """Test decoding escape sequences in strings."""
        toon = 'text: "line1\\nline2"'
        result = decode(toon)
        assert result == {"text": "line1\nline2"}

    def test_decode_error_on_malformed(self):
        """Test that malformed input raises ToonDecodeError."""
        # Test with unterminated quoted string
        with pytest.raises(ToonDecodeError):
            decode('key: "unterminated string')


class TestRoundTrip:
    """Test encoding and decoding round-trips."""

    @pytest.mark.parametrize(
        "data",
        [
            {"name": "Alice", "age": 30},
            {"items": [1, 2, 3]},
            {"active": True, "count": 0},
            {"text": "hello world"},
            [1, 2, 3, 4, 5],
            [{"id": 1, "name": "A"}, {"id": 2, "name": "B"}],
        ],
    )
    def test_round_trip(self, data):
        """Test that encode -> decode returns equivalent data."""
        encoded = encode(data)
        decoded = decode(encoded)
        assert decoded == data

    def test_round_trip_nested(self, nested_data):
        """Test round-trip with nested structures."""
        encoded = encode(nested_data)
        decoded = decode(encoded)
        assert decoded == nested_data

    def test_round_trip_preserves_order(self):
        """Test that key order is preserved."""
        data = {"z": 1, "a": 2, "m": 3}
        encoded = encode(data)
        decoded = decode(encoded)
        assert list(decoded.keys()) == ["z", "a", "m"]
