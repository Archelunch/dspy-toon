# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Tests for ToonAdapter."""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from dspy_toon import ToonAdapter

# =============================================================================
# Test Models
# =============================================================================


class SimpleUser(BaseModel):
    name: str
    age: int


class UserWithDescription(BaseModel):
    name: str = Field(description="Full name of the user")
    age: int = Field(description="Age in years")


# =============================================================================
# Test Schema Rendering
# =============================================================================


class TestSchemaRendering:
    """Tests for schema rendering in TOON format."""

    def test_simple_model_schema(self):
        """Test rendering schema for simple model."""
        from dspy_toon.adapter import _build_toon_schema

        schema = _build_toon_schema(SimpleUser)
        assert "name: string" in schema
        assert "age: int" in schema

    def test_model_with_descriptions(self):
        """Test that field descriptions appear as comments."""
        from dspy_toon.adapter import _build_toon_schema

        schema = _build_toon_schema(UserWithDescription)
        assert "# Full name of the user" in schema
        assert "# Age in years" in schema

    def test_literal_type_rendering(self):
        """Test that Literal types are rendered correctly."""
        from dspy_toon.adapter import _render_type_str

        result = _render_type_str(Literal["A", "B", "C"])
        assert '"A"' in result
        assert '"B"' in result
        assert '"C"' in result

    def test_optional_type_rendering(self):
        """Test that Optional types include 'or null'."""
        from dspy_toon.adapter import _render_type_str

        result = _render_type_str(str | None)
        assert "null" in result


# =============================================================================
# Test Streaming Compatibility
# =============================================================================


class TestStreamingCompatibility:
    """Tests for streaming compatibility with dspy.streamify.

    ToonAdapter supports streaming via the enable_toon_streaming() function
    which patches DSPy's StreamListener to recognize TOON format patterns.
    """

    def test_enable_toon_streaming(self):
        """Test that enable_toon_streaming adds ToonAdapter to supported adapters."""
        from dspy.streaming.streaming_listener import ADAPTER_SUPPORT_STREAMING

        from dspy_toon import enable_toon_streaming

        enable_toon_streaming()

        # ToonAdapter should now be in supported list
        assert ToonAdapter in ADAPTER_SUPPORT_STREAMING

    def test_streaming_patterns_added(self):
        """Test that ToonAdapter patterns are added to StreamListener."""
        import dspy.streaming

        from dspy_toon import enable_toon_streaming

        enable_toon_streaming()

        # Create a listener to check patterns
        listener = dspy.streaming.StreamListener(signature_field_name="answer")

        assert "ToonAdapter" in listener.adapter_identifiers
        assert "start_identifier" in listener.adapter_identifiers["ToonAdapter"]
        assert listener.adapter_identifiers["ToonAdapter"]["start_identifier"] == "answer:"


# =============================================================================
# Test History Support
# =============================================================================


class TestHistorySupport:
    """Tests for conversation history handling."""

    @pytest.fixture
    def adapter(self):
        return ToonAdapter()

    def test_get_history_field_name_with_history_type(self, adapter):
        """Test _get_history_field_name detects History type."""
        import dspy
        from dspy.adapters.types import History

        class ChatSignature(dspy.Signature):
            """Chat with history."""

            history: History = dspy.InputField()
            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        result = adapter._get_history_field_name(ChatSignature)
        assert result == "history"

    def test_get_history_field_name_without_history(self, adapter):
        """Test _get_history_field_name returns None when no History field."""
        import dspy

        class SimpleSignature(dspy.Signature):
            """Simple signature."""

            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        result = adapter._get_history_field_name(SimpleSignature)
        assert result is None

    def test_format_conversation_history_empty(self, adapter):
        """Test formatting empty conversation history."""
        import dspy

        class SimpleSignature(dspy.Signature):
            """Simple signature."""

            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        inputs = {"history_field": None}
        result = adapter.format_conversation_history(SimpleSignature, "history_field", inputs)

        assert result == []
        assert "history_field" not in inputs


# =============================================================================
# Test Parse Error Handling
# =============================================================================


class TestParseErrorHandling:
    """Tests for parse error handling."""

    @pytest.fixture
    def adapter(self):
        return ToonAdapter()

    def test_parse_raises_error_on_missing_fields(self, adapter):
        """Test that parse raises AdapterParseError when fields are missing."""
        import dspy
        from dspy.utils.exceptions import AdapterParseError

        class TestSignature(dspy.Signature):
            """Test signature."""

            text: str = dspy.InputField()
            name: str = dspy.OutputField()
            age: int = dspy.OutputField()

        # Only provide partial output
        completion = "name: Alice"

        with pytest.raises(AdapterParseError):
            adapter.parse(TestSignature, completion)

    def test_parse_succeeds_with_all_fields(self, adapter):
        """Test that parse succeeds when all fields are present."""
        import dspy

        class TestSignature(dspy.Signature):
            """Test signature."""

            text: str = dspy.InputField()
            answer: str = dspy.OutputField()

        completion = "answer: This is the answer"
        result = adapter.parse(TestSignature, completion)

        assert "answer" in result
        assert "This is the answer" in result["answer"]


# =============================================================================
# Test TOON Format Compliance
# =============================================================================


class TestToonFormatCompliance:
    """Tests for TOON format compliance.

    TOON spec requires:
    - Field names directly concatenated with [COUNT] for arrays: fieldname[COUNT]: ...
    - Tabular format: fieldname[COUNT,]{field1,field2}: ...
    - No duplicate "or null" patterns
    """

    def test_primitive_array_format(self):
        """Test that primitive arrays use TOON format: fieldname[COUNT]: values."""
        from dspy_toon.adapter import _build_toon_schema

        class ModelWithStringList(BaseModel):
            names: list[str]

        schema = _build_toon_schema(ModelWithStringList)
        # Should be "names[COUNT]: string,..." not "names: [COUNT]: string,..."
        assert "names[COUNT]:" in schema
        assert "names: [COUNT]" not in schema

    def test_optional_primitive_array_format(self):
        """Test that optional primitive arrays use correct format."""
        from dspy_toon.adapter import _build_toon_schema

        class ModelWithOptionalList(BaseModel):
            tags: list[str] | None = None

        schema = _build_toon_schema(ModelWithOptionalList)
        # Should have field name directly before [COUNT]
        assert "tags[COUNT]:" in schema
        assert "or null" in schema
        # Should not have double "or null"
        assert "null or null" not in schema

    def test_tabular_array_format(self):
        """Test that object arrays use TOON tabular format: fieldname[COUNT]{fields}."""
        from dspy_toon.adapter import _build_toon_schema

        class Item(BaseModel):
            id: int
            name: str

        class ModelWithObjectList(BaseModel):
            items: list[Item]

        schema = _build_toon_schema(ModelWithObjectList)
        # Should be "items[COUNT]{id,name}:" not "items: [COUNT]{id,name}:"
        # Note: comma delimiter is implicit (default), not shown in [N]
        assert "items[COUNT]{id,name}:" in schema
        assert "items:" not in schema.split("\n")[0]  # First line shouldn't be "items:"

    def test_optional_tabular_array_format(self):
        """Test that optional object arrays use correct format."""
        from dspy_toon.adapter import _build_toon_schema

        class Allergy(BaseModel):
            substance: str

        class Patient(BaseModel):
            allergies: list[Allergy] | None = None

        schema = _build_toon_schema(Patient)
        # Should have field name directly before [COUNT]
        # Note: comma delimiter is implicit (default), not shown in [N]
        assert "allergies[COUNT]{substance}:" in schema
        assert "or null" in schema

    def test_no_duplicate_or_null(self):
        """Test that there are no duplicate 'or null' patterns."""
        from dspy_toon.adapter import _build_toon_schema

        class Address(BaseModel):
            line: str | None = None
            country: Literal["US", "CA"] | None = None

        schema = _build_toon_schema(Address)
        # Should not have "or null or null"
        assert "null or null" not in schema
        # Each field should have exactly one "or null"
        lines = [ln for ln in schema.split("\n") if ln.strip()]
        for line in lines:
            if "or null" in line:
                # Count occurrences
                count = line.count("or null")
                assert count == 1, f"Found {count} 'or null' in: {line}"

    def test_nested_model_with_arrays(self):
        """Test complex nested model with arrays."""
        from dspy_toon.adapter import _build_toon_schema

        class Name(BaseModel):
            family: str | None = None
            given: list[str] | None = None

        class Patient(BaseModel):
            name: Name | None = None
            age: int | None = None

        schema = _build_toon_schema(Patient)
        # Check array format in nested model
        assert "given[COUNT]:" in schema
        # No duplicate nulls
        assert "null or null" not in schema

    def test_output_schema_primitive_array(self):
        """Test _get_output_schema for primitive arrays."""
        from dspy_toon.adapter import _get_output_schema

        result = _get_output_schema("tags", list[str])
        # The encoder generates a concrete two-item shape example.
        assert "tags[2]:" in result
        assert "tags: [COUNT]" not in result

    def test_output_schema_object_array(self):
        """Test _get_output_schema for object arrays."""
        from dspy_toon.adapter import _get_output_schema

        class Item(BaseModel):
            id: int
            name: str

        result = _get_output_schema("items", list[Item])
        # Should be "items[2]{id,name}:" format (no comma - it's implicit)
        assert "items[2]{id,name}:" in result
        assert "items:" not in result.split("\n")[0]
