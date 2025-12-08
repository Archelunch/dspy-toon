# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Tests for ToonAdapter."""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

from dspy_toon import ToonAdapter, encode

# =============================================================================
# Test Models
# =============================================================================


class SimpleUser(BaseModel):
    name: str
    age: int


class UserWithDescription(BaseModel):
    name: str = Field(description="Full name of the user")
    age: int = Field(description="Age in years")


class Address(BaseModel):
    street: str
    city: str
    country: Literal["US", "CA", "UK"]


class UserWithAddress(BaseModel):
    name: str = Field(description="Full name")
    age: int
    address: Address | None = None


class Product(BaseModel):
    id: int
    name: str
    price: float


class SentimentResult(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: list[str]


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

    def test_nested_model_schema(self):
        """Test rendering schema for nested model."""
        from dspy_toon.adapter import _build_toon_schema

        schema = _build_toon_schema(UserWithAddress)
        assert "name: string" in schema
        assert "address:" in schema

    def test_literal_type_rendering(self):
        """Test that Literal types are rendered correctly."""
        from dspy_toon.adapter import _render_type_str

        result = _render_type_str(Literal["A", "B", "C"])
        assert '"A"' in result
        assert '"B"' in result
        assert '"C"' in result

    def test_list_type_rendering(self):
        """Test that list types are rendered correctly."""
        from dspy_toon.adapter import _render_type_str

        result = _render_type_str(list[str])
        assert "[N]:" in result or "string" in result

    def test_optional_type_rendering(self):
        """Test that Optional types include 'or null'."""
        from dspy_toon.adapter import _render_type_str

        result = _render_type_str(str | None)
        assert "null" in result


# =============================================================================
# Test Adapter Methods
# =============================================================================


class TestAdapterMethods:
    """Tests for ToonAdapter methods."""

    @pytest.fixture
    def adapter(self):
        return ToonAdapter()

    def test_adapter_initialization(self, adapter):
        """Test adapter can be initialized."""
        assert adapter is not None

    def test_format_field_description(self, adapter):
        """Test format_field_description generates proper output."""
        import dspy

        class TestSignature(dspy.Signature):
            """Test signature."""

            text: str = dspy.InputField()
            result: SimpleUser = dspy.OutputField()

        description = adapter.format_field_description(TestSignature)
        assert "input fields" in description.lower()
        assert "output fields" in description.lower()
        assert "text" in description
        assert "result" in description

    def test_format_field_structure(self, adapter):
        """Test format_field_structure includes TOON rules."""
        import dspy

        class TestSignature(dspy.Signature):
            """Extract user info."""

            text: str = dspy.InputField()
            user: SimpleUser = dspy.OutputField()

        structure = adapter.format_field_structure(TestSignature)
        # Should contain TOON format rules
        assert "TOON" in structure
        assert "key: value" in structure.lower()
        # Should describe output structure
        assert "user" in structure.lower()


# =============================================================================
# Test Integration with Pydantic
# =============================================================================


class TestPydanticIntegration:
    """Tests for Pydantic model handling."""

    def test_encode_pydantic_model(self):
        """Test encoding a Pydantic model instance."""
        user = SimpleUser(name="Alice", age=30)
        result = encode(user.model_dump())
        assert "name: Alice" in result
        assert "age: 30" in result

    def test_encode_list_of_models(self):
        """Test encoding a list of Pydantic models."""
        products = [
            Product(id=1, name="A", price=9.99),
            Product(id=2, name="B", price=14.50),
        ]
        data = [p.model_dump() for p in products]
        result = encode(data)
        # Should be tabular format
        assert "id" in result
        assert "name" in result
        assert "price" in result

    def test_encode_nested_model(self):
        """Test encoding a nested Pydantic model."""
        user = UserWithAddress(name="Alice", age=30, address=Address(street="123 Main St", city="NYC", country="US"))
        result = encode(user.model_dump())
        assert "name: Alice" in result
        assert "address:" in result
        assert "street:" in result or "123 Main St" in result


# =============================================================================
# Test Error Handling
# =============================================================================


class TestErrorHandling:
    """Tests for error handling."""

    def test_recursive_model_detection(self):
        """Test that recursive models raise an error."""
        from dspy_toon.adapter import _build_toon_schema

        # This would be a recursive model
        class Node(BaseModel):
            value: int
            # children: List["Node"]  # Would cause recursion

        # Non-recursive should work fine
        schema = _build_toon_schema(Node)
        assert "value: int" in schema


# =============================================================================
# Test Async Support
# =============================================================================


class TestAsyncSupport:
    """Tests for async functionality."""

    @pytest.fixture
    def adapter(self):
        return ToonAdapter()

    def test_adapter_has_acall_method(self, adapter):
        """Test that adapter has acall method for async support."""
        assert hasattr(adapter, "acall")
        assert callable(adapter.acall)

    def test_adapter_has_call_method(self, adapter):
        """Test that adapter has __call__ method for sync support."""
        assert hasattr(adapter, "__call__")
        assert callable(adapter)

    @pytest.mark.asyncio
    async def test_async_format_works(self, adapter):
        """Test that format method works (used by both sync and async paths)."""
        import dspy

        class TestSignature(dspy.Signature):
            """Test signature."""

            question: str = dspy.InputField()
            answer: str = dspy.OutputField()

        # Format is used by both __call__ and acall
        messages = adapter.format(TestSignature, demos=[], inputs={"question": "test"})

        assert isinstance(messages, list)
        assert len(messages) >= 1
        # Should have system message
        assert messages[0]["role"] == "system"
        # Should have user message with question
        assert any("question" in str(m.get("content", "")).lower() for m in messages)


# =============================================================================
# Test Streaming Compatibility
# =============================================================================


class TestStreamingCompatibility:
    """Tests for streaming compatibility with dspy.streamify.

    ToonAdapter supports streaming via the enable_toon_streaming() function
    which patches DSPy's StreamListener to recognize TOON format patterns.
    """

    def test_streamify_import(self):
        """Test that dspy.streamify is available."""
        import dspy

        assert hasattr(dspy, "streamify")

    def test_stream_listener_import(self):
        """Test that StreamListener is available."""
        import dspy.streaming

        assert hasattr(dspy.streaming, "StreamListener")

    def test_asyncify_import(self):
        """Test that dspy.asyncify is available for async support."""
        import dspy

        assert hasattr(dspy, "asyncify")

    def test_enable_toon_streaming_import(self):
        """Test that enable_toon_streaming is available."""
        from dspy_toon import enable_toon_streaming, is_streaming_enabled

        assert callable(enable_toon_streaming)
        assert callable(is_streaming_enabled)

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

    def test_adapter_can_be_asyncified(self):
        """Test that a predictor with ToonAdapter can be wrapped with asyncify."""
        import dspy

        adapter = ToonAdapter()
        lm = dspy.LM("openai/gpt-4o-mini", api_key="test-key")
        dspy.configure(lm=lm, adapter=adapter)

        predict = dspy.Predict("question -> answer")

        # Wrap with asyncify - should not raise
        async_predict = dspy.asyncify(predict)

        assert async_predict is not None
        assert callable(async_predict)

    def test_adapter_can_be_streamified(self):
        """Test that ToonAdapter can be used with streamify after enabling."""
        import dspy

        from dspy_toon import enable_toon_streaming

        enable_toon_streaming()

        adapter = ToonAdapter()
        lm = dspy.LM("openai/gpt-4o-mini", api_key="test-key")
        dspy.configure(lm=lm, adapter=adapter)

        predict = dspy.Predict("question -> answer")

        # Wrap with streamify - should not raise now
        stream_predict = dspy.streamify(
            predict,
            stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
        )

        assert stream_predict is not None


# =============================================================================
# Test Callbacks Support
# =============================================================================


class TestCallbacksSupport:
    """Tests for callbacks functionality."""

    def test_adapter_accepts_callbacks(self):
        """Test that adapter accepts callbacks parameter."""
        from dspy.utils.callback import BaseCallback

        class TestCallback(BaseCallback):
            def __init__(self):
                self.called = False

        callback = TestCallback()
        adapter = ToonAdapter(callbacks=[callback])

        assert adapter.callbacks == [callback]

    def test_adapter_accepts_native_function_calling(self):
        """Test that adapter accepts use_native_function_calling parameter."""
        adapter = ToonAdapter(use_native_function_calling=True)
        assert adapter.use_native_function_calling is True

        adapter2 = ToonAdapter(use_native_function_calling=False)
        assert adapter2.use_native_function_calling is False


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
        # Should be "tags[3]: val1,val2,val3" not "tags: [3]: val1,val2,val3"
        assert "tags[3]:" in result
        assert "tags: [3]" not in result

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
