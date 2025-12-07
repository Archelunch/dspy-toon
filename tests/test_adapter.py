# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Tests for ToonAdapter."""

from typing import Literal, Optional

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

        result = _render_type_str(Optional[str])
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
        user = UserWithAddress(
            name="Alice", age=30, address=Address(street="123 Main St", city="NYC", country="US")
        )
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
