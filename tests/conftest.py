# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Pytest configuration and fixtures."""

from typing import Literal

import pytest
from pydantic import BaseModel, Field

# =============================================================================
# Simple Pydantic Models for Testing
# =============================================================================


class SimpleUser(BaseModel):
    """Simple user model."""

    name: str
    age: int


class UserWithDescription(BaseModel):
    """User model with field descriptions."""

    name: str = Field(description="Full name of the user")
    age: int = Field(description="Age in years")
    email: str = Field(description="Email address")


class Address(BaseModel):
    """Address model."""

    street: str
    city: str
    country: Literal["US", "CA", "UK", "DE"]


class UserWithAddress(BaseModel):
    """User with nested address."""

    name: str = Field(description="Full name")
    age: int
    address: Address | None = None


class Product(BaseModel):
    """Product model for tabular data."""

    id: int
    name: str
    price: float
    in_stock: bool


class OrderItem(BaseModel):
    """Order item with nested product reference."""

    product_id: int
    quantity: int
    unit_price: float


class Order(BaseModel):
    """Complex order model."""

    order_id: str
    customer_name: str
    items: list[OrderItem]
    total: float
    status: Literal["pending", "shipped", "delivered"]


class SentimentResult(BaseModel):
    """Sentiment analysis result."""

    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    key_phrases: list[str]


class AnalysisResult(BaseModel):
    """Complex analysis result with multiple fields."""

    title: str
    summary: str
    sentiment: Literal["positive", "negative", "neutral"]
    topics: list[str]
    entities: list[dict]
    confidence: float


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def simple_user_data():
    """Simple user test data."""
    return {"name": "Alice", "age": 30}


@pytest.fixture
def user_list_data():
    """List of users for tabular format testing."""
    return [
        {"id": 1, "name": "Alice", "age": 30},
        {"id": 2, "name": "Bob", "age": 25},
        {"id": 3, "name": "Charlie", "age": 35},
    ]


@pytest.fixture
def nested_data():
    """Nested object data."""
    return {"user": {"name": "Alice", "settings": {"theme": "dark", "notifications": True}}}


@pytest.fixture
def product_list():
    """List of products for tabular testing."""
    return [
        Product(id=1, name="Widget A", price=9.99, in_stock=True),
        Product(id=2, name="Widget B", price=14.50, in_stock=True),
        Product(id=3, name="Widget C", price=19.99, in_stock=False),
    ]


@pytest.fixture
def complex_order():
    """Complex order data."""
    return Order(
        order_id="ORD-001",
        customer_name="John Doe",
        items=[
            OrderItem(product_id=1, quantity=2, unit_price=9.99),
            OrderItem(product_id=2, quantity=1, unit_price=14.50),
        ],
        total=34.48,
        status="pending",
    )
