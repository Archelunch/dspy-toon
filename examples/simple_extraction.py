#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Simple structured extraction example with ToonAdapter.

This example demonstrates basic usage of ToonAdapter for extracting
structured information from text using Pydantic models.

Usage:
    # Set your API key first
    export OPENAI_API_KEY="your-key-here"

    # Run the example
    python examples/simple_extraction.py
"""

from typing import Literal

import dspy
from pydantic import BaseModel, Field

from dspy_toon import ToonAdapter

# =============================================================================
# Define Pydantic Models
# =============================================================================


class PersonInfo(BaseModel):
    """Extracted person information."""

    name: str = Field(description="Full name of the person")
    age: int | None = Field(description="Age in years, if mentioned")
    occupation: str | None = Field(description="Job title or profession")
    location: str | None = Field(description="City or location")


class SentimentResult(BaseModel):
    """Sentiment analysis result."""

    sentiment: Literal["positive", "negative", "neutral"] = Field(description="Overall sentiment of the text")
    confidence: float = Field(
        description="Confidence score between 0 and 1",
        ge=0.0,
        le=1.0,
    )
    reasoning: str = Field(description="Brief explanation for the sentiment")


# =============================================================================
# Define DSPy Signatures
# =============================================================================


class ExtractPerson(dspy.Signature):
    """Extract person information from the given text."""

    text: str = dspy.InputField(desc="Text containing information about a person")
    person: PersonInfo = dspy.OutputField(desc="Extracted person information")


class AnalyzeSentiment(dspy.Signature):
    """Analyze the sentiment of the given text."""

    text: str = dspy.InputField(desc="Text to analyze")
    result: SentimentResult = dspy.OutputField(desc="Sentiment analysis result")


# =============================================================================
# Example Usage
# =============================================================================


def extract_person_example():
    """Demonstrate person extraction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 1: Person Extraction")
    print("=" * 60)

    # Sample texts
    texts = [
        "Alice Johnson is a 35-year-old software engineer based in San Francisco.",
        "Dr. Bob Smith, aged 52, works as a cardiologist at Boston Medical Center.",
        "Meet Sarah Chen, a talented artist from New York who has been painting for 20 years.",
    ]

    extractor = dspy.Predict(ExtractPerson)

    for text in texts:
        print(f"\n📄 Input: {text}")
        try:
            result = extractor(text=text)
            print(f"✅ Extracted: {result.person}")
        except Exception as e:
            print(f"❌ Error: {e}")


def analyze_sentiment_example():
    """Demonstrate sentiment analysis."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Sentiment Analysis")
    print("=" * 60)

    texts = [
        "I absolutely love this product! It exceeded all my expectations.",
        "The service was terrible. I waited for 2 hours and nobody helped me.",
        "The weather today is cloudy with a chance of rain in the afternoon.",
    ]

    analyzer = dspy.Predict(AnalyzeSentiment)

    for text in texts:
        print(f"\n📄 Input: {text[:60]}...")
        try:
            result = analyzer(text=text)
            emoji = {"positive": "😊", "negative": "😞", "neutral": "😐"}.get(result.result.sentiment, "❓")
            print(f"{emoji} Sentiment: {result.result.sentiment}")
            print(f"   Confidence: {result.result.confidence:.2f}")
            print(f"   Reasoning: {result.result.reasoning}")
        except Exception as e:
            print(f"❌ Error: {e}")


def main():
    """Run all examples."""
    print("=" * 60)
    print("DSPy-TOON Simple Extraction Examples")
    print("=" * 60)

    # Configure DSPy with ToonAdapter
    # Note: Replace with your actual model and API key
    try:
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm, adapter=ToonAdapter())
        print("\n✅ DSPy configured with ToonAdapter")
    except Exception as e:
        print(f"\n⚠️  Could not configure LM: {e}")
        print("   Running in demo mode (no actual LLM calls)")
        return

    # Run examples
    extract_person_example()
    analyze_sentiment_example()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
