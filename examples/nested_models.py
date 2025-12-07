#!/usr/bin/env python3
# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Complex nested Pydantic models example with ToonAdapter.

This example demonstrates how ToonAdapter handles deeply nested
Pydantic models with various field types.

Usage:
    export OPENAI_API_KEY="your-key-here"
    python examples/nested_models.py
"""

from typing import Literal

import dspy
from pydantic import BaseModel, Field

from dspy_toon import ToonAdapter, encode

# =============================================================================
# Complex Nested Models
# =============================================================================


class Address(BaseModel):
    """Physical address."""

    street: str = Field(description="Street address")
    city: str = Field(description="City name")
    state: str | None = Field(description="State or province")
    country: str = Field(description="Country code (e.g., US, UK, DE)")
    postal_code: str | None = Field(description="ZIP or postal code")


class ContactInfo(BaseModel):
    """Contact information."""

    email: str = Field(description="Email address")
    phone: str | None = Field(description="Phone number")
    address: Address | None = Field(description="Physical address")


class Skill(BaseModel):
    """Professional skill."""

    name: str = Field(description="Skill name")
    level: Literal["beginner", "intermediate", "advanced", "expert"] = Field(description="Proficiency level")
    years: int | None = Field(description="Years of experience")


class Education(BaseModel):
    """Educational background."""

    institution: str = Field(description="School or university name")
    degree: str = Field(description="Degree or certification")
    field: str = Field(description="Field of study")
    graduation_year: int | None = Field(description="Year of graduation")


class WorkExperience(BaseModel):
    """Work experience entry."""

    company: str = Field(description="Company name")
    title: str = Field(description="Job title")
    start_year: int = Field(description="Start year")
    end_year: int | None = Field(description="End year (null if current)")
    responsibilities: list[str] = Field(description="Key responsibilities")


class CandidateProfile(BaseModel):
    """Complete candidate profile for recruitment."""

    name: str = Field(description="Full name")
    summary: str = Field(description="Professional summary")
    contact: ContactInfo = Field(description="Contact information")
    skills: list[Skill] = Field(description="Technical and soft skills")
    education: list[Education] = Field(description="Educational background")
    experience: list[WorkExperience] = Field(description="Work history")
    languages: list[str] = Field(description="Languages spoken")


# =============================================================================
# Simpler Nested Model for Testing
# =============================================================================


class CompanyInfo(BaseModel):
    """Company information."""

    name: str = Field(description="Company name")
    industry: str = Field(description="Industry sector")
    headquarters: Address = Field(description="Headquarters location")


class JobPosting(BaseModel):
    """Job posting with company details."""

    title: str = Field(description="Job title")
    company: CompanyInfo = Field(description="Company information")
    salary_range: str | None = Field(description="Salary range if mentioned")
    requirements: list[str] = Field(description="Job requirements")
    benefits: list[str] = Field(description="Benefits offered")


# =============================================================================
# DSPy Signatures
# =============================================================================


class ExtractJobPosting(dspy.Signature):
    """Extract job posting information from text."""

    text: str = dspy.InputField(desc="Job posting description")
    job: JobPosting = dspy.OutputField(desc="Extracted job posting")


class ExtractCandidate(dspy.Signature):
    """Extract candidate profile from resume text."""

    resume: str = dspy.InputField(desc="Resume or CV text")
    profile: CandidateProfile = dspy.OutputField(desc="Extracted candidate profile")


# =============================================================================
# TOON Format Demonstration
# =============================================================================


def demonstrate_toon_encoding():
    """Show how complex nested data looks in TOON format."""
    print("\n" + "=" * 60)
    print("TOON ENCODING DEMONSTRATION")
    print("=" * 60)

    # Sample nested data
    sample_data = {
        "name": "Alice Johnson",
        "contact": {
            "email": "alice@example.com",
            "phone": "+1-555-0123",
            "address": {
                "street": "123 Tech Lane",
                "city": "San Francisco",
                "state": "CA",
                "country": "US",
                "postal_code": "94105",
            },
        },
        "skills": [
            {"name": "Python", "level": "expert", "years": 8},
            {"name": "Machine Learning", "level": "advanced", "years": 5},
            {"name": "Leadership", "level": "intermediate", "years": 3},
        ],
    }

    print("\n📋 Sample Data (Python dict):")
    import json

    print(json.dumps(sample_data, indent=2))

    print("\n📝 TOON Encoded:")
    toon_output = encode(sample_data)
    print(toon_output)

    # Compare token counts
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        json_tokens = len(encoding.encode(json.dumps(sample_data)))
        toon_tokens = len(encoding.encode(toon_output))
        savings = (json_tokens - toon_tokens) / json_tokens * 100

        print("\n📊 Token Comparison:")
        print(f"   JSON: {json_tokens} tokens")
        print(f"   TOON: {toon_tokens} tokens")
        print(f"   Savings: {savings:.1f}%")
    except ImportError:
        print("\n(Install tiktoken to see token comparison)")


def extract_job_example():
    """Demonstrate job posting extraction."""
    print("\n" + "=" * 60)
    print("EXAMPLE: Job Posting Extraction")
    print("=" * 60)

    job_text = """
    Senior Software Engineer at TechCorp Inc.

    TechCorp Inc. is a leading technology company in the Cloud Computing industry,
    headquartered at 500 Innovation Drive, Seattle, WA 98101, USA.

    We're looking for an experienced Senior Software Engineer to join our team.
    Salary range: $150,000 - $200,000 per year.

    Requirements:
    - 5+ years of experience in software development
    - Strong proficiency in Python and Go
    - Experience with distributed systems
    - Bachelor's degree in Computer Science or related field

    Benefits:
    - Competitive salary and equity
    - Health, dental, and vision insurance
    - Unlimited PTO
    - Remote work options
    - 401(k) matching
    """

    print(f"\n📄 Input Text:\n{job_text[:200]}...")

    extractor = dspy.Predict(ExtractJobPosting)

    try:
        result = extractor(text=job_text)
        print("\n✅ Extracted Job Posting:")
        print(f"   Title: {result.job.title}")
        print(f"   Company: {result.job.company.name}")
        print(f"   Industry: {result.job.company.industry}")
        print(f"   Location: {result.job.company.headquarters.city}, {result.job.company.headquarters.country}")
        print(f"   Salary: {result.job.salary_range}")
        print(f"   Requirements: {len(result.job.requirements)} items")
        print(f"   Benefits: {len(result.job.benefits)} items")
    except Exception as e:
        print(f"❌ Error: {e}")


def main():
    """Run examples."""
    print("=" * 60)
    print("DSPy-TOON Nested Models Examples")
    print("=" * 60)

    # Always show TOON encoding demo (no LLM required)
    demonstrate_toon_encoding()

    # Try to configure DSPy for LLM examples
    try:
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm, adapter=ToonAdapter())
        print("\n✅ DSPy configured with ToonAdapter")

        # Run LLM example
        extract_job_example()
    except Exception as e:
        print(f"\n⚠️  Could not configure LM: {e}")
        print("   Skipping LLM examples")

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
