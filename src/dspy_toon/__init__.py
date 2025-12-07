# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""DSPy adapter using TOON (Token-Oriented Object Notation) format.

TOON is a compact, human-readable serialization format optimized for LLM contexts.
This package provides a DSPy adapter that achieves 30-60% token reduction vs JSON
while maintaining readability and structure.

Example:
    >>> import dspy
    >>> from dspy_toon import ToonAdapter
    >>> from pydantic import BaseModel
    >>>
    >>> class UserInfo(BaseModel):
    ...     name: str
    ...     age: int
    ...
    >>> dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=ToonAdapter())
    >>>
    >>> class ExtractUser(dspy.Signature):
    ...     '''Extract user information from text.'''
    ...     text: str = dspy.InputField()
    ...     user: UserInfo = dspy.OutputField()
    >>>
    >>> extractor = dspy.Predict(ExtractUser)
    >>> result = extractor(text="Alice is 30 years old.")
    >>> print(result.user)
    UserInfo(name='Alice', age=30)
"""

from .adapter import ToonAdapter
from .streaming import enable_toon_streaming, is_streaming_enabled
from .toon import ToonDecodeError, decode, encode

__version__ = "0.1.0"
__all__ = [
    "ToonAdapter",
    "encode",
    "decode",
    "ToonDecodeError",
    "enable_toon_streaming",
    "is_streaming_enabled",
]
