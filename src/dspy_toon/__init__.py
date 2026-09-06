# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""DSPy adapter using TOON (Token-Oriented Object Notation) format.

TOON is a compact, human-readable serialization format optimized for LLM contexts.
This package provides a DSPy adapter for compact structured inputs and outputs.
Token savings depend on data shape, tokenizer, and prompt overhead.

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

from importlib.metadata import version

from .adapter import ToonAdapter
from .streaming import enable_toon_streaming, is_streaming_enabled
from .toon import ToonDecodeError, decode, encode

__version__ = version("dspy-toon")
__all__ = [
    "ToonAdapter",
    "encode",
    "decode",
    "ToonDecodeError",
    "enable_toon_streaming",
    "is_streaming_enabled",
]
