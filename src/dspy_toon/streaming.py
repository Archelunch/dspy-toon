# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Streaming support for ToonAdapter.

This module patches DSPy's StreamListener to support ToonAdapter for token-level
streaming. Import this module to enable streaming with ToonAdapter.

Usage:
    >>> import dspy
    >>> from dspy_toon import ToonAdapter
    >>> from dspy_toon.streaming import enable_toon_streaming
    >>>
    >>> # Enable ToonAdapter streaming support
    >>> enable_toon_streaming()
    >>>
    >>> # Configure DSPy
    >>> dspy.configure(lm=dspy.LM("openai/gpt-4o-mini"), adapter=ToonAdapter())
    >>>
    >>> # Now streaming works with ToonAdapter
    >>> predict = dspy.Predict("question -> answer")
    >>> stream_predict = dspy.streamify(
    ...     predict,
    ...     stream_listeners=[dspy.streaming.StreamListener(signature_field_name="answer")],
    ... )
"""

import re
from queue import Queue

from dspy.dsp.utils.settings import settings  # type: ignore[import-untyped]
from dspy.streaming.streaming_listener import ADAPTER_SUPPORT_STREAMING, StreamListener  # type: ignore[import-untyped]

from .adapter import ToonAdapter

# Track if streaming has been enabled
_streaming_enabled = False


def enable_toon_streaming() -> None:
    """Enable streaming support for ToonAdapter.

    This patches DSPy's StreamListener to recognize ToonAdapter patterns.
    Call this once before using streaming with ToonAdapter.
    """
    global _streaming_enabled

    if _streaming_enabled:
        return

    # Add ToonAdapter to supported adapters list
    if ToonAdapter not in ADAPTER_SUPPORT_STREAMING:
        ADAPTER_SUPPORT_STREAMING.append(ToonAdapter)

    # Store original methods
    _original_init = StreamListener.__init__
    _original_flush = StreamListener.flush

    def _patched_init(self, *args, **kwargs):
        """Patched __init__ to add ToonAdapter patterns."""
        _original_init(self, *args, **kwargs)

        # Add ToonAdapter identifier patterns
        # TOON format: "field_name: value" or "field_name:\n  nested content"
        # Response typically starts directly with "field_name: value"
        self.adapter_identifiers["ToonAdapter"] = {
            # Start when we see "field_name:" - may be at start or after newline
            "start_identifier": f"{self.signature_field_name}:",
            # End when we see another field starting (newline + word + colon) or end
            "end_identifier": re.compile(r"\n[a-zA-Z_][a-zA-Z0-9_]*:"),
            # Start indicator - first char of field name
            "start_indicator": self.signature_field_name[0] if self.signature_field_name else "a",
            # Patterns that could form end identifier
            "end_pattern_prefixes": ["\n"],
            "end_pattern_contains": None,
        }

    def _patched_flush(self) -> str:
        """Patched flush to handle ToonAdapter."""
        # Check if using ToonAdapter
        if isinstance(settings.adapter, ToonAdapter):
            last_tokens = "".join(self.field_end_queue.queue)
            self.field_end_queue = Queue()

            # Find the next field boundary (newline followed by field_name:)
            match = re.search(r"\n[a-zA-Z_][a-zA-Z0-9_]*:", last_tokens)
            if match:
                boundary_index = match.start()
            else:
                boundary_index = len(last_tokens)

            return last_tokens[:boundary_index].strip()

        # Fall back to original for other adapters
        return _original_flush(self)

    # Apply patches
    StreamListener.__init__ = _patched_init
    StreamListener.flush = _patched_flush

    _streaming_enabled = True


def is_streaming_enabled() -> bool:
    """Check if ToonAdapter streaming support is enabled."""
    return _streaming_enabled


# Auto-enable when module is imported (optional - can be disabled)
# enable_toon_streaming()
