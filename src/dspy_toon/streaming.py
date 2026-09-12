# Copyright (c) 2025 dspy-toon
# SPDX-License-Identifier: MIT
"""Opt-in scalar-field streaming for DSPy's StreamListener.

Chunks are raw TOON text (including string quotes and escapes). The final DSPy
Prediction contains decoded, validated values. Array/table fields are delivered
in the final Prediction, not incrementally by this listener.
"""

import re
from queue import Queue
from typing import Any

from dspy.dsp.utils.settings import settings  # type: ignore[import-untyped]
from dspy.streaming.messages import StreamResponse  # type: ignore[import-untyped]
from dspy.streaming.streaming_listener import ADAPTER_SUPPORT_STREAMING, StreamListener  # type: ignore[import-untyped]

from .adapter import ToonAdapter

_streaming_enabled = False
_FIELD_BOUNDARY = re.compile(r"\n[a-zA-Z_][a-zA-Z0-9_]*(?=:|\[)")


def enable_toon_streaming() -> None:
    """Enable scalar TOON streaming; call once before creating StreamListeners.

    DSPy 3.3.1 has no custom-listener registration API, so this opt-in shim
    delegates other adapters to their original methods.
    """
    global _streaming_enabled
    if _streaming_enabled:
        return

    original_init = StreamListener.__init__
    original_receive = StreamListener.receive
    original_flush = StreamListener.flush

    def init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.adapter_identifiers["ToonAdapter"] = {
            "start_identifier": f"{self.signature_field_name}:",
            "end_identifier": _FIELD_BOUNDARY,
        }

    def receive(self: Any, chunk: Any) -> Any:
        if not isinstance(settings.adapter, ToonAdapter):
            return original_receive(self, chunk)
        if self.stream_end:
            if not self.allow_reuse:
                return None
            self.stream_start = self.stream_end = self.cache_hit = False
            self.field_start_queue = []
            self.field_end_queue = Queue()
        content = chunk.choices[0].delta.content if chunk.choices else None
        if not content:
            return None
        if not self.stream_start:
            self.field_start_queue.append(content)
            buffered = "".join(self.field_start_queue)
            start = re.search(rf"(?:^|\n){re.escape(self.signature_field_name)}: ?", buffered)
            if start is None or start.end() == len(buffered):
                return None
            self.stream_start = True
            self.field_start_queue = []
            content = buffered[start.end() :]
        self.field_end_queue.put(content)
        buffered = "".join(self.field_end_queue.queue)
        self.field_end_queue = Queue()
        boundary = _FIELD_BOUNDARY.search(buffered)
        if boundary:
            token = buffered[: boundary.start()]
            self.stream_end = True
        else:
            # Keep a possible next field's entire header until it resolves,
            # even when the newline and field name arrive in separate chunks.
            newline = buffered.rfind("\n")
            token = buffered if newline < 0 else buffered[:newline]
            if newline >= 0:
                self.field_end_queue.put(buffered[newline:])
        if token or self.stream_end:
            return StreamResponse(self.predict_name, self.signature_field_name, token, is_last_chunk=self.stream_end)
        return None

    def flush(self: Any) -> str:
        if not isinstance(settings.adapter, ToonAdapter):
            return original_flush(self)
        buffered = "".join(self.field_end_queue.queue)
        self.field_end_queue = Queue()
        boundary = _FIELD_BOUNDARY.search(buffered)
        return buffered[: boundary.start()] if boundary else buffered

    StreamListener.__init__ = init
    StreamListener.receive = receive
    StreamListener.flush = flush
    if ToonAdapter not in ADAPTER_SUPPORT_STREAMING:
        ADAPTER_SUPPORT_STREAMING.append(ToonAdapter)
    _streaming_enabled = True


def is_streaming_enabled() -> bool:
    """Return whether the opt-in streaming shim has been installed."""
    return _streaming_enabled
