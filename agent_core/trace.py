"""Live trace of what the agent is doing.

The runner emits an event at every step (LLM turn, tool call, tool result,
failure, fallback). Events are stored for the final report and pushed to an
optional callback - the Streamlit UI uses that to render the trace live.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable

# Event kinds, in roughly the order they occur.
RUN_START = "run_start"
LLM_REQUEST = "llm_request"
LLM_TEXT = "llm_text"
TOOL_CALL = "tool_call"
TOOL_RESULT = "tool_result"
TOOL_ERROR = "tool_error"
TOOL_SKIPPED = "tool_skipped"
BUDGET = "budget"
FALLBACK = "fallback"
SAFETY_NET = "safety_net"
RUN_END = "run_end"


@dataclass
class TraceEvent:
    kind: str
    message: str
    data: dict = field(default_factory=dict)
    elapsed_s: float = 0.0


class Tracer:
    def __init__(self, callback: Callable[[TraceEvent], None] | None = None) -> None:
        self.events: list[TraceEvent] = []
        self._callback = callback
        self._t0 = time.monotonic()

    def emit(self, kind: str, message: str, **data) -> TraceEvent:
        event = TraceEvent(kind, message, data, round(time.monotonic() - self._t0, 2))
        self.events.append(event)
        if self._callback is not None:
            try:
                self._callback(event)
            except Exception:
                pass  # a broken UI callback must never break the investigation
        return event
