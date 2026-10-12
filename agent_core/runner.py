"""The agent loop: Claude plans, tools act, the ledger remembers.

    user message ──▶ Claude ──tool_use──▶ execute_safely ──▶ ledger + trace
                       ▲                                      │
                       └────────────── tool_result ◀──────────┘

Safety properties enforced here, independent of any problem domain:

* Budget       - at most `max_tool_calls` tool executions and `max_turns`
                 model turns. Once the budget is spent, extra calls are refused
                 and the model is asked to finish with tool_choice "none".
* Graceful     - a tool that errors or times out returns an error tool_result;
  failure        the model carries on, and the ledger marks its groups unchecked.
* Fallback     - if Claude itself is unavailable (no key, network, refusal),
                 a deterministic playbook runs the same tools in a fixed order.
* Safety net   - tools listed in `ensure` always run, even if the model
                 skipped them, so the rules never decide on missing basics.
* No double    - an identical repeated call returns the cached result and the
  counting       ledger ignores facts it has already recorded.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from . import trace as T
from .evidence import EvidenceLedger
from .llm import LLMUnavailable
from .tools import ToolRegistry, execute_safely
from .trace import Tracer

MAX_RESULT_CHARS = 8000  # cap on a tool_result sent back to the model


@dataclass
class Budget:
    max_tool_calls: int = 8
    max_turns: int = 10


@dataclass
class RunResult:
    ledger: EvidenceLedger
    events: list[T.TraceEvent]
    mode: str  # "agent" (Claude planned) or "playbook" (fixed fallback order)
    stop_reason: str  # finished | budget_exhausted | max_turns | max_tokens | llm_unavailable: ...
    agent_calls: int  # tool calls requested by Claude and executed
    extra_calls: int  # playbook + safety-net calls
    notes: str  # Claude's closing summary (empty in playbook mode)
    tool_outputs: dict[str, list[dict]] = field(default_factory=dict)  # tool name -> data of each successful call


# A playbook receives `call(name, args) -> data | None` and runs tools in a fixed order.
Playbook = Callable[[Callable[[str, dict], dict | None]], None]


class AgentRunner:
    def __init__(
        self,
        llm: Any,
        registry: ToolRegistry,
        *,
        system_prompt: str,
        budget: Budget | None = None,
        on_event: Callable[[T.TraceEvent], None] | None = None,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.system_prompt = system_prompt
        self.budget = budget or Budget()
        self.on_event = on_event

    # ------------------------------------------------------------------ public

    def run(
        self,
        user_message: str,
        ctx: Any,
        *,
        playbook: Playbook | None = None,
        ensure: list[tuple[str, dict]] = (),
    ) -> RunResult:
        self.ledger = EvidenceLedger()
        self.tracer = Tracer(self.on_event)
        self._cache: dict[str, tuple[dict, bool]] = {}
        self._outputs: dict[str, list[dict]] = {}
        self._extra_calls = 0

        self.tracer.emit(T.RUN_START, "Investigation started", budget=self.budget.max_tool_calls)
        mode, notes, agent_calls = "agent", "", 0
        try:
            notes, stop_reason, agent_calls = self._agent_loop(user_message, ctx)
        except LLMUnavailable as e:
            mode, stop_reason = "playbook", f"llm_unavailable: {e}"
            self.tracer.emit(T.FALLBACK, f"Claude unavailable ({e}). Running the fixed playbook instead.")
            if playbook is not None:
                playbook(lambda name, args: self._playbook_call(name, args, ctx))

        for name, args in ensure:
            if self.ledger.tool_status.get(name) != "ok":
                self.tracer.emit(T.SAFETY_NET, f"{name} has not run successfully - running it to guarantee coverage")
                self._extra_calls += 1
                self._call(name, args, ctx, origin="safety-net")

        self.tracer.emit(
            T.RUN_END,
            f"Done ({mode} mode): {agent_calls} agent tool call(s), {self._extra_calls} extra, "
            f"{len(self.ledger.items)} evidence item(s)",
            stop_reason=stop_reason,
        )
        return RunResult(
            ledger=self.ledger,
            events=self.tracer.events,
            mode=mode,
            stop_reason=stop_reason,
            agent_calls=agent_calls,
            extra_calls=self._extra_calls,
            notes=notes,
            tool_outputs=self._outputs,
        )

    # ------------------------------------------------------------- the loop

    def _agent_loop(self, user_message: str, ctx: Any) -> tuple[str, str, int]:
        messages: list[dict] = [{"role": "user", "content": user_message}]
        tools = self.registry.api_schemas()
        calls = 0

        for turn in range(1, self.budget.max_turns + 1):
            budget_left = self.budget.max_tool_calls - calls
            params: dict = {"system": self.system_prompt, "tools": tools, "messages": messages}
            if budget_left <= 0:
                params["tool_choice"] = {"type": "none"}  # out of budget: answer in text only
            self.tracer.emit(T.LLM_REQUEST, f"Turn {turn}: asking Claude for the next step", budget_left=budget_left)

            response = self.llm.create(**params)
            # Append the full content unchanged (thinking blocks included); history is never edited.
            messages.append({"role": "assistant", "content": response.content})

            text = "\n".join(b.text for b in response.content if b.type == "text").strip()
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if text:
                self.tracer.emit(T.LLM_TEXT, text)

            if response.stop_reason == "max_tokens":
                return text, "max_tokens", calls
            if not tool_uses:
                return text, ("finished" if budget_left > 0 else "budget_exhausted"), calls

            results = []
            for block in tool_uses:
                args = dict(block.input or {})
                reason = str(args.pop("reason", ""))
                if calls >= self.budget.max_tool_calls:
                    self.tracer.emit(T.BUDGET, f"Tool budget ({self.budget.max_tool_calls}) spent - refused {block.name}")
                    payload, ok = {
                        "status": "refused",
                        "error": "Tool-call budget exhausted. Write your final summary now.",
                    }, False
                else:
                    calls += 1
                    payload, ok = self._call(block.name, args, ctx, origin="agent", reason=reason, call_no=calls)
                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(payload, default=str)[:MAX_RESULT_CHARS],
                    "is_error": not ok,
                })
            # All results for one assistant turn go back in a single user message.
            messages.append({"role": "user", "content": results})

        self.tracer.emit(T.BUDGET, f"Stopped after {self.budget.max_turns} turns")
        return "", "max_turns", calls

    # ------------------------------------------------------ tool execution

    def _playbook_call(self, name: str, args: dict, ctx: Any) -> dict | None:
        self._extra_calls += 1
        payload, ok = self._call(name, args, ctx, origin="playbook")
        return payload.get("data") if ok else None

    def _call(self, name: str, args: dict, ctx: Any, *, origin: str, reason: str = "", call_no: int | None = None):
        """Execute one tool call; returns (payload for the model, ok)."""
        tool = self.registry.get(name)
        if tool is None:
            self.tracer.emit(T.TOOL_ERROR, f"Unknown tool '{name}'", tool=name)
            return {"status": "error", "error": f"unknown tool '{name}'"}, False

        cache_key = name + json.dumps(args, sort_keys=True, default=str)
        if cache_key in self._cache:
            self.tracer.emit(T.TOOL_CALL, f"{name}: identical repeat call - returning cached result", tool=name, origin=origin)
            return self._cache[cache_key]

        self.tracer.emit(T.TOOL_CALL, f"Calling {name}", tool=name, args=args, reason=reason, origin=origin, call_no=call_no)
        output, error, skipped = execute_safely(tool, args, ctx)

        if skipped:
            self.ledger.record_tool(name, ok=False, groups=tool.groups, skipped=True)
            self.tracer.emit(T.TOOL_SKIPPED, f"{name} skipped: {error}", tool=name, error=error)
            return {
                "status": "skipped",
                "error": error,
                "note": "This check did not run and is recorded as 'skipped', not as evidence.",
            }, False

        if error is not None:
            self.ledger.record_tool(name, ok=False, groups=tool.groups)
            self.tracer.emit(T.TOOL_ERROR, f"{name} failed: {error} - continuing without it", tool=name, error=error)
            # Failures are not cached, so a later retry (e.g. by the safety net) can still succeed.
            return {
                "status": "error",
                "error": error,
                "note": "This check could not be completed and is recorded as 'not checked', not as evidence.",
            }, False

        added = [ev for ev in (self.ledger.add(e, tool=name) for e in output.evidence) if ev is not None]
        groups = output.groups_checked if output.groups_checked is not None else tool.groups
        self.ledger.record_tool(name, ok=True, groups=groups)
        self._outputs.setdefault(name, []).append(output.data)
        self.tracer.emit(
            T.TOOL_RESULT,
            f"{name}: {len(added)} new evidence item(s)",
            tool=name,
            evidence=[e.brief() for e in added],
        )
        result = ({"status": "ok", "data": output.data, "evidence": [e.brief() for e in added]}, True)
        self._cache[cache_key] = result
        return result
