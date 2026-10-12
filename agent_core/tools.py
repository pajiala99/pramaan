"""Tool contract and safe execution.

A Tool is a plain Python function plus the JSON schema Claude sees. The
runner never calls a tool function directly - it goes through
`execute_safely`, which validates input, enforces a timeout and turns any
exception into an error message. A broken tool therefore costs one
"not checked" entry in the report, never a crashed investigation.
"""
from __future__ import annotations

import concurrent.futures
import copy
from dataclasses import dataclass, field
from typing import Any, Callable

from .evidence import Evidence

_JSON_TYPES = {
    "string": str,
    "array": list,
    "object": dict,
    "integer": int,
    "number": (int, float),
    "boolean": bool,
}


class ToolError(Exception):
    """An expected failure a tool reports cleanly, e.g. 'registry returned garbage'."""


class ToolSkipped(ToolError):
    """The tool cannot run in this mode, e.g. it needs a service that is unavailable.
    Reported as 'skipped', which is different from a check that ran and failed."""


@dataclass
class ToolOutput:
    data: dict  # returned to the LLM so it can plan the next step
    evidence: list[Evidence] = field(default_factory=list)  # recorded in the ledger
    # Evidence groups this run actually covered. None = all of the tool's groups.
    # A tool sets this when part of it failed (e.g. WHOIS down but email checks fine).
    groups_checked: list[str] | None = None


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict, Any], ToolOutput]  # fn(args, ctx) -> ToolOutput
    groups: tuple[str, ...] = ()  # evidence groups this tool can speak to
    timeout_s: float = 30.0

    def api_schema(self) -> dict:
        """Tool definition for the Messages API.

        Every tool gets an extra required `reason` field: the model must say
        why it is calling the tool, which is what the live trace shows. The
        runner strips it before the tool function sees the arguments.
        """
        schema = copy.deepcopy(self.input_schema)
        schema.setdefault("type", "object")
        schema.setdefault("properties", {})["reason"] = {
            "type": "string",
            "description": "One short sentence: why you are calling this tool now.",
        }
        schema["required"] = [*schema.get("required", []), "reason"]
        return {"name": self.name, "description": self.description, "input_schema": schema}


class ToolRegistry:
    def __init__(self, tools: list[Tool]) -> None:
        self._tools = {t.name: t for t in tools}

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def api_schemas(self) -> list[dict]:
        return [t.api_schema() for t in self._tools.values()]


def validate_args(schema: dict, args: dict) -> str | None:
    """Minimal check of required keys and top-level types. Returns an error or None."""
    if not isinstance(args, dict):
        return "arguments must be a JSON object"
    for key in schema.get("required", []):
        if key not in args:
            return f"missing required argument '{key}'"
    for key, value in args.items():
        spec = schema.get("properties", {}).get(key)
        if spec is None:
            return f"unknown argument '{key}'"
        expected = _JSON_TYPES.get(spec.get("type", ""))
        if expected and not isinstance(value, expected):
            return f"argument '{key}' should be of type {spec['type']}"
    return None


def execute_safely(tool: Tool, args: dict, ctx: Any) -> tuple[ToolOutput | None, str | None, bool]:
    """Run a tool with validation and a timeout. Never raises.

    Returns (output, None, False) on success, (None, error, False) on failure
    and (None, reason, True) when the tool was skipped.
    """
    problem = validate_args(tool.input_schema, args)
    if problem:
        return None, f"invalid input: {problem}", False

    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = pool.submit(tool.fn, args, ctx)
    try:
        output = future.result(timeout=tool.timeout_s)
    except concurrent.futures.TimeoutError:
        return None, f"timed out after {tool.timeout_s:.0f}s", False
    except ToolSkipped as e:
        return None, str(e), True
    except ToolError as e:
        return None, str(e), False
    except Exception as e:  # any bug or network failure inside a tool
        return None, f"{type(e).__name__}: {e}", False
    finally:
        # Don't wait for a hung worker thread; it is abandoned, not joined.
        pool.shutdown(wait=False, cancel_futures=True)

    if not isinstance(output, ToolOutput):
        return None, "tool returned an unexpected value", False
    return output, None, False
