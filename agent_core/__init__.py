"""Reusable agent core: nothing in this package knows about job offers.

To reuse it for another problem, supply your own tools, evidence groups,
rules, system prompt and playbook (see pramaan/investigate.py for an example).
"""
from .evidence import Citation, Evidence, EvidenceLedger
from .llm import ClaudeLLM, LLMUnavailable, OfflineLLM
from .rules import Decision, Rule, decide
from .runner import AgentRunner, Budget, RunResult
from .tools import Tool, ToolError, ToolOutput, ToolRegistry, ToolSkipped
from .trace import TraceEvent

__all__ = [
    "AgentRunner", "Budget", "Citation", "ClaudeLLM", "Decision", "Evidence", "EvidenceLedger",
    "LLMUnavailable", "OfflineLLM", "Rule", "RunResult", "Tool", "ToolError", "ToolOutput",
    "ToolRegistry", "ToolSkipped", "TraceEvent", "decide",
]
