"""Settings, overridable with environment variables (or a .env file)."""
from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    model: str = os.getenv("PRAMAAN_MODEL", "claude-opus-5-5")
    effort: str = os.getenv("PRAMAAN_EFFORT", "medium")
    max_tool_calls: int = _int_env("PRAMAAN_MAX_TOOL_CALLS", 8)  # the agent's tool-call budget
    max_turns: int = _int_env("PRAMAAN_MAX_TURNS", 10)  # model turns before the loop is stopped
    web_search_max_uses: int = _int_env("PRAMAAN_WEB_SEARCH_MAX_USES", 4)  # searches per web tool call
    web_search_tool: str = "web_search_20260209"  # Claude's server-side web search tool
    max_domains_checked: int = 3  # network lookups per check_domain call
