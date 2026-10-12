"""Sub-calls to Claude with its server-side web_search tool.

verify_vacancy and web_reputation each make one focused Claude request with
web_search enabled and ask for a small JSON answer. The model reads the
pages; our code then *checks its claims*:

  * every URL it reports must be one the search tool actually returned in
    this request (`result_urls`) - a URL it made up is dropped;
  * the quoted snippets (`citations`) it cited are kept so other code can
    check that an identifier really appears in the source.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit, urlunsplit

from agent_core import ToolError, ToolSkipped

MAX_CONTINUATIONS = 3  # pause_turn resumptions per sub-call


@dataclass
class SearchOutcome:
    data: dict  # the JSON object the model returned
    result_urls: dict[str, str] = field(default_factory=dict)  # normalized url -> page title
    citations: list[tuple[str, str]] = field(default_factory=list)  # (normalized url, cited text)
    queries: list[str] = field(default_factory=list)  # searches the model ran

    def verified(self, url: str | None) -> str | None:
        """The URL if the search tool really returned it, else None."""
        if not url:
            return None
        return url if normalize_url(url) in self.result_urls else None

    def text_for(self, url: str) -> str:
        """Title + every cited snippet for a URL, for exact-identifier checks."""
        key = normalize_url(url)
        return " ".join([self.result_urls.get(key, ""), *(t for u, t in self.citations if u == key)])


def normalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    host = (parts.hostname or "").lower().removeprefix("www.")
    return urlunsplit((parts.scheme.lower() or "https", host, parts.path.rstrip("/"), parts.query, ""))


def _harvest(node, outcome: SearchOutcome) -> None:
    """Walk a response dump and collect search results, citations and queries, wherever they are nested."""
    if isinstance(node, dict):
        kind = node.get("type")
        if kind == "web_search_result" and node.get("url"):
            outcome.result_urls[normalize_url(node["url"])] = node.get("title") or ""
        elif kind == "web_search_result_location" and node.get("url"):
            outcome.citations.append((normalize_url(node["url"]), node.get("cited_text") or ""))
        elif kind == "server_tool_use" and node.get("name") == "web_search":
            query = (node.get("input") or {}).get("query")
            if query:
                outcome.queries.append(query)
        for value in node.values():
            _harvest(value, outcome)
    elif isinstance(node, list):
        for item in node:
            _harvest(item, outcome)


def parse_json_object(text: str) -> dict:
    """First JSON object in the model's text (bare, fenced in ```json, or embedded)."""
    candidates = [text.strip()]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    if fenced:
        candidates.append(fenced.group(1))
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start:end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            continue
    raise ToolError("web search answer was not valid JSON")


def web_search_json(llm, prompt: str, *, tool_type: str, max_uses: int, allowed_domains: list[str] | None = None) -> SearchOutcome:
    """Run one Claude request with web_search and return its parsed JSON answer + harvested sources."""
    if not getattr(llm, "available", False):
        raise ToolSkipped("web search needs the Claude API, which is unavailable in this run")

    tool: dict = {"type": tool_type, "name": "web_search", "max_uses": max_uses}
    if allowed_domains:
        tool["allowed_domains"] = allowed_domains

    outcome = SearchOutcome(data={})
    messages: list[dict] = [{"role": "user", "content": prompt}]
    response = None
    for _ in range(MAX_CONTINUATIONS + 1):
        response = llm.create(messages=messages, tools=[tool])  # LLMUnavailable propagates -> tool error
        _harvest(response.model_dump(), outcome)
        if response.stop_reason != "pause_turn":
            break
        # Server-side search loop paused: send the partial turn back and the API resumes it.
        messages = [messages[0], {"role": "assistant", "content": response.content}]

    text = "\n".join(b.text for b in response.content if b.type == "text")
    outcome.data = parse_json_object(text)
    return outcome
