"""Test doubles: a scripted Claude and fake network lookups. No API key or internet needed."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

from agent_core import LLMUnavailable


def text(t: str):
    return SimpleNamespace(type="text", text=t)


def tool_use(id_: str, name: str, **inp):
    return SimpleNamespace(type="tool_use", id=id_, name=name, input={"reason": "test", **inp})


class Response(SimpleNamespace):
    def model_dump(self):
        return self.dump if hasattr(self, "dump") else {}


def response(*blocks, stop: str = "tool_use", dump: dict | None = None):
    r = Response(content=list(blocks), stop_reason=stop)
    if dump is not None:
        r.dump = dump
    return r


class ScriptedLLM:
    """Returns pre-written responses in order and records every request."""

    available = True

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def create(self, **params):
        # Snapshot: the runner keeps appending to the same messages list after this call.
        self.requests.append({**params, "messages": list(params.get("messages", []))})
        if not self.responses:
            raise AssertionError("ScriptedLLM ran out of responses")
        return self.responses.pop(0)


class BrokenLLM:
    available = False

    def create(self, **params):
        raise LLMUnavailable("simulated outage")


OLD = datetime(1992, 6, 1, tzinfo=timezone.utc)
NOW = datetime(2026, 10, 12, tzinfo=timezone.utc)


def fake_tls(domain: str) -> dict:
    return {"valid": True, "host": domain, "issuer": "Test CA"}


def patch_network(monkeypatch, established=("infosys.com",)) -> None:
    """RDAP says the given domains are decades old and every other domain is unregistered."""
    from pramaan.tools import check_domain as cd

    def rdap(domain):
        if domain in established:
            return OLD
        raise cd.NotRegistered(domain)

    monkeypatch.setattr(cd, "rdap_registration_date", rdap)
    monkeypatch.setattr(cd, "tls_probe", fake_tls)
