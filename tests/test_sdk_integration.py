"""The real Anthropic SDK against a fake Messages API (mocked HTTP transport).

The other tests use a scripted stand-in for ClaudeLLM. These go one layer
deeper: the actual SDK builds and serializes every request - including the
assistant turns (thinking, tool_use, server tool results) we echo back - and
parses responses shaped like real API output. No key or network needed.
"""
import json

import anthropic
import httpx2
import pytest
from anthropic import DefaultHttpxClient

from agent_core import ClaudeLLM
from agent_core.llm import FALLBACK_BETA
from pramaan.case import JobOfferCase
from pramaan.investigate import investigate

from .fakes import patch_network

SCAM = JobOfferCase(**json.loads(open("samples/1_scam_amazon_wfh.json").read())["case"])
THINKING = {"type": "thinking", "thinking": "", "signature": "sig-abc"}
COMPLAINT_URL = "https://consumer-complaints.example/amazon-hr-98765-43210"


def message(content, stop_reason):
    return {
        "id": "msg_test", "type": "message", "role": "assistant", "model": "claude-opus-5-5",
        "content": content, "stop_reason": stop_reason, "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def tool_use(id_, name, **inp):
    return {"type": "tool_use", "id": id_, "name": name, "input": {"reason": "test", **inp}}


def search_turn(query, results, answer=None, cited=None, stop="end_turn"):
    """A web-search response: server_tool_use + results (+ the final JSON text with citations)."""
    content = [
        {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search", "input": {"query": query}},
        {"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1", "content": [
            {"type": "web_search_result", "url": u, "title": t, "encrypted_content": "enc", "page_age": None}
            for u, t in results
        ]},
    ]
    if answer is not None:
        citations = [
            {"type": "web_search_result_location", "url": u, "title": "t", "cited_text": c, "encrypted_index": "i"}
            for u, c in (cited or [])
        ]
        content.append({"type": "text", "text": json.dumps(answer), "citations": citations or None})
    return message(content, stop)


class FakeMessagesAPI:
    """Answers planner requests in order and web-search sub-calls by prompt type."""

    def __init__(self, planner, searches, errors=()):
        self.planner = list(planner)
        self.searches = {k: list(v) for k, v in searches.items()}
        self.errors = list(errors)  # (status, error_type, message) returned first, in order
        self.requests: list[tuple[dict, dict]] = []

    def __call__(self, request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        self.requests.append((dict(request.headers), body))
        if self.errors:
            status, kind, msg = self.errors.pop(0)
            return httpx2.Response(status, json={"type": "error", "error": {"type": kind, "message": msg}})
        if any(t.get("type", "").startswith("web_search") for t in body.get("tools", [])):
            kind = "vacancy" if "job vacancy" in body["messages"][0]["content"] else "reputation"
            return httpx2.Response(200, json=self.searches[kind].pop(0))
        return httpx2.Response(200, json=self.planner.pop(0))

    def planner_requests(self):
        return [(h, b) for h, b in self.requests if not any(t.get("type", "").startswith("web_search") for t in b.get("tools", []))]

    def search_requests(self):
        return [(h, b) for h, b in self.requests if any(t.get("type", "").startswith("web_search") for t in b.get("tools", []))]


def llm_for(api: FakeMessagesAPI) -> ClaudeLLM:
    client = anthropic.Anthropic(
        api_key="test-key", max_retries=0, http_client=DefaultHttpxClient(transport=httpx2.MockTransport(api))
    )
    return ClaudeLLM(client=client)


@pytest.fixture(autouse=True)
def offline_network(monkeypatch):
    patch_network(monkeypatch)


def full_investigation_api(reputation_pauses=False):
    planner = [
        message([THINKING, tool_use("toolu_1", "extract_entities"), tool_use("toolu_2", "detect_red_flags")], "tool_use"),
        message([THINKING,
                 tool_use("toolu_3", "check_domain", company="Amazon"),
                 tool_use("toolu_4", "verify_vacancy", company="Amazon", role="Data Entry Executive"),
                 tool_use("toolu_5", "web_reputation", company="Amazon")], "tool_use"),
        message([THINKING, {"type": "text", "text": "- Checked all five sources; key evidence E1, E9.", "citations": None}], "end_turn"),
    ]
    vacancy = search_turn(
        "Amazon careers Data Entry Executive",
        [("https://www.amazon.jobs/en/", "Amazon Jobs")],
        {"official_domain": "amazon.jobs", "careers_url": "https://www.amazon.jobs/en/", "role_listed": "no",
         "listing_url": None, "fraud_advisory_url": None, "notes": "No such role."},
    )
    results = [(COMPLAINT_URL, "Fake Amazon HR on 98765 43210 asks for Rs 499 fee")]
    answer = {"reports": [{"identifier": "+919876543210", "about": "identifier", "url": COMPLAINT_URL,
                           "claim": "Candidates report this number asking for a registration fee."}]}
    if reputation_pauses:
        reputation = [search_turn("98765 43210 scam", results, stop="pause_turn"),
                      search_turn("98765 43210 fraud", results, answer, cited=[(COMPLAINT_URL, "98765 43210 asked me to pay")])]
    else:
        reputation = [search_turn("98765 43210 scam", results, answer, cited=[(COMPLAINT_URL, "98765 43210 asked me to pay")])]
    return FakeMessagesAPI(planner, {"vacancy": [vacancy], "reputation": reputation})


def test_full_agent_run_through_the_real_sdk():
    api = full_investigation_api()
    report = investigate(SCAM, llm=llm_for(api))

    assert report.run.mode == "agent" and report.run.agent_calls == 5 and report.run.extra_calls == 0
    assert report.decision.verdict == "Likely scam"
    assert all(report.group_status(g) == "checked" for g in report.decision.facts["checked_groups"])
    assert len(report.decision.facts["checked_groups"]) == 5
    signals = {e.signal: e.severity for e in report.run.ledger.items}
    assert signals["IDENTIFIER_REPORTED"] == "high"  # number appears in the result title
    assert signals["OFFICIAL_DOMAIN_MISMATCH"] == "medium"
    assert signals["ROLE_NOT_LISTED"] == "low"
    assert "E1" in report.run.notes


def test_requests_have_the_shape_the_api_expects():
    api = full_investigation_api()
    investigate(SCAM, llm=llm_for(api))
    (headers, first), (_, second), (_, third) = api.planner_requests()

    assert first["model"] == "claude-opus-5-5"
    assert first["output_config"] == {"effort": "medium"}
    assert first["fallbacks"] == "default" and FALLBACK_BETA in headers["anthropic-beta"]
    assert "thinking" not in first  # Opus 5.5: thinking is always on, never configured
    assert all("reason" in t["input_schema"]["required"] for t in first["tools"])
    assert "<offer>" in first["messages"][0]["content"]

    # Assistant turn echoed back unchanged (thinking signature included), then tool results.
    assistant, results = second["messages"][1], second["messages"][2]
    assert assistant["role"] == "assistant" and assistant["content"][0] == THINKING
    assert [r["tool_use_id"] for r in results["content"]] == ["toolu_1", "toolu_2"]
    assert all(r["type"] == "tool_result" and r["is_error"] is False for r in results["content"])
    assert len(third["messages"]) == 5  # history is append-only

    for _, search in api.search_requests():
        assert search["tools"] == [{"type": "web_search_20260209", "name": "web_search", "max_uses": 4}]


def test_paused_web_search_is_resumed_with_server_blocks_echoed():
    api = full_investigation_api(reputation_pauses=True)
    report = investigate(SCAM, llm=llm_for(api))
    reputation = [b for _, b in api.search_requests() if "scam or fraud reports" in b["messages"][0]["content"]]
    assert len(reputation) == 2
    resumed = reputation[1]["messages"]
    assert [m["role"] for m in resumed] == ["user", "assistant"]
    assert [c["type"] for c in resumed[1]["content"]] == ["server_tool_use", "web_search_tool_result"]
    assert any(e.signal == "IDENTIFIER_REPORTED" for e in report.run.ledger.items)


def test_rejected_api_key_falls_back_to_the_playbook():
    api = FakeMessagesAPI([], {}, errors=[(401, "authentication_error", "invalid x-api-key")])
    report = investigate(SCAM, llm=llm_for(api))
    assert report.run.mode == "playbook" and "API key was rejected" in report.run.stop_reason
    assert len(api.requests) == 1  # web tools don't keep hitting a dead key
    assert report.decision.verdict == "Likely scam"


def test_account_without_fallback_beta_retries_without_it():
    api = full_investigation_api()
    api.errors = [(400, "invalid_request_error", "fallbacks: unknown parameter")]
    report = investigate(SCAM, llm=llm_for(api))
    retried = api.requests[1][1]
    assert "fallbacks" not in retried
    assert report.run.mode == "agent" and report.decision.verdict == "Likely scam"
