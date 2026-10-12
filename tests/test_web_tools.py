"""Web-search tools: claims from the model are verified before they become evidence."""
import pytest

from agent_core import ToolError, ToolSkipped
from pramaan.signals import IDENTITY, LISTING, REPORTS
from pramaan.tools import verify_vacancy, web_reputation
from pramaan.websearch import SearchOutcome, normalize_url, parse_json_object, web_search_json

from .fakes import ScriptedLLM, response, text


def outcome(data, urls, citations=()):
    return SearchOutcome(
        data=data,
        result_urls={normalize_url(u): t for u, t in urls.items()},
        citations=[(normalize_url(u), c) for u, c in citations],
    )


def test_parse_json_object_tolerates_fences_and_prose():
    assert parse_json_object('Here:\n```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_object('Result {"b": 2} end') == {"b": 2}
    with pytest.raises(ToolError):
        parse_json_object("no json here")


def test_web_search_json_resumes_pause_turn_and_harvests_sources():
    dump = {"content": [
        {"type": "server_tool_use", "name": "web_search", "input": {"query": "acme careers"}},
        {"type": "web_search_tool_result", "content": [
            {"type": "web_search_result", "url": "https://www.acme.com/careers/", "title": "Careers"},
        ]},
    ]}
    llm = ScriptedLLM([
        response(stop="pause_turn", dump=dump),
        response(text('{"ok": true}'), stop="end_turn"),
    ])
    out = web_search_json(llm, "find", tool_type="web_search_20260209", max_uses=2)
    assert out.data == {"ok": True}
    assert out.verified("https://acme.com/careers") and out.queries == ["acme careers"]
    assert len(llm.requests) == 2  # resumed once


def test_web_search_is_skipped_without_an_available_llm():
    class Offline:
        available = False

    with pytest.raises(ToolSkipped):
        web_search_json(Offline(), "x", tool_type="t", max_uses=1)


def test_vacancy_listing_with_invented_url_is_dropped():
    o = outcome(
        {"official_domain": "acme.com", "role_listed": "yes", "listing_url": "https://acme.com/jobs/made-up"},
        {"https://acme.com/careers": "Careers"},
    )
    evidence, data, _ = verify_vacancy.assess(o, company="Acme", role="Analyst", offer_domains=[], known_official=frozenset())
    assert not any(e.signal == "ROLE_LISTED_OFFICIALLY" for e in evidence)
    assert data["dropped_unverified_urls"] == ["https://acme.com/jobs/made-up"]


def test_vacancy_verified_listing_and_domain_mismatch():
    o = outcome(
        {"official_domain": "acme.com", "role_listed": "yes", "listing_url": "https://boards.greenhouse.io/acme/1",
         "careers_url": "https://acme.com/careers"},
        {"https://acme.com/careers": "Careers", "https://boards.greenhouse.io/acme/1": "Analyst"},
    )
    evidence, _, groups = verify_vacancy.assess(
        o, company="Acme", role="Analyst", offer_domains=["acme-hiring.in"], known_official=frozenset()
    )
    s = {e.signal: (e.group, e.polarity) for e in evidence}
    assert s["ROLE_LISTED_OFFICIALLY"] == (LISTING, "trust")  # ATS page counts as official
    assert s["OFFICIAL_DOMAIN_MISMATCH"] == (IDENTITY, "risk")
    assert groups == [LISTING, IDENTITY]


def test_reputation_grades_reports_by_how_well_they_are_verified():
    url_a, url_b, fake = "https://complaints.example/a", "https://forum.example/b", "https://invented.example/x"
    o = outcome(
        {"reports": [
            {"identifier": "+919876543210", "about": "identifier", "url": url_a, "claim": "Fake HR number"},
            {"identifier": "fees@ybl", "about": "identifier", "url": url_b, "claim": "UPI scam"},
            {"identifier": "x@gmail.com", "about": "identifier", "url": fake, "claim": "made up"},
        ]},
        {url_a: "Scam alert 98765 43210", url_b: "Job scam thread"},
    )
    evidence, data = web_reputation.assess(o, identifiers=["+919876543210", "fees@ybl", "x@gmail.com"], company="Acme")
    s = {e.subject: e.severity for e in evidence if e.signal == "IDENTIFIER_REPORTED"}
    assert s == {"+919876543210": "high", "fees@ybl": "medium"}  # number visible in the title; UPI ID not
    assert data["dropped_unverified_urls"] == [fake]
    assert all(e.group == REPORTS for e in evidence)


def test_big_brand_is_never_itself_reported_as_fake():
    url = "https://news.example/amazon-job-scams"
    o = outcome({"reports": [{"identifier": None, "about": "company", "url": url, "claim": "Amazon job scams rise"}]}, {url: "News"})
    evidence, _ = web_reputation.assess(o, identifiers=["+919876543210"], company="Amazon")
    assert [(e.signal, e.polarity) for e in evidence] == [("BRAND_IMPERSONATION_WARNING", "info")]


def test_nothing_found_is_info_not_trust():
    evidence, _ = web_reputation.assess(outcome({"reports": []}, {}), identifiers=["a@b.com"], company="")
    assert [(e.signal, e.polarity) for e in evidence] == [("NO_REPORTS_FOUND", "info")]
