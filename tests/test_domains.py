"""Domain helpers and check_domain, with the network lookups faked."""
from datetime import timedelta

import pytest

from pramaan.case import InvestigationContext, JobOfferCase
from pramaan.domains import lookalike_company, official_company, registrable_domain
from pramaan.signals import IDENTITY, REGISTRY
from pramaan.tools.check_domain import NotRegistered, check_domain

from .fakes import NOW, OLD, fake_tls


@pytest.mark.parametrize("domain, brand", [
    ("amaz0n-careers.in", "Amazon"),       # homoglyph + extra word
    ("tcs-careers.com", "TCS"),            # short brand as a whole token
    ("lnfosys.com", "Infosys"),            # one-letter typo
    ("amazon.jobs-hiring.com", "Amazon"),  # brand hidden in a subdomain
    ("wipr0.com", "Wipro"),
    ("infosys.com", None),                 # official, not a lookalike
    ("acme-analytics.in", None),
])
def test_lookalike_detection(domain, brand):
    assert lookalike_company(domain) == brand


def test_registrable_domain_handles_multi_part_suffixes():
    assert registrable_domain("https://careers.acme.co.in/apply") == "acme.co.in"
    assert official_company("career.infosys.com") == "Infosys"


def run(case, rdap):
    ctx = InvestigationContext(case=case, llm=None)
    return check_domain({}, ctx, rdap=rdap, tls=fake_tls, now=NOW)


def signals(out):
    return {e.signal: (e.group, e.polarity, e.severity) for e in out.evidence}


def test_gmail_recruiter_with_lookalike_new_domain():
    case = JobOfferCase(message="Amazon hiring", company="Amazon", website="amazon-hr-jobs.in", recruiter_email="amazon.hr@gmail.com")
    out = run(case, rdap=lambda d: NOW - timedelta(days=20))
    s = signals(out)
    assert s["FREE_EMAIL_RECRUITER"] == (IDENTITY, "risk", "medium")
    assert s["LOOKALIKE_DOMAIN"] == (IDENTITY, "risk", "high")
    assert s["NEW_DOMAIN"] == (REGISTRY, "risk", "medium")
    assert out.groups_checked == [IDENTITY, REGISTRY]


def test_official_established_domain_is_trusted():
    case = JobOfferCase(message="Offer", company="Infosys", website="infosys.com", recruiter_email="careers@infosys.com")
    s = signals(run(case, rdap=lambda d: OLD))
    assert s["KNOWN_OFFICIAL_DOMAIN"][1:] == ("trust", "medium")
    assert s["ESTABLISHED_DOMAIN"] == (REGISTRY, "trust", "medium")
    assert not any(v[1] == "risk" for v in s.values())


def test_same_company_domains_are_not_a_mismatch():
    case = JobOfferCase(message="Offer", company="Zoho", website="zoho.com", recruiter_email="hr@zohocorp.com")
    assert "DOMAIN_MISMATCH" not in signals(run(case, rdap=lambda d: OLD))


def test_unregistered_website_and_email_domain_mismatch():
    def rdap(domain):
        raise NotRegistered(domain)

    case = JobOfferCase(message="Offer", website="acme-careers.in", recruiter_email="hr@acme-jobs.com")
    s = signals(run(case, rdap))
    assert s["DOMAIN_NOT_REGISTERED"] == (REGISTRY, "risk", "medium")
    assert s["DOMAIN_MISMATCH"] == (IDENTITY, "risk", "medium")


def test_registry_outage_marks_group_unchecked_instead_of_failing():
    def rdap(domain):
        raise TimeoutError("registry down")

    case = JobOfferCase(message="Offer", website="acme-careers.in")
    out = run(case, rdap)
    assert out.groups_checked == [IDENTITY]
    assert any("failed" in n for n in out.data["notes"])
