"""Tool 1 - extract_entities: pull identifiers out of the offer with regexes.

Deterministic on purpose: the same text always yields the same entities, and
the output feeds the other tools. It produces no evidence by itself - finding
a phone number is not suspicious; what other tools learn about it might be.

Note the email / UPI distinction: 'hr@gmail.com' is an email (has a TLD),
'payhr@ybl' is a UPI payment ID (no dot after the @).
"""
from __future__ import annotations

import re

from agent_core import ToolOutput
from agent_core.tools import Tool

from ..case import JobOfferCase
from ..domains import (
    KNOWN_COMPANIES,
    host_of,
    is_free_email,
    is_generic_platform,
    known_company,
    registrable_domain,
)

EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b")
# No '.' or '-' + more domain after the handle: 'hr@acme-analytics.in' is not a UPI ID.
UPI_RE = re.compile(r"\b[A-Za-z0-9._-]{2,64}@[A-Za-z]{2,20}\b(?![.\-][A-Za-z0-9])")
INDIAN_PHONE_RE = re.compile(r"(?<![\d+])(?:\+?91[\s-]?|0)?([6-9]\d{4}[\s-]?\d{5})(?!\d)")
INTL_PHONE_RE = re.compile(r"(?<!\d)\+(?!91)\d{1,3}[\s-]?\d[\d\s-]{6,13}\d(?!\d)")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"')\]]+", re.I)
BARE_DOMAIN_RE = re.compile(r"\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,24}\b", re.I)
AMOUNT_RE = re.compile(r"(?:₹|rs\.?|inr)\s?\d[\d,]*(?:\.\d+)?(?:\s?/-)?|\b\d[\d,]*\s?(?:/-|rupees\b|rs\b|inr\b)", re.I)
PAY_WORDS_RE = re.compile(r"\b(pay|paid|payment|fee|fees|deposit|charge|charges|transfer|send|refundable)\b", re.I)
COMPANY_SUFFIX_RE = re.compile(
    r"\b((?:[A-Z][A-Za-z0-9&.]*\s){0,3}[A-Z][A-Za-z0-9&.]*)\s+"
    r"(?:Pvt\.?\s?Ltd\.?|Private Limited|Ltd\.?|Limited|LLP|Inc\.?|Technologies|Solutions|Analytics|Services|Corporation)\b"
)
ROLE_RE = re.compile(
    r"\b(?:position|role|post|opening|vacancy|profile|designation|job title)\s*(?:of|for|as|is|:|-)?\s*(?:an?\s+)?"
    r"([A-Z][A-Za-z/&+\-]*(?:\s+[A-Z][A-Za-z/&+\-]*){0,4})"
)
ROLE_AS_RE = re.compile(r"\b(?:as|for)\s+(?:an?\s+|the\s+)?([A-Z][A-Za-z/&+\-]*(?:\s+[A-Z][A-Za-z/&+\-]*){1,4})")


def _unique(items):
    seen, out = set(), []
    for item in items:
        if item and item.lower() not in seen:
            seen.add(item.lower())
            out.append(item)
    return out


def _sentences_with(text: str, pattern: re.Pattern) -> list[str]:
    # Split only before a capital letter, so "Rs. 499" stays in one sentence.
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z])|\n+", text)
    return [p.strip() for p in parts if pattern.search(p)]


def extract(case: JobOfferCase) -> dict:
    """All identifiers in the offer, merged with the details the user typed in."""
    text = case.message or ""

    emails = _unique([*EMAIL_RE.findall(text), case.recruiter_email.strip()])
    text_without_emails = EMAIL_RE.sub(" ", text)
    upi_ids = _unique(UPI_RE.findall(text_without_emails))
    # 'amazonhr.jobs@ybl' must not also be read as the domain 'amazonhr.jobs'.
    text_without_emails = UPI_RE.sub(" ", text_without_emails)

    phones = [f"+91{re.sub(r'[^0-9]', '', m)}" for m in INDIAN_PHONE_RE.findall(text)]
    phones += ["+" + re.sub(r"[^0-9]", "", m) for m in INTL_PHONE_RE.findall(text)]
    if case.recruiter_phone.strip():
        digits = re.sub(r"[^0-9]", "", case.recruiter_phone)
        phones.append("+91" + digits[-10:] if len(digits) in (10, 12) and digits[-10] in "6789" else "+" + digits)
    phones = _unique(phones)

    urls = _unique([u.rstrip(".,;:") for u in URL_RE.findall(text_without_emails)] + [case.website.strip()])
    hosts = [host_of(u) for u in urls]
    hosts += [m for m in BARE_DOMAIN_RE.findall(URL_RE.sub(" ", text_without_emails)) if registrable_domain(m)]

    email_info = []
    for e in emails:
        domain = e.rsplit("@", 1)[1].lower()
        email_info.append({"address": e, "domain": domain, "free_provider": is_free_email(domain)})

    # Domains worth checking: website/links/email domains, minus free-mail, shorteners,
    # job boards and form hosts (a LinkedIn link says nothing about the employer).
    domains = _unique(
        [registrable_domain(h) for h in hosts if h and not is_generic_platform(h)]
        + [registrable_domain(e["domain"]) for e in email_info if not e["free_provider"]]
    )
    domains = [d for d in domains if d and not is_free_email(d)]

    companies = [case.company.strip()]
    for company, (_official, keys) in KNOWN_COMPANIES.items():
        if re.search(rf"\b{re.escape(company)}\b", text, re.I) or any(
            len(k) >= 4 and re.search(rf"\b{re.escape(k)}\b", text, re.I) for k in keys
        ):
            companies.append(company)
    companies += COMPANY_SUFFIX_RE.findall(text)
    companies = _unique([c.strip() for c in companies])

    roles = _unique([m.strip() for m in ROLE_RE.findall(text) + ROLE_AS_RE.findall(text)])

    amounts = _unique([m.strip() for m in AMOUNT_RE.findall(text)])
    payment_asks = [s for s in _sentences_with(text, AMOUNT_RE) if PAY_WORDS_RE.search(s)]

    return {
        "company_candidates": companies,
        "known_company": next((k for k in (known_company(c) for c in companies) if k), None),
        "role_candidates": roles[:5],
        "emails": email_info,
        "phones": phones,
        "upi_ids": upi_ids,
        "urls": [u for u in urls if u],
        "domains": domains,
        "amounts": amounts,
        "payment_asks": payment_asks,
    }


def _run(args: dict, ctx) -> ToolOutput:
    return ToolOutput(data=extract(ctx.case))


TOOL = Tool(
    name="extract_entities",
    description=(
        "Extract identifiers from the submitted job offer: company names, role titles, "
        "emails (flagging free providers like gmail), phone numbers, UPI payment IDs, URLs, "
        "domains, money amounts and sentences that ask for payment. Reads the offer directly; "
        "takes no input. Call this first."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    fn=_run,
    groups=(),  # produces data for other tools, not evidence
    timeout_s=10,
)
