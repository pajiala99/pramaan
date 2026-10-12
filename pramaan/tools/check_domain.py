"""Tool 2 - check_domain: who is really behind the emails and websites?

Two evidence groups, two sources of truth:

sender_identity (offline reasoning)
    * recruiter uses a free mailbox (gmail.com ...)            risk  medium
    * domain imitates a known brand (amaz0n-careers.in)        risk  HIGH
    * claims to be a known company but uses another domain     risk  medium
    * recruiter email domain differs from the company website  risk  medium
    * domain is a known official domain of the company         trust medium

domain_registry (network: RDAP - the modern WHOIS - and a TLS handshake)
    * domain not registered at all                             risk  medium
    * registered < 90 days ago                                 risk  medium
    * registered < 1 year ago                                  risk  low
    * registered >= 3 years ago                                trust medium
    * no valid HTTPS certificate                               risk  low
  A valid certificate is NOT trust evidence: certificates are free and
  scam sites have them too.
"""
from __future__ import annotations

import json
import socket
import ssl
import urllib.error
import urllib.request
from datetime import datetime, timezone

from agent_core import Citation, Evidence, ToolOutput
from agent_core.tools import Tool

from ..domains import (
    KNOWN_COMPANIES,
    is_free_email,
    is_generic_platform,
    known_company,
    lookalike_company,
    official_company,
    registrable_domain,
)
from ..signals import IDENTITY, REGISTRY
from .extract_entities import extract

NEW_DOMAIN_DAYS = 90
YOUNG_DOMAIN_DAYS = 365
ESTABLISHED_DOMAIN_DAYS = 3 * 365


class NotRegistered(Exception):
    """The registry has no record of this domain."""


# ---------------------------------------------------------------- network lookups

def rdap_registration_date(domain: str, timeout: float = 6.0) -> datetime:
    """Registration date from RDAP (rdap.org redirects to the authoritative registry)."""
    req = urllib.request.Request(
        f"https://rdap.org/domain/{domain}",
        headers={"Accept": "application/rdap+json", "User-Agent": "pramaan-scam-checker/1.0"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise NotRegistered(domain) from e
        raise
    for event in data.get("events", []):
        if event.get("eventAction") == "registration":
            return datetime.fromisoformat(event["eventDate"].replace("Z", "+00:00"))
    raise LookupError("registry published no registration date")


def tls_probe(domain: str, timeout: float = 4.0) -> dict:
    """Try an HTTPS handshake on the domain, then on www.<domain>."""
    last_error = "unknown error"
    for host in (domain, f"www.{domain}"):
        try:
            with socket.create_connection((host, 443), timeout=timeout) as sock:
                with ssl.create_default_context().wrap_socket(sock, server_hostname=host) as tls:
                    cert = tls.getpeercert()
            issuer = dict(x[0] for x in cert.get("issuer", ()))
            return {"valid": True, "host": host, "issuer": issuer.get("organizationName", "?")}
        except ssl.SSLCertVerificationError as e:
            return {"valid": False, "host": host, "error": f"certificate invalid: {e.verify_message}"}
        except socket.gaierror:
            last_error = "domain does not resolve"
        except (OSError, ssl.SSLError) as e:
            last_error = f"no HTTPS server ({type(e).__name__})"
    return {"valid": False, "host": domain, "error": last_error}


# ---------------------------------------------------------------- the tool

def check_domain(args: dict, ctx, *, rdap=None, tls=None, now: datetime | None = None) -> ToolOutput:
    # Network lookups are injectable (and looked up at call time) so tests run offline.
    rdap = rdap or rdap_registration_date
    tls = tls or tls_probe
    case = ctx.case
    found = extract(case)
    now = now or datetime.now(timezone.utc)
    max_lookups = ctx.settings.max_domains_checked

    emails = {e.strip().lower() for e in [*args.get("emails", []), *(e["address"] for e in found["emails"])] if "@" in e}
    domains = [registrable_domain(d) for d in [case.website, *args.get("domains", []), *found["domains"]] if d]
    domains = list(dict.fromkeys(d for d in domains if d and not is_free_email(d) and not is_generic_platform(d)))
    if not emails and not domains:
        # Not a failure - there is simply no input. The groups stay "not checked".
        return ToolOutput(data={"note": "nothing to check: no email address, website or domain in the offer"}, groups_checked=[])

    company = args.get("company") or case.company or found["known_company"] or ""
    claimed = known_company(company) if company else None
    website = registrable_domain(case.website) if case.website else (domains[0] if domains else None)

    evidence: list[Evidence] = []
    notes: list[str] = []

    # --- sender identity: offline reasoning ----------------------------------
    for email in sorted(emails):
        domain = email.rsplit("@", 1)[1]
        if is_free_email(domain):
            claim = f" while claiming to represent {company}" if company else ""
            evidence.append(Evidence(
                "FREE_EMAIL_RECRUITER", IDENTITY, "risk", "medium",
                f"Recruiter writes from a free {domain} mailbox ({email}){claim}; company recruiters use the company domain.",
                subject=email, citations=[Citation("lookup", f"{domain} is a free email provider")],
            ))
            continue
        email_domain = registrable_domain(domain)
        # zohocorp.com vs zoho.com is fine: both are official domains of the same company.
        same_company = official_company(email_domain) is not None and official_company(email_domain) == official_company(website or "")
        if website and email_domain != website and not same_company:
            evidence.append(Evidence(
                "DOMAIN_MISMATCH", IDENTITY, "risk", "medium",
                f"Recruiter email domain {email_domain} differs from the company website {website}.",
                subject=email, citations=[Citation("lookup", f"{email} vs {website}")],
            ))

    for domain in domains:
        official_for = official_company(domain)
        imitates = lookalike_company(domain)
        if official_for and (claimed is None or claimed == official_for):
            evidence.append(Evidence(
                "KNOWN_OFFICIAL_DOMAIN", IDENTITY, "trust", "medium",
                f"{domain} is an official domain of {official_for}.",
                subject=domain, citations=[Citation("lookup", f"PRAMAAN list of official {official_for} domains")],
            ))
        elif imitates:
            official = ", ".join(sorted(KNOWN_COMPANIES[imitates][0]))
            evidence.append(Evidence(
                "LOOKALIKE_DOMAIN", IDENTITY, "risk", "high",
                f"{domain} imitates {imitates} but is not one of its official domains ({official}).",
                subject=domain, citations=[Citation("lookup", f"brand-lookalike check against {official}")],
            ))
        elif claimed:
            official = ", ".join(sorted(KNOWN_COMPANIES[claimed][0]))
            evidence.append(Evidence(
                "OFF_BRAND_DOMAIN", IDENTITY, "risk", "medium",
                f"The offer claims to be from {claimed}, but {domain} is not a {claimed} domain (official: {official}).",
                subject=domain, citations=[Citation("lookup", f"PRAMAAN list of official {claimed} domains")],
            ))

    # --- domain registry: network lookups ------------------------------------
    registry_answered = False
    domain_reports = []
    for domain in domains[:max_lookups]:
        report = {"domain": domain, "official_for": official_company(domain), "lookalike_of": lookalike_company(domain)}
        rdap_ref = Citation("lookup", f"RDAP registry record: https://rdap.org/domain/{domain}")
        try:
            created = rdap(domain)
            registry_answered = True
            age = (now - created).days
            report.update(registered=created.date().isoformat(), age_days=age)
            if age < NEW_DOMAIN_DAYS:
                evidence.append(Evidence("NEW_DOMAIN", REGISTRY, "risk", "medium",
                                         f"{domain} was registered only {age} days ago ({created.date()}).",
                                         subject=domain, citations=[rdap_ref]))
            elif age < YOUNG_DOMAIN_DAYS:
                evidence.append(Evidence("YOUNG_DOMAIN", REGISTRY, "risk", "low",
                                         f"{domain} is less than a year old (registered {created.date()}).",
                                         subject=domain, citations=[rdap_ref]))
            elif age >= ESTABLISHED_DOMAIN_DAYS:
                evidence.append(Evidence("ESTABLISHED_DOMAIN", REGISTRY, "trust", "medium",
                                         f"{domain} has been registered since {created.date()} ({age // 365} years).",
                                         subject=domain, citations=[rdap_ref]))
        except NotRegistered:
            registry_answered = True
            report["registered"] = None
            evidence.append(Evidence("DOMAIN_NOT_REGISTERED", REGISTRY, "risk", "medium",
                                     f"{domain} is not registered - the website named in the offer does not exist.",
                                     subject=domain, citations=[rdap_ref]))
            domain_reports.append(report)
            continue
        except Exception as e:  # registry down, unsupported TLD, rate limit ...
            notes.append(f"WHOIS/RDAP lookup for {domain} failed: {type(e).__name__}")

        try:
            report["tls"] = tls(domain)
            if not report["tls"]["valid"]:
                evidence.append(Evidence("NO_VALID_HTTPS", REGISTRY, "risk", "low",
                                         f"{domain} has no valid HTTPS site ({report['tls']['error']}).",
                                         subject=domain, citations=[Citation("lookup", f"TLS handshake with {domain}:443")]))
        except Exception as e:
            notes.append(f"TLS check for {domain} failed: {type(e).__name__}")
        domain_reports.append(report)

    if len(domains) > max_lookups:
        notes.append(f"Registry lookups limited to the first {max_lookups} domains.")

    groups = [IDENTITY] + ([REGISTRY] if registry_answered else [])
    if not registry_answered and domains:
        notes.append("Domain registry could not be checked; that evidence group is marked unchecked.")
    data = {
        "company": company or None,
        "known_company": claimed,
        "emails_checked": sorted(emails),
        "domains": domain_reports or [{"domain": d} for d in domains],
        "notes": notes,
    }
    return ToolOutput(data=data, evidence=evidence, groups_checked=groups)


TOOL = Tool(
    name="check_domain",
    description=(
        "Check the recruiter's email addresses and the websites/domains in the offer: free-mail "
        "recruiter (gmail etc.), mismatch between email domain and company website, lookalike "
        "domains imitating known brands, domain registration age via RDAP/WHOIS, and HTTPS. "
        "The offer's own emails and website are always included; pass any extra ones you found."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "emails": {"type": "array", "items": {"type": "string"}, "description": "Recruiter email addresses."},
            "domains": {"type": "array", "items": {"type": "string"}, "description": "Websites or domains to check."},
            "company": {"type": "string", "description": "Company the offer claims to be from."},
        },
        "required": [],
    },
    fn=lambda args, ctx: check_domain(args, ctx),
    groups=(IDENTITY, REGISTRY),
    timeout_s=60,
)
