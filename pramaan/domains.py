"""Pure, offline helpers for reasoning about domains. No network here.

* registrable_domain  "careers.infosys.com" -> "infosys.com"  (uses the Public Suffix List)
* is_free_email       gmail.com, outlook.com, rediffmail.com ...
* official_company    "infosys.com" -> "Infosys"  (small curated list of often-impersonated employers)
* lookalike_company   "amaz0n-careers.in" -> "Amazon"  (brand in the name but not an official domain)
"""
from __future__ import annotations

import re
from urllib.parse import urlsplit

import tldextract

# Bundled Public Suffix List snapshot only - never fetch it over the network.
_extract = tldextract.TLDExtract(suffix_list_urls=())

FREE_EMAIL_PROVIDERS = frozenset({
    "gmail.com", "googlemail.com", "yahoo.com", "yahoo.co.in", "yahoo.in", "ymail.com",
    "rocketmail.com", "outlook.com", "outlook.in", "hotmail.com", "live.com", "msn.com",
    "rediffmail.com", "rediff.com", "aol.com", "icloud.com", "me.com", "mac.com",
    "protonmail.com", "proton.me", "pm.me", "zohomail.com", "zohomail.in", "gmx.com",
    "gmx.net", "mail.com", "yandex.com", "yandex.ru", "tutanota.com", "tuta.io",
    "fastmail.com", "mail.ru", "qq.com", "163.com",
})

# Link shorteners and chat links: never the employer's own domain.
SHORTENER_OR_CHAT = frozenset({
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "cutt.ly", "rb.gy", "is.gd", "shorturl.at",
    "tiny.cc", "wa.me", "whatsapp.com", "t.me", "telegram.me", "telegram.org",
})

# Job boards, social networks and form hosts: links to these say nothing about who the employer is.
GENERIC_PLATFORMS = frozenset({
    "linkedin.com", "naukri.com", "indeed.com", "foundit.in", "glassdoor.com", "glassdoor.co.in",
    "internshala.com", "youtube.com", "instagram.com", "facebook.com", "twitter.com", "x.com",
})
GENERIC_HOSTS = frozenset({
    "docs.google.com", "forms.gle", "drive.google.com", "sites.google.com", "forms.office.com", "meet.google.com",
})

# Applicant-tracking systems that host official job listings for many companies.
ATS_DOMAINS = frozenset({
    "greenhouse.io", "lever.co", "myworkdayjobs.com", "workday.com", "smartrecruiters.com",
    "ashbyhq.com", "darwinbox.in", "darwinbox.com", "zohorecruit.com", "zohorecruit.in",
    "keka.com", "freshteam.com", "icims.com", "successfactors.com", "successfactors.eu",
    "taleo.net", "oraclecloud.com", "jobvite.com", "recruitee.com", "workable.com", "bamboohr.com",
})

# Company -> (official registrable domains, brand keys to spot in lookalike names).
# Deliberately small: employers most often impersonated in Indian job scams.
KNOWN_COMPANIES: dict[str, tuple[frozenset[str], tuple[str, ...]]] = {
    "Amazon": (frozenset({"amazon.com", "amazon.in", "amazon.jobs", "aboutamazon.com", "aboutamazon.in", "amazonaws.com"}), ("amazon",)),
    "Flipkart": (frozenset({"flipkart.com", "flipkartcareers.com"}), ("flipkart",)),
    "TCS": (frozenset({"tcs.com"}), ("tcs",)),
    "Infosys": (frozenset({"infosys.com", "infosysbpm.com"}), ("infosys",)),
    "Wipro": (frozenset({"wipro.com"}), ("wipro",)),
    "HCLTech": (frozenset({"hcltech.com", "hcl.com"}), ("hcltech", "hcl")),
    "Tech Mahindra": (frozenset({"techmahindra.com"}), ("techmahindra",)),
    "Accenture": (frozenset({"accenture.com"}), ("accenture",)),
    "Deloitte": (frozenset({"deloitte.com"}), ("deloitte",)),
    "Cognizant": (frozenset({"cognizant.com"}), ("cognizant",)),
    "Capgemini": (frozenset({"capgemini.com"}), ("capgemini",)),
    "IBM": (frozenset({"ibm.com"}), ("ibm",)),
    "Google": (frozenset({"google.com", "google.co.in", "googleapis.com"}), ("google",)),
    "Microsoft": (frozenset({"microsoft.com", "microsoftonline.com"}), ("microsoft",)),
    "Zoho": (frozenset({"zoho.com", "zohocorp.com", "zoho.in"}), ("zoho",)),
    "Reliance": (frozenset({"ril.com", "jio.com", "relianceretail.com"}), ("reliance", "jio")),
    "Swiggy": (frozenset({"swiggy.in", "swiggy.com"}), ("swiggy",)),
    "Zomato": (frozenset({"zomato.com"}), ("zomato",)),
    "Paytm": (frozenset({"paytm.com"}), ("paytm",)),
    "Meesho": (frozenset({"meesho.com"}), ("meesho",)),
    "Airtel": (frozenset({"airtel.in", "airtel.com"}), ("airtel",)),
    "HDFC Bank": (frozenset({"hdfcbank.com"}), ("hdfc",)),
    "ICICI Bank": (frozenset({"icicibank.com"}), ("icici",)),
    "Myntra": (frozenset({"myntra.com"}), ("myntra",)),
    "Genpact": (frozenset({"genpact.com"}), ("genpact",)),
    "KPMG": (frozenset({"kpmg.com"}), ("kpmg",)),
    "PwC": (frozenset({"pwc.com", "pwc.in"}), ("pwc",)),
}

# Characters scammers swap in to imitate a brand: amaz0n, paytrn, wipr0 ...
_HOMOGLYPHS = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "$": "s", "@": "a"})


def host_of(value: str) -> str:
    """Hostname from a URL, bare domain or email address ('' if none)."""
    value = value.strip().lower()
    if "@" in value and "/" not in value:
        value = value.rsplit("@", 1)[1]
    if "://" not in value:
        value = "http://" + value
    host = urlsplit(value).hostname or ""
    return host[4:] if host.startswith("www.") else host


def registrable_domain(value: str) -> str | None:
    """The domain someone actually registers: 'careers.infosys.com' -> 'infosys.com'."""
    ext = _extract(host_of(value))
    if not ext.domain or not ext.suffix:
        return None
    return f"{ext.domain}.{ext.suffix}"


def is_generic_platform(value: str) -> bool:
    """Shorteners, chat links, job boards, social media, form hosts and applicant-tracking
    systems - links to these never tell us who the employer is."""
    reg = registrable_domain(value)
    return host_of(value) in GENERIC_HOSTS or reg in GENERIC_PLATFORMS or reg in SHORTENER_OR_CHAT or reg in ATS_DOMAINS


def is_free_email(domain: str) -> bool:
    return (registrable_domain(domain) or domain.lower()) in FREE_EMAIL_PROVIDERS


def official_company(domain: str) -> str | None:
    """Which known company officially owns this domain, if any."""
    reg = registrable_domain(domain)
    for company, (official, _keys) in KNOWN_COMPANIES.items():
        if reg in official:
            return company
    return None


def known_company(name: str) -> str | None:
    """Match a free-text company name to a known company ('infosys ltd' -> 'Infosys')."""
    norm = re.sub(r"[^a-z0-9]", "", name.lower())
    if not norm:
        return None
    for company, (_official, keys) in KNOWN_COMPANIES.items():
        canonical = re.sub(r"[^a-z0-9]", "", company.lower())
        if norm.startswith(canonical) or any(norm == k or (len(k) >= 5 and norm.startswith(k)) for k in keys):
            return company
    return None


def lookalike_company(domain: str) -> str | None:
    """Known brand imitated by a domain that is NOT one of its official domains.

    Checks every label (subdomains too: 'amazon.jobs-hiring.com'), after
    undoing homoglyph swaps, for: the brand as a whole token ('tcs-careers'),
    the brand inside a longer name ('amazonhrjobs'), or a one-letter typo of
    the brand ('infosis').
    """
    if official_company(domain):
        return None
    ext = _extract(host_of(domain))
    labels = [*(ext.subdomain.split(".") if ext.subdomain else []), ext.domain]
    for label in labels:
        norm = label.lower().translate(_HOMOGLYPHS).replace("rn", "m").replace("vv", "w")
        tokens = [t for t in re.split(r"[-_]+", norm) if t]
        joined = "".join(tokens)
        for company, (_official, keys) in KNOWN_COMPANIES.items():
            for key in keys:
                if key in tokens:
                    return company
                if len(key) >= 5 and key in joined:
                    return company
                if len(key) >= 5 and any(len(t) >= 4 and _levenshtein(t, key) == 1 for t in tokens):
                    return company
    return None


def _levenshtein(a: str, b: str) -> int:
    if abs(len(a) - len(b)) > 1:
        return 2  # we only care whether the distance is exactly 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]
