"""Tool 5 - detect_red_flags: fixed pattern rules over the offer text.

Each rule has a severity:
  high   - a direct fraud indicator (asks the candidate for money / payment
           credentials). Genuine employers never do this.
  medium - a strong risk factor (WhatsApp-only, no interview, pay-per-task).
  low    - weak on its own (urgency, asking for ID documents early).

Every hit is cited with the exact quote from the message. Negated mentions
("we never charge a registration fee") are skipped - real companies often
include exactly that disclaimer.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from agent_core import Citation, Evidence, ToolOutput
from agent_core.tools import Tool

from ..domains import SHORTENER_OR_CHAT, registrable_domain
from ..signals import MESSAGE
from .extract_entities import UPI_RE, URL_RE

_CURRENCY = r"(?:₹|rs\.?|inr|rupees)"
# "don't worry, the fee is refundable" is a scam line, not a negation.
_NEGATION_RE = re.compile(
    r"\b(never|not|no|don't|do not|doesn't|does not|won't|without|free of|zero)\b(?!\s+worry)[^.!?\n]*$", re.I
)


@dataclass(frozen=True)
class RedFlagRule:
    signal: str
    severity: str
    summary: str
    patterns: tuple[str, ...]
    check_negation: bool = True


RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule(
        "PAYMENT_REQUEST", "high",
        "Asks the candidate to pay money (fee, deposit or charge) to get the job",
        (
            r"\b(?:registration|training|security|processing|onboarding|joining|verification|documentation|document|"
            r"kit|application|interview|form|laptop|uniform|refundable)\s+(?:fee|fees|charge|charges|deposit|amount)\b",
            rf"\b(?:fee|fees|deposit|charges?)\s+(?:of\s+)?{_CURRENCY}\s?\d",
            rf"\b(?:pay|deposit|transfer|send)\b[^.\n]{{0,40}}{_CURRENCY}\s?\d",
            r"\brefundable\b[^.\n]{0,30}\b(?:amount|deposit|fee)\b",
        ),
    ),
    RedFlagRule(
        "UPI_OR_CRYPTO_PAYMENT", "high",
        "Directs payment to a UPI app, wallet or cryptocurrency",
        (
            r"\b(?:pay|send|transfer|deposit|fee|recharge)\b[^.\n]{0,60}\b(?:upi|gpay|google pay|phonepe|phone pe|paytm|bhim|"
            r"usdt|bitcoin|btc|crypto|binance|trc20|wallet)\b",
            r"\b(?:upi|gpay|google pay|phonepe|paytm|usdt|bitcoin|crypto)\b[^.\n]{0,40}\b(?:pay|send|transfer|deposit)\b",
        ),
    ),
    RedFlagRule(
        "UPI_ID_IN_OFFER", "high",
        "Contains a UPI payment ID - a recruiter has no reason to share a payment handle",
        (UPI_RE.pattern,),
        check_negation=False,
    ),
    RedFlagRule(
        "SENSITIVE_CREDENTIALS_REQUEST", "high",
        "Asks for OTPs, PINs, card or banking passwords",
        (r"\b(?:share|send|provide|tell|enter)\b[^.\n]{0,40}\b(?:otp|cvv|upi pin|atm pin|pin|password|card (?:number|details))\b",),
    ),
    RedFlagRule(
        "TASK_BASED_EARNING", "medium",
        "Pay-per-task work such as liking videos or rating products (a common task-scam pattern)",
        (
            r"\b(?:like|subscribe to)\b[^.\n]{0,30}\b(?:videos?|youtube|channels?|posts?)\b",
            r"\b(?:rate|review)\b[^.\n]{0,20}\b(?:hotels?|restaurants?|products?|apps?)\b[^.\n]{0,40}\b(?:earn|per|commission|paid)\b",
            r"\b(?:prepaid|merchant|recharge)\s+tasks?\b",
            r"\btask[- ]based\b",
            r"\bper\s+(?:like|task|review|rating)\b",
        ),
    ),
    RedFlagRule(
        "UNREALISTIC_PAY", "medium",
        "Promises high daily or hourly earnings for simple work",
        (
            rf"\b(?:earn|income|earning|salary|pay|payout)\b[^.\n]{{0,25}}{_CURRENCY}\s?[\d,]+[^.\n]{{0,25}}(?:per|/|a)\s*(?:day|hour|hr)\b",
            r"\b(?:daily|per day)\s+(?:income|earning|earnings|payout|payment)\b",
        ),
    ),
    RedFlagRule(
        "CHAT_APP_ONLY", "medium",
        "Insists on WhatsApp / Telegram as the only contact channel",
        (
            r"\b(?:only|just|strictly)\b[^.\n]{0,25}\b(?:whatsapp|telegram)\b",
            r"\b(?:whatsapp|telegram)\b[^.\n]{0,15}\bonly\b",
            r"\bno\s+(?:calls?|emails?)\b",
        ),
        check_negation=False,
    ),
    RedFlagRule(
        "NO_INTERVIEW", "medium",
        "Offers the job without any interview or assessment",
        (
            r"\b(?:no|without)\s+(?:an?\s+|any\s+)?interviews?\b",
            r"\bdirect(?:ly)?\s+(?:selected|joining|selection|hiring|appointment)\b",
        ),
        check_negation=False,
    ),
    RedFlagRule(
        "CHAT_APP_CONTACT", "low",
        "Moves the conversation to WhatsApp / Telegram",
        (r"\b(?:whatsapp|telegram)\b", r"\b(?:wa\.me|t\.me)/"),
        check_negation=False,
    ),
    RedFlagRule(
        "URGENCY_PRESSURE", "low",
        "Pressures the candidate to act fast",
        (
            r"\b(?:urgent(?:ly)?|immediately|asap)\b",
            r"\bwithin\s+\d+\s+(?:hours?|hrs?|minutes?|mins?)\b",
            r"\b(?:today only|limited (?:seats|slots|vacancies)|last date (?:is )?today|offer (?:expires|valid only))\b",
        ),
        check_negation=False,
    ),
    RedFlagRule(
        "EARLY_ID_DOCUMENTS", "low",
        "Asks for ID or bank documents up front (normal only after a formal offer)",
        (r"\b(?:aadhaa?r|pan card|bank (?:account|details)|passbook|cancelled cheque)\b",),
    ),
)


def _negated(text: str, start: int) -> bool:
    """True if a negation word appears earlier in the same sentence, close to the match."""
    window = text[max(0, start - 45):start]
    return bool(_NEGATION_RE.search(window))


def _snippet(text: str, start: int, end: int, pad: int = 50) -> str:
    """The match plus some context, widened to whole words."""
    s, e = max(0, start - pad), min(len(text), end + pad)
    while s > 0 and not text[s - 1].isspace():
        s -= 1
    while e < len(text) and not text[e].isspace():
        e += 1
    return ("…" if s > 0 else "") + " ".join(text[s:e].split()) + ("…" if e < len(text) else "")


def scan(text: str) -> list[dict]:
    """Return one hit per rule: {signal, severity, summary, quotes}."""
    hits = []
    for rule in RULES:
        quotes, used = [], []
        for pattern in rule.patterns:
            for m in re.finditer(pattern, text, re.I):
                if rule.check_negation and _negated(text, m.start()):
                    continue
                if any(abs(m.start() - u) < 80 for u in used):
                    break  # same passage already quoted by another pattern
                used.append(m.start())
                quotes.append(_snippet(text, m.start(), m.end()))
                break  # one quote per pattern is enough
        if quotes:
            hits.append({"signal": rule.signal, "severity": rule.severity, "summary": rule.summary, "quotes": quotes[:2]})

    shorteners = [u for u in URL_RE.findall(text) if registrable_domain(u) in SHORTENER_OR_CHAT - {"wa.me", "t.me", "whatsapp.com", "telegram.me", "telegram.org"}]
    if shorteners:
        hits.append({
            "signal": "LINK_SHORTENER", "severity": "low",
            "summary": "Uses a link shortener that hides the real destination", "quotes": shorteners[:2],
        })
    return hits


def _run(args: dict, ctx) -> ToolOutput:
    hits = scan(ctx.case.message or "")
    evidence = [
        Evidence(
            signal=h["signal"],
            group=MESSAGE,
            polarity="risk",
            severity=h["severity"],
            summary=h["summary"],
            subject="message",
            citations=[Citation("quote", q) for q in h["quotes"]],
        )
        for h in hits
    ]
    return ToolOutput(data={"flags": hits, "rules_checked": len(RULES) + 1}, evidence=evidence)


TOOL = Tool(
    name="detect_red_flags",
    description=(
        "Scan the submitted offer text for scam red flags: registration/training fees or deposits, "
        "UPI/wallet/crypto payment requests, UPI IDs, requests for OTPs or PINs, pay-per-task work, "
        "unrealistic daily pay, WhatsApp/Telegram-only contact, no interview, urgency pressure. "
        "Reads the offer directly; takes no input. Each flag is cited with a quote."
    ),
    input_schema={"type": "object", "properties": {}, "required": []},
    fn=_run,
    groups=(MESSAGE,),
    timeout_s=10,
)
