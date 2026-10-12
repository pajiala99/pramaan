"""The rulebook: evidence -> one of four verdicts. No LLM involved.

Facts first, then an ordered rule list where the first match wins:

  R1  Likely scam          at least one HIGH risk (a direct fraud indicator)
                           AND medium+ risk in >= 2 independent groups
  R2  Suspicious           any medium+ risk that R1 did not cover
                           (incl. a direct fraud indicator seen in only one group)
  R3  Likely genuine       no medium+ risk, medium+ trust in >= 2 independent groups,
                           and the message itself was scanned
  R4  Not enough evidence  everything else

R1 is the safety rule from the brief: PRAMAAN never calls an offer a scam
on a single piece of evidence. "Asks for a fee" plus "asks via UPI" is one
observation (both read from the message), so it needs corroboration from a
different source - the sender identity, the domain registry, the official
careers page or public reports.
"""
from __future__ import annotations

from agent_core import EvidenceLedger, Rule

from .signals import ALL_GROUPS, GROUP_LABELS, MESSAGE

LIKELY_GENUINE = "Likely genuine"
SUSPICIOUS = "Suspicious"
LIKELY_SCAM = "Likely scam"
NOT_ENOUGH = "Not enough evidence"
VERDICTS = (LIKELY_SCAM, SUSPICIOUS, LIKELY_GENUINE, NOT_ENOUGH)

MIN_GROUPS_FOR_SCAM = 2
MIN_GROUPS_FOR_GENUINE = 2


def compute_facts(ledger: EvidenceLedger) -> dict:
    return {
        "risk_groups": sorted(ledger.groups_with("risk", "medium")),
        "high_risk_groups": sorted(ledger.groups_with("risk", "high")),
        "trust_groups": sorted(ledger.groups_with("trust", "medium")),
        "checked_groups": sorted(ledger.checked_groups),
        "unchecked_groups": [g for g in ALL_GROUPS if g not in ledger.checked_groups],
    }


def _names(groups: list[str]) -> str:
    return "; ".join(GROUP_LABELS[g].split(" - ")[0] for g in groups) or "none"


def _explain_scam(f: dict) -> str:
    return (
        f"Direct fraud indicator(s) in: {_names(f['high_risk_groups'])}. "
        f"Corroborated by risk evidence from {len(f['risk_groups'])} independent sources: {_names(f['risk_groups'])}."
    )


def _explain_suspicious(f: dict) -> str:
    if f["high_risk_groups"]:
        return (
            f"A direct fraud indicator was found ({_names(f['high_risk_groups'])}), but no independent source "
            f"corroborates it yet. PRAMAAN never calls an offer a scam on a single source - treat it with "
            f"great caution and do not pay anything."
        )
    return (
        f"Risk signals in: {_names(f['risk_groups'])}, but none is a direct fraud indicator "
        f"(such as a request for money). Verify independently before sharing documents."
    )


def _explain_genuine(f: dict) -> str:
    return (
        f"No medium or high risk signals, and positive evidence from {len(f['trust_groups'])} independent "
        f"sources: {_names(f['trust_groups'])}. Still never pay to get a job."
    )


def _explain_not_enough(f: dict) -> str:
    return (
        f"Neither risk nor trust evidence is strong enough to decide. Checked: {_names(f['checked_groups'])}. "
        f"Not checked: {_names(f['unchecked_groups'])}."
    )


RULES: list[Rule] = [
    Rule(
        "R1_CORROBORATED_SCAM", LIKELY_SCAM,
        when=lambda f: bool(f["high_risk_groups"]) and len(f["risk_groups"]) >= MIN_GROUPS_FOR_SCAM,
        explain=_explain_scam,
    ),
    Rule(
        "R2_UNCORROBORATED_RISK", SUSPICIOUS,
        when=lambda f: bool(f["risk_groups"]),
        explain=_explain_suspicious,
    ),
    Rule(
        "R3_CORROBORATED_GENUINE", LIKELY_GENUINE,
        when=lambda f: len(f["trust_groups"]) >= MIN_GROUPS_FOR_GENUINE and MESSAGE in f["checked_groups"],
        explain=_explain_genuine,
    ),
    Rule(
        "R4_NOT_ENOUGH_EVIDENCE", NOT_ENOUGH,
        when=lambda f: True,
        explain=_explain_not_enough,
    ),
]
