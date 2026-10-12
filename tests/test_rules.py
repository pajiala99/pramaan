"""The verdict rulebook, including the two-independent-sources safety rule."""
from agent_core import Evidence, EvidenceLedger, decide
from pramaan.signals import IDENTITY, LISTING, MESSAGE, REGISTRY, REPORTS
from pramaan.verdict_rules import LIKELY_GENUINE, LIKELY_SCAM, NOT_ENOUGH, RULES, SUSPICIOUS, compute_facts


def verdict(*items, checked=(MESSAGE, IDENTITY, REGISTRY, LISTING, REPORTS)):
    ledger = EvidenceLedger()
    for i, (group, polarity, severity) in enumerate(items):
        ledger.add(Evidence(f"S{i}", group, polarity, severity, "test", subject=str(i)))
    ledger.checked_groups.update(checked)
    return decide(RULES, compute_facts(ledger))


def test_two_direct_indicators_from_the_same_source_are_not_enough_for_scam():
    d = verdict((MESSAGE, "risk", "high"), (MESSAGE, "risk", "high"), (MESSAGE, "risk", "medium"))
    assert d.verdict == SUSPICIOUS and d.rule_id == "R2_UNCORROBORATED_RISK"


def test_direct_indicator_corroborated_by_independent_source_is_scam():
    d = verdict((MESSAGE, "risk", "high"), (IDENTITY, "risk", "medium"))
    assert d.verdict == LIKELY_SCAM and d.rule_id == "R1_CORROBORATED_SCAM"


def test_two_risk_sources_without_a_direct_indicator_is_only_suspicious():
    assert verdict((IDENTITY, "risk", "medium"), (REGISTRY, "risk", "medium")).verdict == SUSPICIOUS


def test_low_severity_risk_alone_does_not_move_the_verdict():
    assert verdict((MESSAGE, "risk", "low"), (LISTING, "risk", "low")).verdict == NOT_ENOUGH


def test_two_trust_sources_and_no_risk_is_genuine():
    assert verdict((IDENTITY, "trust", "medium"), (REGISTRY, "trust", "medium")).verdict == LIKELY_GENUINE


def test_genuine_requires_the_message_to_have_been_scanned():
    d = verdict((IDENTITY, "trust", "medium"), (REGISTRY, "trust", "medium"), checked=(IDENTITY, REGISTRY))
    assert d.verdict == NOT_ENOUGH


def test_any_medium_risk_blocks_genuine():
    d = verdict((IDENTITY, "trust", "medium"), (REGISTRY, "trust", "medium"), (MESSAGE, "risk", "medium"))
    assert d.verdict == SUSPICIOUS


def test_no_evidence_is_not_enough_evidence():
    assert verdict(checked=()).verdict == NOT_ENOUGH


def test_info_items_never_count():
    assert verdict((REPORTS, "info", "low"), (LISTING, "info", "low")).verdict == NOT_ENOUGH
