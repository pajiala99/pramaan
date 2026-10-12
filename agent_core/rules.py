"""Deterministic verdict engine: an ordered list of rules, first match wins.

This works like a firewall rule list. Each rule is a predicate over a
`facts` dict (computed from the evidence ledger by the domain code) plus the
verdict it produces. The same evidence always gives the same verdict, and
the report can name exactly which rule fired and why.

The LLM is not involved here at all.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Rule:
    id: str
    verdict: str
    when: Callable[[dict], bool]  # predicate over the facts
    explain: Callable[[dict], str]  # human-readable reason, shown in the report


@dataclass
class Decision:
    verdict: str
    rule_id: str
    explanation: str
    facts: dict


def decide(rules: list[Rule], facts: dict) -> Decision:
    for rule in rules:
        if rule.when(facts):
            return Decision(rule.verdict, rule.id, rule.explain(facts), facts)
    raise ValueError("rule list must end with a catch-all rule")
