"""The investigation report: verdict + the evidence behind it, with sources."""
from __future__ import annotations

from dataclasses import dataclass

from agent_core import Decision, Evidence, RunResult

from .case import JobOfferCase
from .signals import ALL_GROUPS, GROUP_LABELS

DISCLAIMER = (
    "PRAMAAN is a decision aid, not legal advice. The verdict comes from fixed rules over the "
    "evidence listed here. Genuine employers never ask candidates to pay for a job."
)


@dataclass
class Report:
    case: JobOfferCase
    decision: Decision
    run: RunResult
    entities: dict
    budget: int

    def group_status(self, group: str) -> str:
        if group in self.run.ledger.checked_groups:
            return "checked"
        if group in self.run.ledger.failed_groups:
            return "check failed"
        if group in self.run.ledger.skipped_groups:
            return "skipped (needs Claude API)"
        return "not checked"

    def evidence_by_group(self) -> dict[str, list[Evidence]]:
        return self.run.ledger.by_group()

    def to_dict(self) -> dict:
        return {
            "verdict": self.decision.verdict,
            "rule": self.decision.rule_id,
            "explanation": self.decision.explanation,
            "facts": self.decision.facts,
            "mode": self.run.mode,
            "stop_reason": self.run.stop_reason,
            "tool_calls": {"agent": self.run.agent_calls, "budget": self.budget, "extra": self.run.extra_calls},
            "coverage": {g: self.group_status(g) for g in ALL_GROUPS},
            "evidence": [e.to_dict() for e in self.run.ledger.items],
            "entities": self.entities,
            "agent_notes": self.run.notes,
            "trace": [{"t": e.elapsed_s, "kind": e.kind, "message": e.message} for e in self.run.events],
        }

    def to_markdown(self) -> str:
        d, run = self.decision, self.run
        out = [
            f"# PRAMAAN verdict: {d.verdict}",
            "",
            f"**Rule fired:** `{d.rule_id}` - {d.explanation}",
            "",
            f"Mode: {run.mode} · Agent tool calls: {run.agent_calls}/{self.budget} · "
            f"Extra (playbook/safety-net) calls: {run.extra_calls} · Stop reason: {run.stop_reason}",
            "",
            "## Evidence by independent source",
        ]
        by_group = self.evidence_by_group()
        for group in ALL_GROUPS:
            out += ["", f"### {GROUP_LABELS[group]} ({self.group_status(group)})"]
            items = by_group.get(group, [])
            if not items:
                out.append("- No evidence recorded.")
            for e in items:
                out.append(f"- **{e.id}** `{e.signal}` [{e.polarity}, {e.severity}] {e.summary}")
                for c in e.citations:
                    if c.kind == "url":
                        out.append(f"  - Source: <{c.ref}>")
                    elif c.kind == "quote":
                        out.append(f'  - Quote: "{c.ref}"')
                    else:
                        out.append(f"  - Lookup: {c.ref}")
        if run.notes:
            out += ["", "## Agent notes (not used for the verdict)", "", run.notes]
        out += ["", "---", DISCLAIMER]
        return "\n".join(out)
