"""Evidence: the only thing the verdict is allowed to look at.

Tools turn what they observe into Evidence items. The LLM never writes
evidence itself - it can only call tools - so the rules that compute the
verdict see facts produced by deterministic code, each with a citation.

Two ideas matter here:

* group  - the *independent source* a fact came from. Two facts in the same
           group (e.g. "asks for a fee" and "asks via UPI", both read from the
           message) are correlated, so they count once. Facts in different
           groups (the message vs. a domain registry vs. public complaints)
           corroborate each other.
* severity - low / medium / high. Rules usually ignore "low": it is shown to
           the user but is too weak to move a verdict on its own.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal

Polarity = Literal["risk", "trust", "info"]
Severity = Literal["low", "medium", "high"]

SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}


@dataclass(frozen=True)
class Citation:
    """Where a piece of evidence can be checked by a human."""

    kind: Literal["url", "quote", "lookup"]  # web page / excerpt of the input / registry or network lookup
    ref: str  # the URL, the quoted text, or a description of the lookup
    title: str = ""


@dataclass
class Evidence:
    signal: str  # stable machine-readable code, e.g. "PAYMENT_REQUEST"
    group: str  # independence group (see module docstring)
    polarity: Polarity
    severity: Severity
    summary: str  # one human-readable sentence
    subject: str = ""  # what the fact is about: a domain, an email, "message" ...
    citations: list[Citation] = field(default_factory=list)
    tool: str = ""  # set by the runner
    id: str = ""  # set by the ledger: E1, E2, ...

    def to_dict(self) -> dict:
        return asdict(self)

    def brief(self) -> dict:
        """Compact form that is sent back to the LLM."""
        return {
            "id": self.id,
            "signal": self.signal,
            "polarity": self.polarity,
            "severity": self.severity,
            "summary": self.summary,
        }


class EvidenceLedger:
    """Append-only record of everything the tools found, plus which checks ran."""

    def __init__(self) -> None:
        self.items: list[Evidence] = []
        self.tool_status: dict[str, str] = {}  # tool name -> "ok" | "error" | "skipped"
        self.checked_groups: set[str] = set()  # groups at least one tool covered successfully
        self.failed_groups: set[str] = set()  # groups whose tool ran and failed
        self.skipped_groups: set[str] = set()  # groups whose tool could not run in this mode

    def add(self, ev: Evidence, tool: str = "") -> Evidence | None:
        """Record a fact. Returns None if the same fact is already recorded.

        The same signal about the same subject is one fact, however many
        times a tool is called - repeating a check must not add weight.
        """
        key = (ev.signal, ev.subject.lower())
        if any((e.signal, e.subject.lower()) == key for e in self.items):
            return None
        ev.tool = ev.tool or tool
        ev.id = f"E{len(self.items) + 1}"
        self.items.append(ev)
        return ev

    def record_tool(self, name: str, ok: bool, groups: tuple[str, ...] | list[str], skipped: bool = False) -> None:
        # A later success overrides an earlier failure, never the other way round.
        if ok:
            self.tool_status[name] = "ok"
            self.checked_groups.update(groups)
        elif skipped:
            self.tool_status.setdefault(name, "skipped")
            self.skipped_groups.update(groups)
        else:
            self.tool_status.setdefault(name, "error")
            self.failed_groups.update(groups)

    def groups_with(self, polarity: Polarity, min_severity: Severity = "medium") -> set[str]:
        """Independent groups holding at least one fact of this polarity and strength."""
        floor = SEVERITY_RANK[min_severity]
        return {
            e.group
            for e in self.items
            if e.polarity == polarity and SEVERITY_RANK[e.severity] >= floor
        }

    def by_group(self) -> dict[str, list[Evidence]]:
        out: dict[str, list[Evidence]] = {}
        for e in self.items:
            out.setdefault(e.group, []).append(e)
        return out
