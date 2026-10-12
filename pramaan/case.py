"""The investigation input and the context every tool receives."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import Settings


@dataclass
class JobOfferCase:
    message: str  # the pasted offer text (untrusted)
    company: str = ""  # optional details the user typed in
    website: str = ""
    recruiter_email: str = ""
    recruiter_phone: str = ""

    def as_prompt(self) -> str:
        """The user message for the agent. The offer is fenced as data, not instructions."""
        return (
            "Investigate this job offer.\n\n"
            f"<offer>\n{self.message.strip()}\n</offer>\n\n"
            "Details supplied by the user (may be empty):\n"
            f"- Company: {self.company or '-'}\n"
            f"- Website: {self.website or '-'}\n"
            f"- Recruiter email: {self.recruiter_email or '-'}\n"
            f"- Recruiter phone: {self.recruiter_phone or '-'}\n"
        )


@dataclass
class InvestigationContext:
    """Passed to every tool. Tools read the offer from here, not from LLM arguments,
    so the model can't paraphrase, truncate or alter the text being analysed."""

    case: JobOfferCase
    llm: Any  # ClaudeLLM or OfflineLLM; web tools use it for web_search sub-calls
    settings: Settings = field(default_factory=Settings)
