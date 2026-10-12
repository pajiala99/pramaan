"""Entry point: wire the generic agent core to the recruitment-scam domain.

    JobOfferCase ──▶ AgentRunner(Claude + 5 tools) ──▶ EvidenceLedger ──▶ rules ──▶ Report
                          │ (Claude unavailable)
                          └──▶ fixed playbook, same tools

Everything domain-specific is passed in here: the tools, the system prompt,
the playbook and the rulebook. agent_core itself knows nothing about jobs.
"""
from __future__ import annotations

from typing import Callable

from agent_core import AgentRunner, Budget, OfflineLLM, TraceEvent, decide

from .case import InvestigationContext, JobOfferCase
from .config import Settings
from .prompts import system_prompt
from .report import Report
from .tools import build_registry
from .tools.extract_entities import extract
from .verdict_rules import RULES, compute_facts

# Deterministic tools (no LLM needed) that must always have run before the rules decide,
# even if the model skipped them - e.g. because text in the offer talked it out of checking.
ALWAYS_RUN = [("extract_entities", {}), ("detect_red_flags", {}), ("check_domain", {})]


def playbook(case: JobOfferCase):
    """Fixed tool order used when Claude can't plan (no key, network down, refusal)."""

    def run(call: Callable[[str, dict], dict | None]) -> None:
        entities = call("extract_entities", {}) or extract(case)
        call("detect_red_flags", {})
        company = case.company or entities.get("known_company") or next(iter(entities.get("company_candidates", [])), "")
        if entities["emails"] or entities["domains"]:
            call("check_domain", {"company": company} if company else {})
        if company:
            roles = entities.get("role_candidates") or []
            call("verify_vacancy", {"company": company, **({"role": roles[0]} if roles else {})})
        if entities["phones"] or entities["upi_ids"] or entities["emails"] or entities["domains"]:
            call("web_reputation", {"company": company} if company else {})

    return run


def investigate(
    case: JobOfferCase,
    *,
    llm=None,
    settings: Settings | None = None,
    on_event: Callable[[TraceEvent], None] | None = None,
) -> Report:
    settings = settings or Settings()
    llm = llm or OfflineLLM()
    runner = AgentRunner(
        llm,
        build_registry(),
        system_prompt=system_prompt(settings.max_tool_calls),
        budget=Budget(max_tool_calls=settings.max_tool_calls, max_turns=settings.max_turns),
        on_event=on_event,
    )
    result = runner.run(
        case.as_prompt(),
        InvestigationContext(case=case, llm=llm, settings=settings),
        playbook=playbook(case),
        ensure=ALWAYS_RUN,
    )
    decision = decide(RULES, compute_facts(result.ledger))
    return Report(case=case, decision=decision, run=result, entities=extract(case), budget=settings.max_tool_calls)
