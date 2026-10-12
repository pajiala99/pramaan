"""PRAMAAN - Streamlit UI.  Run:  streamlit run app.py

Layout: sidebar (API key, mode, budget, samples) | offer form | live trace |
verdict + evidence report. The UI only displays; all logic lives in pramaan/.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import replace
from pathlib import Path

import streamlit as st

from agent_core import ClaudeLLM, OfflineLLM
from agent_core import trace as T
from pramaan.case import JobOfferCase
from pramaan.config import Settings
from pramaan.investigate import investigate
from pramaan.report import DISCLAIMER
from pramaan.signals import ALL_GROUPS, GROUP_LABELS
from pramaan.verdict_rules import LIKELY_GENUINE, LIKELY_SCAM, NOT_ENOUGH, SUSPICIOUS

st.set_page_config(page_title="PRAMAAN - job offer scam check", page_icon="🛡️", layout="wide")

FIELDS = ("message", "company", "website", "recruiter_email", "recruiter_phone")
SAMPLES = {
    (s := json.loads(p.read_text()))["name"]: s
    for p in sorted(Path(__file__).parent.joinpath("samples").glob("*.json"))
}
VERDICT_STYLE = {
    LIKELY_SCAM: (st.error, "🚨"),
    SUSPICIOUS: (st.warning, "⚠️"),
    LIKELY_GENUINE: (st.success, "✅"),
    NOT_ENOUGH: (st.info, "❔"),
}
EVENT_ICONS = {
    T.RUN_START: "🚀", T.LLM_REQUEST: "🧠", T.LLM_TEXT: "💬", T.TOOL_CALL: "🔧", T.TOOL_RESULT: "✅",
    T.TOOL_ERROR: "⚠️", T.TOOL_SKIPPED: "⏭️", T.BUDGET: "⛔", T.FALLBACK: "🛟", T.SAFETY_NET: "🧷", T.RUN_END: "🏁",
}


def md_escape(text: str) -> str:
    """Untrusted text (offer quotes, web claims) must not be rendered as markdown/LaTeX."""
    return re.sub(r"([\\`*_\[\]<>#$|~])", r"\\\1", str(text))


def badge(polarity: str, severity: str) -> str:
    if polarity == "trust":
        return "🟢"
    if polarity == "info":
        return "⚪"
    return {"high": "🔴", "medium": "🟠", "low": "🟡"}[severity]


def format_event(ev: T.TraceEvent) -> str:
    head = f"`{ev.elapsed_s:5.1f}s` {EVENT_ICONS.get(ev.kind, '•')} "
    d = ev.data
    if ev.kind == T.TOOL_CALL and "args" in d:
        reason = f" - _{md_escape(d['reason'])}_" if d.get("reason") else ""
        origin = "" if d.get("origin") == "agent" else f" `[{d.get('origin')}]`"
        args = f"  \n&nbsp;&nbsp;&nbsp;&nbsp;args: `{json.dumps(d['args'], ensure_ascii=False)[:200]}`" if d["args"] else ""
        return f"{head}**{d['tool']}**{origin}{reason}{args}"
    if ev.kind == T.TOOL_RESULT:
        items = ", ".join(f"{badge(e['polarity'], e['severity'])} {e['id']} {e['signal']}" for e in d.get("evidence", []))
        return f"{head}{md_escape(ev.message)}" + (f"  \n&nbsp;&nbsp;&nbsp;&nbsp;{items}" if items else "")
    if ev.kind == T.LLM_TEXT:
        return f"{head}_Claude:_ " + md_escape(ev.message[:400]).replace("\n", " ")
    return head + md_escape(ev.message)


def load_sample() -> None:
    sample = SAMPLES.get(st.session_state.sample_choice)
    if sample:
        for field in FIELDS:
            st.session_state[field] = sample["case"].get(field, "")
        st.session_state.pop("report", None)


for field in FIELDS:
    st.session_state.setdefault(field, "")

# ------------------------------------------------------------------ sidebar
settings = Settings()
with st.sidebar:
    st.header("Settings")
    env_key = bool(os.getenv("ANTHROPIC_API_KEY"))
    api_key = st.text_input(
        "Anthropic API key", type="password",
        placeholder="using ANTHROPIC_API_KEY" if env_key else "sk-ant-...",
        help="Kept in this browser session only. Leave empty to use the ANTHROPIC_API_KEY environment variable.",
    )
    have_key = env_key or bool(api_key)
    agent_mode = st.radio(
        "Mode",
        ["Agent - Claude plans the checks", "Offline playbook - no API calls"],
        index=0 if have_key else 1,
        help="Offline runs the same tools in a fixed order; web search checks are skipped.",
    ).startswith("Agent")
    budget = st.slider("Tool-call budget", 3, 12, settings.max_tool_calls)
    st.caption(f"Model: `{settings.model}` · effort: `{settings.effort}`")
    st.divider()
    st.subheader("Sample cases")
    st.selectbox("Load a sample", ["-", *SAMPLES], key="sample_choice", on_change=load_sample)
    if st.session_state.get("sample_choice") in SAMPLES:
        s = SAMPLES[st.session_state.sample_choice]
        st.caption(f"Type: **{s['kind']}** · expected: **{s['expected_verdict']}**  \n{s['why']}")

# ------------------------------------------------------------------ input form
st.title("🛡️ PRAMAAN")
st.caption("Recruitment-scam investigation agent - Claude gathers the evidence, fixed rules decide the verdict.")

with st.form("offer"):
    st.text_area("Job offer (paste the message or email)", key="message", height=230)
    c1, c2 = st.columns(2)
    c1.text_input("Company (optional)", key="company")
    c2.text_input("Company website (optional)", key="website")
    c3, c4 = st.columns(2)
    c3.text_input("Recruiter email (optional)", key="recruiter_email")
    c4.text_input("Recruiter phone (optional)", key="recruiter_phone")
    submitted = st.form_submit_button("Investigate", type="primary", use_container_width=True)

if submitted:
    if not st.session_state.message.strip():
        st.warning("Paste the job offer text first.")
        st.stop()
    case = JobOfferCase(**{f: st.session_state[f] for f in FIELDS})
    run_settings = replace(settings, max_tool_calls=budget)
    llm = ClaudeLLM(model=run_settings.model, effort=run_settings.effort, api_key=api_key or None) if agent_mode else OfflineLLM()

    with st.status("Investigating...", expanded=True) as status:
        report = investigate(case, llm=llm, settings=run_settings, on_event=lambda ev: status.markdown(format_event(ev)))
        status.update(
            label=f"Investigation complete - {report.run.agent_calls}/{budget} agent tool calls, "
                  f"{report.run.extra_calls} playbook/safety-net calls ({report.run.mode} mode)",
            state="complete", expanded=False,
        )
    st.session_state.report = report

# ------------------------------------------------------------------ report
report = st.session_state.get("report")
if report is not None:
    d, facts = report.decision, report.decision.facts
    show, icon = VERDICT_STYLE[d.verdict]
    show(f"### {icon} {d.verdict}\n\n{d.explanation}")
    st.caption(f"Rule fired: `{d.rule_id}` · mode: {report.run.mode} · stop reason: {report.run.stop_reason}")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Risk sources", f"{len(facts['risk_groups'])} / {len(ALL_GROUPS)}",
              help="Independent evidence groups with medium or high risk")
    m2.metric("Trust sources", f"{len(facts['trust_groups'])} / {len(ALL_GROUPS)}",
              help="Independent evidence groups with medium or high trust")
    m3.metric("Evidence", len(report.run.ledger.items))
    m4.metric("Tool calls", f"{report.run.agent_calls} / {report.budget}",
              help="Calls planned by Claude, out of the budget. Playbook and safety-net calls are counted separately.")

    tab_ev, tab_trace, tab_entities, tab_notes, tab_export = st.tabs(
        ["Evidence", "Tool-call trace", "Extracted entities", "Agent notes", "Export"]
    )
    with tab_ev:
        by_group = report.evidence_by_group()
        for group in ALL_GROUPS:
            items = by_group.get(group, [])
            strong = any(e.severity != "low" and e.polarity != "info" for e in items)
            with st.expander(f"{GROUP_LABELS[group]} · {report.group_status(group)} · {len(items)} item(s)", expanded=strong):
                if not items:
                    st.caption("No evidence recorded for this source.")
                for e in items:
                    st.markdown(f"{badge(e.polarity, e.severity)} **{e.id}** `{e.signal}` ({e.polarity}, {e.severity}) - {md_escape(e.summary)}")
                    for c in e.citations:
                        if c.kind == "url":
                            st.caption(f"↳ source: [{md_escape(c.title or c.ref)}]({c.ref})")
                        elif c.kind == "quote":
                            st.caption(f"↳ quote: “{md_escape(c.ref)}”")
                        else:
                            st.caption(f"↳ lookup: {md_escape(c.ref)}")
        st.caption("🔴 high risk · 🟠 medium risk · 🟡 low risk (shown, not used by the rules) · 🟢 trust · ⚪ context")
    with tab_trace:
        for ev in report.run.events:
            st.markdown(format_event(ev))
    with tab_entities:
        st.json(report.entities)
    with tab_notes:
        if report.run.notes:
            st.caption("Claude's summary of its investigation. It is shown for context and is not used by the verdict rules.")
            st.markdown(report.run.notes)
        else:
            st.caption("No agent notes - this run used the offline playbook.")
    with tab_export:
        st.download_button("Download report (Markdown)", report.to_markdown(), "pramaan_report.md", "text/markdown")
        st.download_button("Download report (JSON)", json.dumps(report.to_dict(), indent=2, default=str), "pramaan_report.json", "application/json")
    st.caption(DISCLAIMER)
