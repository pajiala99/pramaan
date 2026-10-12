# PRAMAAN — recruitment-scam investigation agent

*Pramaan (प्रमाण) = evidence / proof.* Homingo Hackathon, Agents track.

Paste a job offer (message or email, plus optional company, website, recruiter email and phone).
A Claude agent decides which checks to run and calls five tools. Every finding becomes a
cited **evidence item**, and a **fixed rulebook, not the LLM**, turns that evidence into one of
four verdicts: **Likely genuine · Suspicious · Likely scam · Not enough evidence**.

```
  offer ─▶ Claude (planner) ──tool calls──▶ 5 tools ──▶ Evidence ledger ──▶ rulebook ──▶ verdict + report
             ▲   budget, timeouts, retries      │          (cited facts)      (no LLM)
             └──────────── tool results ◀───────┘
             (if Claude is unavailable: a fixed playbook calls the same tools)
```

## Quick start

```bash
cd pramaan
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # optional: add ANTHROPIC_API_KEY for agent mode
streamlit run app.py
```

Without an API key the app runs in **offline playbook mode**. The message, domain and WHOIS
checks still work; the two web-search checks are reported as "skipped (needs Claude API)".

```bash
pytest -q                     # 61 tests, no API key or internet needed
```

## The five tools

| Tool | What it does | How | Evidence group |
|---|---|---|---|
| `extract_entities` | Company, role, emails, phones, UPI IDs, URLs/domains, amounts, payment sentences | Regex + Public Suffix List (deterministic) | (feeds the others) |
| `detect_red_flags` | Fees/deposits, UPI/crypto payment, OTP/PIN asks, task scams, unrealistic pay, WhatsApp-only, no interview, urgency | Fixed regex rules with severities; every hit is cited with a quote; negations ("we never charge a fee") are skipped | message content |
| `check_domain` | Free-mail recruiter, lookalike domains (`amaz0n-careers.in`), off-brand domains, email-vs-website mismatch, domain age, HTTPS | Offline brand list + homoglyph/typo matching; **RDAP** (modern WHOIS) and a TLS handshake | sender identity, domain registry |
| `verify_vacancy` | Is the role on the company's official careers page? Is the offer's domain the official one? | Claude sub-call with the **`web_search` server tool**; answer verified in code | official listing, sender identity |
| `web_reputation` | Scam complaints about the exact phone/email/UPI/domain | Claude sub-call with **`web_search`**; answer verified in code | public reports |

## How the verdict is decided (`pramaan/verdict_rules.py`)

Each evidence item has a **polarity** (risk / trust / info), a **severity** (low / medium / high)
and an **independence group**, meaning the source of truth it came from:

| Group | Question | Source |
|---|---|---|
| message content | What does the offer ask for? | the submitted text |
| sender identity | Is the sender who they claim to be? | email/domain vs. the company |
| domain registry | Is the website infrastructure real? | RDAP registry, TLS |
| official listing | Does the company itself advertise this job? | company careers page |
| public reports | Have other people reported it? | complaints on the web |

The rules are ordered and the first match wins:

| Rule | Verdict | Condition |
|---|---|---|
| R1 | **Likely scam** | ≥1 **high** risk (a direct fraud indicator such as a payment ask, a lookalike domain, or a verified complaint naming the exact number) **and** medium+ risk in **≥2 independent groups** |
| R2 | **Suspicious** | any other medium+ risk, including a direct fraud indicator seen in only one group |
| R3 | **Likely genuine** | no medium+ risk, medium+ trust in **≥2 independent groups**, and the message was scanned |
| R4 | **Not enough evidence** | everything else |

"Asks for a fee" and "asks via UPI" both come from reading the message, so they count as **one**
source. To reach *Likely scam* you need corroboration from a different source. Low-severity
items (urgency, a young domain) are shown to the user but never move the verdict.

## Safety

- **Two-independent-sources rule.** R1 can't fire on a single source (tested in `tests/test_rules.py`).
- **Tool-call budget.** At most `max_tool_calls` (default 8) executions and 10 model turns. Extra
  calls are refused, and the model is then forced to answer in text (`tool_choice: none`).
- **Graceful failure.** Every tool runs through `execute_safely`: input validation, a timeout,
  and exceptions caught. A check that breaks is recorded as *check failed* and one that can't
  run in this mode as *skipped*. Neither ever counts as evidence.
- **Real SDK path tested without a key.** `tests/test_sdk_integration.py` runs the actual
  Anthropic SDK against a mocked Messages API. It checks request shape, thinking blocks echoed
  back unchanged, `pause_turn` resumption, a rejected key falling back to the playbook, and the
  retry when the fallback beta is unavailable.
- **Fallback.** If Claude is unavailable (no key, network, refusal), a fixed playbook runs the
  same tools. Anthropic's server-side refusal fallback (`fallbacks: "default"`) is also enabled.
- **Safety net.** The deterministic tools (`extract_entities`, `detect_red_flags`, `check_domain`)
  always run, even if the model skipped them.
- **No hallucinated citations.** A URL from the web tools only counts if the `web_search` tool
  really returned it in that request. A complaint is graded *high* only if the identifier
  literally appears in the cited snippet or title.
- **Prompt-injection resistant.** The offer is fenced as untrusted `<offer>` data, the tools read
  the offer directly (not from LLM arguments), and the LLM can't set the verdict.
  `tests/test_samples.py` includes an injection attempt that still ends as *Likely scam*.
- **Brand fairness.** "Scammers impersonate Amazon" warnings are context, not evidence, and a
  well-known company is never itself labelled fake.

## Reusable agent core

`agent_core/` knows nothing about job offers:

| File | Responsibility |
|---|---|
| `evidence.py` | `Evidence`, `Citation`, `EvidenceLedger` (dedup, coverage per group) |
| `tools.py` | `Tool` contract, registry, `execute_safely` (validation, timeout, error capture) |
| `runner.py` | The Claude tool-use loop: budget, caching, playbook fallback, safety net |
| `rules.py` | First-match rule engine → `Decision` |
| `llm.py` | The only file that imports the Anthropic SDK; maps failures to `LLMUnavailable` |
| `trace.py` | Live trace events for the UI |

To reuse it for another problem (e.g. fake-invoice or phishing-link checks), write new tools,
evidence groups, a rulebook, a system prompt and a playbook, then call `AgentRunner.run()`, as
`pramaan/investigate.py` does. `tests/test_runner.py` exercises the core with toy tools only.

## Sample cases (`samples/`)

| Sample | Expected | Why |
|---|---|---|
| `1_scam_amazon_wfh` | Likely scam | ₹499 "refundable registration fee" to a UPI ID (message, high) + gmail recruiter and Amazon-lookalike domain (identity, high) |
| `2_genuine_infosys_campus` | Likely genuine | No asks; the recruiter and portal are on `infosys.com`, which is official (identity) and registered for decades (registry). Offline without internet → *Not enough evidence* |
| `3_ambiguous_startup_intern` | Suspicious | No money asked and a real interview offered, but a gmail recruiter and an unverifiable website: risk without a direct fraud indicator |

All names, numbers and domains in the samples are illustrative, written for this demo.

## Project layout

```
pramaan/
├── app.py                  Streamlit UI (live trace, verdict, evidence, export)
├── agent_core/             reusable agent core (see above)
├── pramaan/
│   ├── tools/              the five tools
│   ├── websearch.py        web_search sub-calls + URL verification
│   ├── domains.py          offline domain knowledge (PSL, free-mail, brands, lookalikes)
│   ├── signals.py          the five evidence groups
│   ├── verdict_rules.py    the rulebook
│   ├── prompts.py          system prompt
│   ├── investigate.py      wiring: runner + tools + playbook + rules
│   └── report.py           Markdown / JSON report
├── samples/                3 cases
└── tests/                  61 offline tests (scripted Claude, real SDK on a mocked API, faked network)
```

## AI tools used

- **Claude Opus 5.5 via the Anthropic API** (`claude-opus-5-5`) is the planning agent (client
  tool use) and the reader in the two web-search sub-calls (the server-side `web_search_20260209`
  tool). It uses adaptive thinking with `effort: medium`, plus server-side refusal fallback.
- **Claude Code** (Anthropic's coding agent, also Claude Opus 5.5) helped write the code, tests and
  this README. It worked from a prompt drafted with Claude on claude.ai; every design decision
  and every file was reviewed by the author.

## Limitations

- The known-brand list in `domains.py` is small and curated. Lookalike detection only covers those brands.
- RDAP coverage varies by TLD. When the registry can't answer, that group is marked *not checked*.
- Pattern rules are tuned for English/Hinglish Indian job scams.
- A *Likely genuine* verdict is not a guarantee. Genuine employers never ask candidates to pay.
