"""agent_core.runner with toy tools - proves the core is domain-agnostic and safe."""
import time

from agent_core import AgentRunner, Budget, Evidence, Tool, ToolOutput, ToolRegistry, ToolSkipped

from .fakes import BrokenLLM, ScriptedLLM, response, text, tool_use

calls: list[str] = []


def _ok(args, ctx):
    calls.append("ok")
    return ToolOutput({"echo": args}, [Evidence("PING", "g1", "risk", "medium", "pong", subject="x")])


def _boom(args, ctx):
    raise RuntimeError("network exploded")


def _slow(args, ctx):
    time.sleep(2)
    return ToolOutput({})


def _needs_service(args, ctx):
    raise ToolSkipped("service unavailable in this mode")


TOOLS = ToolRegistry([
    Tool("ok", "always works", {"type": "object", "properties": {"n": {"type": "integer"}}}, _ok, groups=("g1",)),
    Tool("boom", "always fails", {"type": "object", "properties": {}}, _boom, groups=("g2",)),
    Tool("slow", "too slow", {"type": "object", "properties": {}}, _slow, groups=("g3",), timeout_s=0.2),
    Tool("optional", "needs a service", {"type": "object", "properties": {}}, _needs_service, groups=("g4",)),
])


def make_runner(llm, max_calls=8):
    return AgentRunner(llm, TOOLS, system_prompt="test", budget=Budget(max_tool_calls=max_calls, max_turns=5))


def setup_function():
    calls.clear()


def test_reason_field_is_required_in_schema_and_stripped_before_the_tool():
    schema = TOOLS.get("ok").api_schema()["input_schema"]
    assert "reason" in schema["required"]
    llm = ScriptedLLM([response(tool_use("t1", "ok", n=1)), response(text("done"), stop="end_turn")])
    result = make_runner(llm).run("go", ctx=None)
    assert result.tool_outputs["ok"][0] == {"echo": {"n": 1}}  # no 'reason' reached the tool
    assert result.stop_reason == "finished" and result.notes == "done"


def test_budget_refuses_extra_calls_and_forces_a_text_answer():
    llm = ScriptedLLM([
        response(tool_use("a", "ok", n=1), tool_use("b", "ok", n=2), tool_use("c", "ok", n=3)),
        response(text("summary"), stop="end_turn"),
    ])
    result = make_runner(llm, max_calls=2).run("go", ctx=None)
    assert result.agent_calls == 2 and calls == ["ok", "ok"]
    tool_results = llm.requests[1]["messages"][-1]["content"]
    assert [r["is_error"] for r in tool_results] == [False, False, True]  # third call refused
    assert llm.requests[1]["tool_choice"] == {"type": "none"}
    assert result.stop_reason == "budget_exhausted"


def test_failing_tool_is_reported_not_raised():
    llm = ScriptedLLM([response(tool_use("a", "boom")), response(text("ok"), stop="end_turn")])
    result = make_runner(llm).run("go", ctx=None)
    sent = llm.requests[1]["messages"][-1]["content"][0]
    assert sent["is_error"] is True and "network exploded" in sent["content"]
    assert result.ledger.tool_status["boom"] == "error"
    assert "g2" in result.ledger.failed_groups and "g2" not in result.ledger.checked_groups


def test_skipped_tool_is_distinct_from_a_failed_one():
    llm = ScriptedLLM([response(tool_use("a", "optional")), response(text("ok"), stop="end_turn")])
    result = make_runner(llm).run("go", ctx=None)
    sent = llm.requests[1]["messages"][-1]["content"][0]
    assert '"status": "skipped"' in sent["content"]
    assert result.ledger.tool_status["optional"] == "skipped"
    assert "g4" in result.ledger.skipped_groups and "g4" not in result.ledger.failed_groups
    assert any(e.kind == "tool_skipped" for e in result.events)


def test_slow_tool_times_out():
    llm = ScriptedLLM([response(tool_use("a", "slow")), response(text("ok"), stop="end_turn")])
    started = time.monotonic()
    result = make_runner(llm).run("go", ctx=None)
    assert time.monotonic() - started < 1.5
    assert any("timed out" in e.message for e in result.events)


def test_repeated_call_is_cached_and_evidence_not_double_counted():
    llm = ScriptedLLM([
        response(tool_use("a", "ok", n=1)),
        response(tool_use("b", "ok", n=1)),
        response(text("done"), stop="end_turn"),
    ])
    result = make_runner(llm).run("go", ctx=None)
    assert calls == ["ok"]  # second call served from cache
    assert len(result.ledger.items) == 1


def test_llm_outage_switches_to_playbook():
    def playbook(call):
        call("ok", {"n": 7})
        call("boom", {})

    result = make_runner(BrokenLLM()).run("go", ctx=None, playbook=playbook)
    assert result.mode == "playbook" and result.stop_reason.startswith("llm_unavailable")
    assert result.ledger.tool_status == {"ok": "ok", "boom": "error"}
    assert result.extra_calls == 2


def test_safety_net_runs_required_tools_the_model_skipped():
    llm = ScriptedLLM([response(text("nothing to do"), stop="end_turn")])
    result = make_runner(llm).run("go", ctx=None, ensure=[("ok", {})])
    assert calls == ["ok"] and result.extra_calls == 1
    assert any(e.kind == "safety_net" for e in result.events)


def test_unknown_tool_does_not_crash():
    llm = ScriptedLLM([response(tool_use("a", "nope")), response(text("ok"), stop="end_turn")])
    result = make_runner(llm).run("go", ctx=None)
    assert llm.requests[1]["messages"][-1]["content"][0]["is_error"] is True
    assert result.stop_reason == "finished"
