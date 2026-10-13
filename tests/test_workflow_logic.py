"""Unit tests for the deterministic parts of the workflow (no LLM / network needed)."""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from logistics_agent.classifier import make_confidence_gate
from logistics_agent.config import ConfigError, Settings
from logistics_agent.evaluation.metrics import decision_correct, task_completed, tool_accuracy
from logistics_agent.router import (
    VALIDATION_NUDGE, make_after_check_route, make_agent_should_continue, make_safety_check,
)
from logistics_agent.schemas import DecisionResult, DisruptionEvent
from logistics_agent.state import Session


@pytest.fixture
def settings():
    return Settings(openai_api_key="test-key")


@pytest.fixture
def session():
    return Session()


def _result(**kw):
    base = dict(thread_id="t", order_id="o", status="COMPLETED")
    base.update(kw)
    return DecisionResult(**base)


def test_settings_require_api_key():
    with pytest.raises(ConfigError):
        Settings(openai_api_key="")


def test_event_derives_delay_gap():
    e = DisruptionEvent(order_id="1", delivery_status="Late", shipping_mode="Standard Class",
                        days_scheduled=2, days_real=6, late_risk=1, customer_segment="Consumer",
                        sales=10.0, category="Books", region="Canada")
    assert e.delay_gap == 4


def test_confidence_gate(settings):
    gate = make_confidence_gate(settings)
    assert gate({"classification": {"confidence": 0.59}}) == "escalate"
    assert gate({"classification": {"confidence": 0.60}}) == "setup"


def test_agent_forced_exit_at_iteration_cap(settings, session):
    cont = make_agent_should_continue(session, settings)
    call = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    assert cont({"messages": [call], "agent_iterations": 1}) == "tools"
    assert cont({"messages": [call], "agent_iterations": settings.max_iterations}) == "check"


def test_agent_exits_when_validation_passed(settings, session):
    session.validation = {"validation_passed": True}
    call = AIMessage(content="", tool_calls=[{"name": "x", "args": {}, "id": "1"}])
    assert make_agent_should_continue(session, settings)({"messages": [call]}) == "check"


def test_route_delivers_on_pass(settings, session):
    session.validation = {"validation_passed": True}
    route = make_after_check_route(session, settings)
    assert route({"messages": [AIMessage(content="DONE")]}) == "deliver"


def test_route_escalates_after_repeated_failures(settings, session):
    session.validation_fail_count = settings.max_validation_failures
    route = make_after_check_route(session, settings)
    assert route({"messages": [AIMessage(content="DONE")]}) == "escalate"


def test_route_escalates_when_nothing_validated(settings, session):
    route = make_after_check_route(session, settings)
    assert route({"messages": [AIMessage(content="DONE")]}) == "escalate"


def test_safety_check_nudges_once_then_stops_at_cap(settings, session):
    check = make_safety_check(session, settings)
    out = check({"messages": [AIMessage(content="DONE")], "agent_iterations": 2})
    assert out["messages"][0].content == VALIDATION_NUDGE
    # Regression: at the iteration cap we must not nudge again (would loop forever).
    assert check({"messages": [AIMessage(content="DONE")],
                  "agent_iterations": settings.max_iterations}) == {"messages": []}


def test_route_returns_to_agent_after_nudge(settings, session):
    route = make_after_check_route(session, settings)
    assert route({"messages": [HumanMessage(content=VALIDATION_NUDGE)]}) == "agent"


def test_metrics():
    r = _result(final_decision="monitor", policy_bank={"POL-004": {}, "POL-001": {}})
    assert task_completed(r)
    assert decision_correct(r, "monitor") and not decision_correct(r, "reroute")
    assert tool_accuracy(r, "POL-004") == 1.0
    assert tool_accuracy(r, "POL-004|POL-006") == 0.5
    assert tool_accuracy(r, float("nan")) == 1.0
