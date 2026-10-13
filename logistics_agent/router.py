"""Router-agent nodes and deterministic loop control."""
from __future__ import annotations

from typing import Callable, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import Runnable

from logistics_agent.config import Settings
from logistics_agent.prompts import build_router_system_prompt
from logistics_agent.state import Session, WorkflowState

VALIDATION_NUDGE = "You did not call self_validate_tool. This is MANDATORY. Please validate now."


def make_setup_node(session: Session, settings: Settings) -> Callable[[WorkflowState], dict]:
    system_prompt = build_router_system_prompt(settings)

    def setup_node(state: WorkflowState) -> dict:
        """Reset the session and prepare the agent's initial messages."""
        session.reset()
        session.event = state["event_data"]
        session.classification = state["classification"]
        e, c = session.event, session.classification
        context = (
            f"DISRUPTION EVENT:\n"
            f"Order {e['order_id']} | {e['delivery_status']} | {e['shipping_mode']}\n"
            f"Delay: {e['delay_gap']}d | Late risk: {e['late_risk']} | {e['customer_segment']}\n"
            f"Value: ${e['sales']} | {e['category']} | {e['region']}\n\n"
            f"CLASSIFICATION: {', '.join(c['disruption_types'])} | Primary: {c['primary_disruption_type']}\n"
            f"Severity: {c['severity']} ({c['severity_score']}) | Confidence: {c['confidence']}\n\n"
            f"Begin: search, draft, validate. Follow the validator's feedback."
        )
        return {
            "messages": [SystemMessage(content=system_prompt), HumanMessage(content=context)],
            "agent_iterations": 0,
        }

    return setup_node


def make_agent_node(router_llm: Runnable) -> Callable[[WorkflowState], dict]:
    def agent_node(state: WorkflowState) -> dict:
        response = router_llm.invoke(state["messages"])
        return {"messages": [response], "agent_iterations": state.get("agent_iterations", 0) + 1}

    return agent_node


def make_agent_should_continue(
    session: Session, settings: Settings
) -> Callable[[WorkflowState], Literal["tools", "check"]]:
    def agent_should_continue(state: WorkflowState) -> Literal["tools", "check"]:
        """Go to tools if requested; force-exit when validation resolved or caps are hit."""
        if session.validation.get("validation_passed"):
            return "check"
        if session.validation_fail_count >= settings.max_validation_failures:
            return "check"
        if state.get("agent_iterations", 0) >= settings.max_iterations:
            return "check"
        if getattr(state["messages"][-1], "tool_calls", None):
            return "tools"
        return "check"

    return agent_should_continue


def make_safety_check(session: Session, settings: Settings) -> Callable[[WorkflowState], dict]:
    def safety_check(state: WorkflowState) -> dict:
        """Nudge the agent if validation was skipped; otherwise pass through."""
        if session.validation_fail_count >= settings.max_validation_failures:
            return {"messages": []}
        if session.validation.get("validation_passed"):
            return {"messages": []}
        # Never nudge once the iteration cap is hit, or the graph would loop forever.
        if state.get("agent_iterations", 0) >= settings.max_iterations:
            return {"messages": []}

        called = {
            tc["name"]
            for msg in state["messages"]
            if isinstance(msg, AIMessage)
            for tc in (getattr(msg, "tool_calls", None) or [])
        }
        if "self_validate_tool" not in called:
            return {"messages": [HumanMessage(content=VALIDATION_NUDGE)]}
        return {"messages": []}

    return safety_check


def make_after_check_route(
    session: Session, settings: Settings
) -> Callable[[WorkflowState], Literal["agent", "deliver", "escalate"]]:
    def after_check_route(state: WorkflowState) -> Literal["agent", "deliver", "escalate"]:
        last = state["messages"][-1]
        if isinstance(last, HumanMessage) and last.content == VALIDATION_NUDGE:
            return "agent"
        if session.validation_fail_count >= settings.max_validation_failures:
            return "escalate"
        if session.validation.get("validation_passed"):
            return "deliver"
        if session.decision and session.validation.get("overall_score", 0) >= settings.deliver_fallback_score:
            return "deliver"
        return "escalate"

    return after_check_route
