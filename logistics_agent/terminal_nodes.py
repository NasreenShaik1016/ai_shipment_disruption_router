"""Terminal nodes: deliver a validated decision or escalate to a human."""
from __future__ import annotations

import logging
from typing import Callable

from langgraph.types import interrupt

from logistics_agent.config import Settings
from logistics_agent.state import Session, WorkflowState

logger = logging.getLogger(__name__)


def make_deliver_node(session: Session) -> Callable[[WorkflowState], dict]:
    def deliver_node(state: WorkflowState) -> dict:
        decision = session.decision.get("decision", "monitor")
        trail = state.get("reasoning_trail", []) + [f"Delivered: {decision}"]
        return {
            "final_decision": decision,
            "requires_human": decision == "escalate_human",
            "routing_decision": session.decision,
            "validation_result": session.validation,
            "policy_bank": dict(session.policy_bank),
            "reasoning_trail": trail,
        }

    return deliver_node


def make_escalate_node(session: Session, settings: Settings) -> Callable[[WorkflowState], dict]:
    def escalate_node(state: WorkflowState) -> dict:
        event = state["event_data"]
        conf = state.get("classification", {}).get("confidence", 1.0)

        if conf < settings.confidence_threshold:
            reason = f"Low classifier confidence ({conf:.2f})"
        elif session.validation_fail_count >= settings.max_validation_failures:
            reason = f"Validation failed {session.validation_fail_count} times"
        else:
            reason = "Router agent could not produce a validated decision"

        logger.warning("Escalating order %s: %s", event["order_id"], reason)

        # Pauses the graph. Execution resumes with the human's resolution.
        resolution = interrupt({
            "reason": reason,
            "order_id": event["order_id"],
            "segment": event["customer_segment"],
            "value": event["sales"],
            "status": event["delivery_status"],
            "delay": event["delay_gap"],
        })

        trail = state.get("reasoning_trail", []) + [f"Escalated: {reason}"]
        return {
            "final_decision": "escalate_human",
            "requires_human": True,
            "escalation_reason": reason,
            "human_resolution": resolution if isinstance(resolution, dict) else {"value": resolution},
            "routing_decision": session.decision or {},
            "validation_result": session.validation or {},
            "policy_bank": dict(session.policy_bank),
            "reasoning_trail": trail,
        }

    return escalate_node
