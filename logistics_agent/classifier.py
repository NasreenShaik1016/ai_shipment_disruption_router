"""Classifier node and deterministic confidence gate."""
from __future__ import annotations

from typing import Callable, Literal

from langchain_core.runnables import Runnable

from logistics_agent.config import Settings
from logistics_agent.state import WorkflowState


def make_classify_node(chain: Runnable) -> Callable[[WorkflowState], dict]:
    def classify_node(state: WorkflowState) -> dict:
        result = chain.invoke(state["event_data"])
        c = result.model_dump()
        c["confidence"] = min(max(float(c["confidence"]), 0.0), 1.0)
        trail = state.get("reasoning_trail", []) + [
            f"Classify: {c['primary_disruption_type']} severity={c['severity']} conf={c['confidence']:.2f}"
        ]
        return {"classification": c, "reasoning_trail": trail}

    return classify_node


def make_confidence_gate(settings: Settings) -> Callable[[WorkflowState], Literal["setup", "escalate"]]:
    """Deterministic gate: low-confidence classifications skip the router agent."""

    def confidence_gate(state: WorkflowState) -> Literal["setup", "escalate"]:
        if state["classification"]["confidence"] < settings.confidence_threshold:
            return "escalate"
        return "setup"

    return confidence_gate
