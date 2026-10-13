"""Workflow state (persisted by LangGraph) and per-run session (tool scratchpad)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, Dict, List, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class WorkflowState(TypedDict, total=False):
    event_data: Dict[str, Any]
    classification: Dict[str, Any]
    messages: Annotated[List[AnyMessage], add_messages]
    agent_iterations: int
    routing_decision: Dict[str, Any]
    validation_result: Dict[str, Any]
    policy_bank: Dict[str, Dict[str, Any]]
    final_decision: str
    requires_human: bool
    escalation_reason: str
    human_resolution: Dict[str, Any]
    reasoning_trail: List[str]


@dataclass
class Session:
    """Mutable data shared by the router tools during ONE workflow run.

    A new Session is created for every run, so concurrent runs never share state.
    """

    event: Dict[str, Any] = field(default_factory=dict)
    classification: Dict[str, Any] = field(default_factory=dict)
    policy_bank: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    search_count: int = 0
    decision: Dict[str, Any] = field(default_factory=dict)
    validation: Dict[str, Any] = field(default_factory=dict)
    validation_fail_count: int = 0

    def reset(self) -> None:
        self.event = {}
        self.classification = {}
        self.policy_bank = {}
        self.search_count = 0
        self.decision = {}
        self.validation = {}
        self.validation_fail_count = 0
