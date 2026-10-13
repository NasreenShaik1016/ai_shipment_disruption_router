"""Pydantic schemas: input events, LLM structured outputs, and service results."""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, model_validator

DecisionType = Literal["reroute", "hold", "expedite", "escalate_human", "monitor"]
SeverityLevel = Literal["low", "medium", "high", "critical"]


# ----------------------------------------------------------------- input event
class DisruptionEvent(BaseModel):
    order_id: str
    delivery_status: str
    shipping_mode: str
    days_scheduled: int
    days_real: int
    delay_gap: Optional[int] = None
    late_risk: int
    customer_segment: str
    sales: float
    category: str
    region: str
    city: str = ""

    @model_validator(mode="after")
    def _derive_delay_gap(self) -> "DisruptionEvent":
        if self.delay_gap is None:
            self.delay_gap = self.days_real - self.days_scheduled
        return self


# ------------------------------------------------------- LLM structured output
class DisruptionClassification(BaseModel):
    disruption_types: List[str] = Field(
        description="All applicable: weather, carrier_failure, capacity_breach, customs_hold, standard_delay"
    )
    primary_disruption_type: str
    severity: SeverityLevel = Field(description="low, medium, high, or critical")
    severity_score: float
    confidence: float
    reasoning: str


class RoutingDecision(BaseModel):
    decision: DecisionType = Field(description="reroute, hold, expedite, escalate_human, or monitor")
    action_plan: List[str]
    policy_citations: List[str]
    geographic_routing: Optional[str] = None
    confidence: float
    reasoning: str


class ValidationDimension(BaseModel):
    dimension: str
    score: float
    feedback: str
    remediation: str = Field(description="Concrete action if score < 0.6, empty if passed")


class ValidationResult(BaseModel):
    dimensions: List[ValidationDimension]
    overall_score: float
    issues_found: List[str]
    recommended_next_action: str = Field(description="re-search / redraft / escalate / accept")
    reasoning: str


class ReasoningEvaluation(BaseModel):
    logical_soundness: float
    policy_alignment: float
    completeness: float
    clarity: float
    overall_quality: float
    justification: str


# ------------------------------------------------------------- service output
class DecisionResult(BaseModel):
    thread_id: str
    order_id: str
    status: Literal["COMPLETED", "PENDING_HUMAN_REVIEW"]
    final_decision: Optional[str] = None
    requires_human: bool = False
    escalation_reason: Optional[str] = None
    classification: Dict[str, Any] = Field(default_factory=dict)
    routing_decision: Dict[str, Any] = Field(default_factory=dict)
    validation_result: Dict[str, Any] = Field(default_factory=dict)
    policy_bank: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    reasoning_trail: List[str] = Field(default_factory=list)
    event: Dict[str, Any] = Field(default_factory=dict)
