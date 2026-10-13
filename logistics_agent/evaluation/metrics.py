"""Evaluation metrics operating on DecisionResult objects."""
from __future__ import annotations

import logging
from typing import List, Union

from langchain_core.language_models import BaseChatModel

from logistics_agent.prompts import JUDGE_PROMPT
from logistics_agent.schemas import DecisionResult, ReasoningEvaluation

logger = logging.getLogger(__name__)


def task_completed(result: DecisionResult) -> bool:
    return bool(result.final_decision)


def decision_correct(result: DecisionResult, expected: str) -> bool:
    return (result.final_decision or "") == expected


def tool_accuracy(result: DecisionResult, expected_policies: Union[str, List[str], None]) -> float:
    """Fraction of expected policies that were retrieved into the policy bank."""
    if isinstance(expected_policies, str):
        expected = [p.strip() for p in expected_policies.split("|") if p.strip()]
    elif isinstance(expected_policies, (list, tuple)):
        expected = list(expected_policies)
    else:
        expected = []
    if not expected:
        return 1.0
    return sum(1 for p in expected if p in result.policy_bank) / len(expected)


class ReasoningJudge:
    """LLM-as-judge for decision reasoning quality."""

    def __init__(self, llm: BaseChatModel):
        self._chain = JUDGE_PROMPT | llm.with_structured_output(ReasoningEvaluation)

    def score(self, result: DecisionResult) -> float:
        routing = result.routing_decision
        if not routing:
            return 0.0
        try:
            evaluation = self._chain.invoke({
                "context": f"severity={result.classification.get('severity', '')} "
                           f"delay={result.event.get('delay_gap')}d",
                "decision": routing.get("decision", ""),
                "confidence": routing.get("confidence", 0),
                "reasoning": routing.get("reasoning", "")[:300],
                "citations": ", ".join(routing.get("policy_citations", [])),
                "plan": "; ".join(routing.get("action_plan", [])[:3]),
            })
            return evaluation.overall_quality
        except Exception:
            logger.exception("Reasoning judge failed")
            return 0.0
