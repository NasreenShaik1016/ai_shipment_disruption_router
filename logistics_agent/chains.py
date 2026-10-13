"""LangChain runnables (prompt | structured-output LLM)."""
from dataclasses import dataclass

from langchain_core.runnables import Runnable

from logistics_agent.llm import LLMs
from logistics_agent.prompts import (
    CLASSIFY_PROMPT, DRAFT_PROMPT, REDRAFT_PROMPT, VALIDATE_PROMPT,
)
from logistics_agent.schemas import (
    DisruptionClassification, RoutingDecision, ValidationResult,
)


@dataclass(frozen=True)
class Chains:
    classify: Runnable
    draft: Runnable
    validate: Runnable
    redraft: Runnable


def build_chains(llms: LLMs) -> Chains:
    return Chains(
        classify=CLASSIFY_PROMPT | llms.fast.with_structured_output(DisruptionClassification),
        draft=DRAFT_PROMPT | llms.fast.with_structured_output(RoutingDecision),
        validate=VALIDATE_PROMPT | llms.validator.with_structured_output(ValidationResult),
        redraft=REDRAFT_PROMPT | llms.fast.with_structured_output(RoutingDecision),
    )
