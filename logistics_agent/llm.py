"""LLM client factory.

gpt-4o-mini: classification, drafting, router loop (cost-efficient).
gpt-4o:      independent validation and judging (stronger reasoning).
"""
from dataclasses import dataclass

from langchain_openai import ChatOpenAI

from logistics_agent.config import Settings


@dataclass(frozen=True)
class LLMs:
    fast: ChatOpenAI
    validator: ChatOpenAI


def build_llms(settings: Settings) -> LLMs:
    common = dict(
        temperature=0,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    return LLMs(
        fast=ChatOpenAI(model=settings.fast_model, **common),
        validator=ChatOpenAI(model=settings.validator_model, **common),
    )
