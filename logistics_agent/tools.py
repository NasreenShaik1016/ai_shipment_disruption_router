"""Router-agent tools.

Tools are created by a factory so each workflow run gets tools bound to its OWN
Session (no module-level mutable state -> safe for concurrent runs).
"""
from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_chroma import Chroma
from langchain_core.tools import BaseTool, tool
from langsmith import traceable

from logistics_agent.chains import Chains
from logistics_agent.config import Settings
from logistics_agent.state import Session

logger = logging.getLogger(__name__)

_EVENT_KEYS = (
    "order_id", "delivery_status", "shipping_mode", "delay_gap",
    "customer_segment", "sales", "category", "region",
)


def _event_inputs(event: Dict[str, Any]) -> Dict[str, Any]:
    return {key: event[key] for key in _EVENT_KEYS}


def _format_policies(session: Session, max_chars: int) -> str:
    return "\n".join(f"{pid}: {d['content'][:max_chars]}" for pid, d in session.policy_bank.items())


def build_router_tools(
    session: Session, vectorstore: Chroma, chains: Chains, settings: Settings
) -> List[BaseTool]:

    @tool
    @traceable(name="search_policies_tool")
    def search_policies_tool(query: str, k: int = 5) -> str:
        """
        Search the policy vector store.
        Provide a query string and k (3-10).
        Maximum searches per run are limited.
        Results are added to the shared policy bank.
        """
        if session.search_count >= settings.max_searches:
            return json.dumps({
                "status": "SEARCH_LIMIT_REACHED",
                "policies_in_bank": list(session.policy_bank.keys()),
            })

        k = max(3, min(int(k), 10))
        try:
            results = vectorstore.similarity_search_with_score(query, k=k)
        except Exception:
            logger.exception("Policy search failed")
            return json.dumps({"status": "ERROR", "message": "Policy search failed"})

        session.search_count += 1
        new_ids: List[str] = []
        for doc, score in results:
            pid = doc.metadata.get("policy_id", "UNKNOWN")
            if pid in session.policy_bank:
                if doc.page_content.strip() not in session.policy_bank[pid]["content"]:
                    session.policy_bank[pid]["content"] += "\n\n" + doc.page_content
                    session.policy_bank[pid]["chunks"] += 1
            else:
                session.policy_bank[pid] = {"content": doc.page_content, "score": score, "chunks": 1}
                new_ids.append(pid)

        return json.dumps({
            "status": "SUCCESS",
            "new_policies": new_ids,
            "all_policies": list(session.policy_bank.keys()),
        })

    @tool
    @traceable(name="draft_decision_tool")
    def draft_decision_tool() -> str:
        """Draft a routing decision from the retrieved policies. Call after searching."""
        if not session.policy_bank:
            return json.dumps({"status": "NO_POLICIES", "message": "Call search_policies_tool first"})

        try:
            result = chains.draft.invoke({
                "disruption_types": ", ".join(session.classification["disruption_types"]),
                "severity": session.classification["severity"],
                **_event_inputs(session.event),
                "policies": _format_policies(session, settings.policy_snippet_chars),
            })
        except Exception:
            logger.exception("Drafting failed")
            return json.dumps({"status": "ERROR", "message": "Drafting failed"})

        session.decision = result.model_dump()
        return json.dumps({"status": "SUCCESS", "decision": session.decision})

    @tool
    @traceable(name="self_validate_tool")
    def self_validate_tool() -> str:
        """Validate the current decision. Returns pass/fail with per-dimension remediation."""
        if not session.decision:
            return json.dumps({"status": "NO_DECISION", "recommended_next_action": "redraft"})

        # Deterministic pre-check: every cited policy must have been retrieved.
        cited = session.decision.get("policy_citations", [])
        bank_ids = set(session.policy_bank.keys())
        hallucinated = [p for p in cited if p not in bank_ids]
        if hallucinated:
            session.validation_fail_count += 1
            session.validation = {
                "validation_passed": False,
                "overall_score": 0.0,
                "issues": [f"HALLUCINATED CITATIONS: {hallucinated} not in {list(bank_ids)}"],
                "remediation": [],
                "recommended_next_action":
                    "redraft: remove hallucinated citations, use only retrieved policies",
            }
            return json.dumps({"status": "FAIL", "fail_count": session.validation_fail_count,
                               **session.validation})

        try:
            result = chains.validate.invoke({
                "decision": json.dumps(session.decision, indent=2),
                **_event_inputs(session.event),
                "severity": session.classification["severity"],
                "policies": _format_policies(session, settings.policy_snippet_chars),
            })
        except Exception:
            logger.exception("Validation failed to run")
            return json.dumps({"status": "ERROR", "message": "Validator unavailable"})

        all_pass = all(d.score >= settings.dimension_threshold for d in result.dimensions)
        passed = all_pass and result.overall_score >= settings.overall_threshold
        if not passed:
            session.validation_fail_count += 1

        issues = list(result.issues_found)
        remediation: List[str] = []
        for d in result.dimensions:
            if d.score < settings.dimension_threshold:
                issues.append(f"{d.dimension} {d.score:.2f}: {d.feedback}")
                if d.remediation:
                    remediation.append(f"[{d.dimension}] {d.remediation}")

        session.validation = {
            "validation_passed": passed,
            "overall_score": result.overall_score,
            "dimensions": [d.model_dump() for d in result.dimensions],
            "issues": issues,
            "remediation": remediation,
            "recommended_next_action": "accept" if passed else result.recommended_next_action,
        }
        return json.dumps({"status": "PASS" if passed else "FAIL",
                           "fail_count": session.validation_fail_count, **session.validation})

    @tool
    @traceable(name="redraft_decision_tool")
    def redraft_decision_tool() -> str:
        """Revise the decision using validator feedback. Same policies, no new search.
        Call after self_validate_tool returns FAIL with recommended_next_action='redraft'."""
        if not session.decision:
            return json.dumps({"status": "NO_DECISION", "message": "Draft first before redrafting"})
        if not session.validation:
            return json.dumps({"status": "NOTHING_TO_REDRAFT", "message": "No prior validation feedback"})

        try:
            result = chains.redraft.invoke({
                "previous_decision": json.dumps(session.decision, indent=2),
                "issues": "\n".join(f"- {i}" for i in session.validation.get("issues", [])) or "(none)",
                "remediation": "\n".join(f"- {r}" for r in session.validation.get("remediation", [])) or "(none)",
                "policies": _format_policies(session, settings.policy_snippet_chars),
                **_event_inputs(session.event),
            })
        except Exception:
            logger.exception("Redraft failed")
            return json.dumps({"status": "ERROR", "message": "Redraft failed"})

        session.decision = result.model_dump()
        return json.dumps({"status": "SUCCESS", "decision": session.decision})

    return [search_policies_tool, draft_decision_tool, redraft_decision_tool, self_validate_tool]
