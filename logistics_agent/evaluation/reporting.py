"""Console reporting helpers for evaluation runs."""
from __future__ import annotations

import json
from typing import Any, Dict

from logistics_agent.schemas import DecisionResult


def console_step_printer(node: str, update: Any) -> None:
    """Workflow step observer that prints a one-line summary per node."""
    if not isinstance(update, dict):
        return  # e.g. the __interrupt__ marker

    if node == "classify":
        c = update.get("classification", {})
        print(f"  [classify] primary={c.get('primary_disruption_type')} "
              f"severity={c.get('severity')} conf={c.get('confidence', 0):.2f}")
    elif node == "setup":
        print("  [setup] session initialized")
    elif node == "agent":
        msgs = update.get("messages", [])
        if msgs:
            m = msgs[-1]
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    args = json.dumps(tc.get("args", {}), default=str)[:80]
                    print(f"  [agent -> tool] {tc['name']}({args})")
            else:
                content = (m.content or "").strip()[:120].replace("\n", " ")
                print(f"  [agent -> DONE] {content}")
    elif node == "tools":
        for m in update.get("messages", []):
            if not hasattr(m, "name"):
                continue
            try:
                payload = json.loads(m.content)
            except (TypeError, ValueError):
                print(f"  [tool {m.name}] (unparsed)")
                continue
            print(f"  [tool {m.name}] {payload.get('status', '?')}{_tool_extra(m.name, payload)}")
    elif node == "check":
        print("  [check]")
    elif node == "deliver":
        print(f"  [deliver] final={update.get('final_decision', '?')}")
    elif node == "escalate":
        print(f"  [escalate] final={update.get('final_decision', '?')} "
              f"reason={update.get('escalation_reason', '?')}")


def _tool_extra(name: str, payload: Dict[str, Any]) -> str:
    if name == "search_policies_tool":
        return f" new={payload.get('new_policies', [])} all={payload.get('all_policies', [])}"
    if name in ("draft_decision_tool", "redraft_decision_tool"):
        d = payload.get("decision")
        if isinstance(d, dict):
            return f" decision={d.get('decision', '?')} citations={d.get('policy_citations', [])}"
        return ""
    if name == "self_validate_tool":
        extra = (f" score={payload.get('overall_score', 0):.2f} fails={payload.get('fail_count', 0)} "
                 f"next={payload.get('recommended_next_action', '')}")
        for r in payload.get("remediation", [])[:2]:
            extra += f"\n    remediation: {r[:100]}"
        return extra
    return ""


def print_case_report(label: str, expected: str, expected_policies: str,
                      result: DecisionResult, metrics: Dict[str, Any]) -> None:
    print("=" * 72)
    print(f"TEST: {label}")
    print(f"Expected: decision={expected} | policies={expected_policies}")
    print("-" * 72)
    if result.policy_bank:
        print(f"  Retrieved policies: {list(result.policy_bank)}")
    r = result.routing_decision
    if r:
        print(f"  Decision: {r.get('decision')} (conf {r.get('confidence', 0):.2f}) "
              f"citations={r.get('policy_citations', [])}")
        print(f"  Reasoning: {r.get('reasoning', '')[:200]}")
    v = result.validation_result
    if v:
        print(f"  Validation: passed={v.get('validation_passed')} score={v.get('overall_score', 0):.2f}")
        for d in v.get("dimensions", []):
            print(f"    {d['dimension']}: {d['score']:.2f}")
    print(f"  Task Complete:     {'PASS' if metrics['task_complete'] else 'FAIL'}")
    print(f"  Decision Accuracy: {'PASS' if metrics['decision_correct'] else 'FAIL'} "
          f"(AI Decision={result.final_decision})")
    print(f"  Tool Accuracy:     {metrics['tool_accuracy']:.0%}")
    print(f"  Reasoning Quality: {metrics['reasoning']:.0%}")
    print("=" * 72 + "\n")
