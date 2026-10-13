"""Batch evaluation of test cases against the decision service."""
from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd

from logistics_agent.data_loader import row_to_event
from logistics_agent.evaluation import metrics
from logistics_agent.evaluation.reporting import print_case_report
from logistics_agent.service import DecisionService

logger = logging.getLogger(__name__)


class EvaluationRunner:
    def __init__(
        self,
        service: DecisionService,
        judge: metrics.ReasoningJudge,
        *,
        step_observer: Optional[Callable[[str, Any], None]] = None,
        auto_approve_escalations: bool = True,
    ):
        self._service = service
        self._judge = judge
        self._observer = step_observer
        self._auto_approve = auto_approve_escalations

    def run_case(self, label: str, row: pd.Series) -> Dict[str, Any]:
        event = row_to_event(row)
        result = self._service.process_event(event, on_step=self._observer)

        if result.status == "PENDING_HUMAN_REVIEW" and self._auto_approve:
            logger.info("Auto-approving escalation for %s (simulated dispatcher)", event.order_id)
            result = self._service.resume(result.thread_id, {"action": "approve"}, on_step=self._observer)

        case_metrics = {
            "task_complete": metrics.task_completed(result),
            "decision_correct": metrics.decision_correct(result, row["decision_expected"]),
            "tool_accuracy": metrics.tool_accuracy(result, row["expected_policies"]),
            "reasoning": self._judge.score(result),
        }
        print_case_report(label, row["decision_expected"], row["expected_policies"], result, case_metrics)

        return {
            "label": label,
            "expected": row["decision_expected"],
            "actual": result.final_decision,
            **case_metrics,
            "escalated": result.requires_human,
            "confidence": result.classification.get("confidence", 0),
        }

    def run_all(self, cases: List[Tuple[str, pd.Series]]) -> pd.DataFrame:
        rows = []
        for label, row in cases:
            try:
                rows.append(self.run_case(label, row))
            except Exception:
                logger.exception("Case %s crashed", label)
                rows.append({"label": label, "expected": row["decision_expected"], "actual": None,
                             "task_complete": False, "decision_correct": False, "tool_accuracy": 0.0,
                             "reasoning": 0.0, "escalated": False, "confidence": 0.0})
        return pd.DataFrame(rows)

    @staticmethod
    def print_summary(df: pd.DataFrame) -> None:
        if df.empty:
            print("No tests executed.")
            return
        total = len(df)
        print(df.to_string(index=False))
        print(f"\nTotal: {total}")
        print(f"Decision Accuracy: {int(df['decision_correct'].sum())}/{total}")
        print(f"Escalated: {int(df['escalated'].sum())}/{total}")
        print(f"Avg Tool Accuracy: {df['tool_accuracy'].mean():.0%}")
        print(f"Avg Reasoning: {df['reasoning'].mean():.0%}")
        print(f"Avg Confidence: {df['confidence'].mean():.2f}")
