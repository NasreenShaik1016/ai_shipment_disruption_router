"""Public entry point: run events through the workflow and resume human reviews."""
from __future__ import annotations

import logging
import threading
import uuid
from typing import Any, Callable, Dict, Optional, Tuple, Union

from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from logistics_agent.chains import build_chains
from logistics_agent.config import Settings
from logistics_agent.graph import build_workflow
from logistics_agent.llm import build_llms
from logistics_agent.schemas import DecisionResult, DisruptionEvent
from logistics_agent.state import Session
from logistics_agent.vector_store import build_vectorstore

logger = logging.getLogger(__name__)

StepObserver = Callable[[str, Any], None]


class UnknownThreadError(KeyError):
    """Raised when resuming a thread that is not awaiting human review."""


def _pending_payload(snapshot: Any) -> Optional[Dict[str, Any]]:
    for task in getattr(snapshot, "tasks", ()) or ():
        for intr in getattr(task, "interrupts", ()) or ():
            return intr.value
    return None


class DecisionService:
    """Builds expensive resources once (LLMs, vector store) and runs events safely.

    Each event gets its own Session + compiled graph, so runs are isolated.
    Pass a durable `checkpointer` (Postgres/SQLite) for production persistence.
    """

    def __init__(self, settings: Settings, *, checkpointer: Any = None, vectorstore: Any = None):
        self.settings = settings
        self.llms = build_llms(settings)
        self.chains = build_chains(self.llms)
        self.vectorstore = vectorstore or build_vectorstore(settings)
        self._checkpointer = checkpointer or MemorySaver()
        self._pending: Dict[str, Tuple[Any, Session]] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ public
    def process_event(
        self,
        event: Union[DisruptionEvent, Dict[str, Any]],
        *,
        thread_id: Optional[str] = None,
        on_step: Optional[StepObserver] = None,
    ) -> DecisionResult:
        evt = event if isinstance(event, DisruptionEvent) else DisruptionEvent.model_validate(event)
        event_dict = evt.model_dump()
        thread_id = thread_id or f"{evt.order_id}_{uuid.uuid4().hex[:8]}"

        session = Session()
        graph = build_workflow(
            settings=self.settings, llms=self.llms, chains=self.chains,
            vectorstore=self.vectorstore, session=session, checkpointer=self._checkpointer,
        )
        config = self._run_config(thread_id)
        initial = {"event_data": event_dict, "reasoning_trail": [], "messages": []}

        logger.info("Processing order %s (thread %s)", evt.order_id, thread_id)
        self._stream(graph, initial, config, on_step)
        return self._finalize(thread_id, event_dict, graph, session, config)

    def resume(
        self, thread_id: str, resolution: Dict[str, Any], *, on_step: Optional[StepObserver] = None
    ) -> DecisionResult:
        with self._lock:
            entry = self._pending.pop(thread_id, None)
        if entry is None:
            raise UnknownThreadError(f"No pending human review for thread {thread_id!r}")

        graph, session = entry
        config = self._run_config(thread_id)
        logger.info("Resuming thread %s with resolution %s", thread_id, resolution)
        self._stream(graph, Command(resume=resolution), config, on_step)
        return self._finalize(thread_id, session.event, graph, session, config)

    # ---------------------------------------------------------------- internals
    def _run_config(self, thread_id: str) -> Dict[str, Any]:
        return {"configurable": {"thread_id": thread_id},
                "recursion_limit": self.settings.recursion_limit}

    @staticmethod
    def _stream(graph: Any, payload: Any, config: Dict[str, Any], on_step: Optional[StepObserver]) -> None:
        for step in graph.stream(payload, config, stream_mode="updates"):
            if on_step:
                for node, update in step.items():
                    on_step(node, update)

    def _finalize(self, thread_id: str, event: Dict[str, Any], graph: Any,
                  session: Session, config: Dict[str, Any]) -> DecisionResult:
        snapshot = graph.get_state(config)
        values = snapshot.values
        pending = bool(snapshot.next)

        payload = None
        if pending:
            payload = _pending_payload(snapshot)
            with self._lock:
                self._pending[thread_id] = (graph, session)

        result = DecisionResult(
            thread_id=thread_id,
            order_id=event["order_id"],
            status="PENDING_HUMAN_REVIEW" if pending else "COMPLETED",
            final_decision=values.get("final_decision"),
            requires_human=values.get("requires_human", pending),
            escalation_reason=values.get("escalation_reason") or (payload or {}).get("reason"),
            classification=values.get("classification", {}),
            routing_decision=values.get("routing_decision") or session.decision,
            validation_result=values.get("validation_result") or session.validation,
            policy_bank=values.get("policy_bank") or dict(session.policy_bank),
            reasoning_trail=values.get("reasoning_trail", []),
            event=event,
        )
        logger.info("Order %s -> status=%s decision=%s", result.order_id, result.status, result.final_decision)
        return result
