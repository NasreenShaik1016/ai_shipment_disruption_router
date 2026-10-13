# Logistics Disruption Decision System

Multi-agent workflow (LangGraph) that classifies a logistics disruption, retrieves policies (RAG),
drafts a routing decision, validates it with an independent LLM auditor, and escalates to a human when needed.

## Project layout

```
logistics_decision_system/
├── main.py                      # CLI (evaluate / decide)
├── requirements.txt
├── .env.example                 # copy to .env
├── data/                        # put logistics_policy_rulebook.txt + test_cases_curated.csv here
├── logistics_agent/
│   ├── config.py                # validated settings from env (replaces config.json)
│   ├── schemas.py               # Pydantic: input event, LLM outputs, DecisionResult
│   ├── state.py                 # WorkflowState + per-run Session
│   ├── prompts.py               # every prompt + priority rules
│   ├── llm.py / chains.py       # model clients and prompt|LLM chains
│   ├── vector_store.py          # rulebook ingestion -> Chroma
│   ├── classifier.py            # classify node + confidence gate
│   ├── tools.py                 # router tools (factory, session-bound)
│   ├── router.py                # setup / agent / loop-control nodes
│   ├── terminal_nodes.py        # deliver + human-escalation (interrupt)
│   ├── graph.py                 # LangGraph assembly
│   ├── service.py               # DecisionService: process_event() / resume()
│   ├── data_loader.py           # CSV -> DisruptionEvent
│   └── evaluation/              # metrics, runner, console reporting
└── tests/                       # unit tests for deterministic logic (no API calls)
```

## Setup (Windows / VS Code)

```
py -3.12 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
copy .env.example .env         # then fill in your keys
```
Use Python 3.12/3.13 (64-bit): pandas 2.2.3 has no prebuilt wheel for newer versions.

## Run

```
python main.py evaluate --verbose --output evaluation_results.csv
python main.py decide --event event.json
pip install -r requirements-dev.txt && pytest
```

## Use as a library

```python
from logistics_agent import Settings, DecisionService

service = DecisionService(Settings.from_env())
result = service.process_event({...})          # dict or DisruptionEvent
if result.status == "PENDING_HUMAN_REVIEW":
    result = service.resume(result.thread_id, {"action": "approve", "reviewer": "jdoe"})
```

## Changes from the notebook

- Secrets and tunables come from environment variables / `.env`; nothing hard-coded, no `config.json` in the repo.
- The global `session` singleton is gone. Each run gets its own `Session` and tools, so concurrent runs are isolated.
- Fixed the requirements typo `langgraph--1.2.11` -> `langgraph==1.2.11`.
- Fixed a possible infinite loop: if the iteration cap was hit before `self_validate_tool` was ever called, the
  safety check nudged forever. It now stops nudging at the cap and the case escalates.
- Nudge detection uses an exact constant instead of a substring match; `recursion_limit` is set as a backstop.
- Tool failures (search, draft, validate, redraft) are caught and returned as `ERROR` statuses instead of crashing the run.
- Decision and severity use `Literal` types; events are validated by Pydantic; classifier confidence is clamped to 0-1.
- Human review is a real API: `process_event` returns `PENDING_HUMAN_REVIEW`, and `resume()` continues it.
- Evaluation is separate from core logic and isolates crashes per test case.

## Production notes

- **Checkpointer:** the default `MemorySaver` is in-process. Pass a durable one (Postgres/SQLite) to `DecisionService`.
- **Pending reviews:** the session for an interrupted run is held in memory. For multi-worker deployments,
  move `Session` into the persisted graph state, or pin reviews to a worker, before scaling out.
- **Observability:** set `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY`, `LANGCHAIN_PROJECT` for LangSmith tracing.
- **Policy text limit:** the drafter and validator only see the first `policy_snippet_chars` (300) of each policy.
  Raise it if policies are being truncated.
