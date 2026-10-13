"""LangGraph workflow assembly."""
from __future__ import annotations

from typing import Any

from langchain_chroma import Chroma
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from logistics_agent.chains import Chains
from logistics_agent.classifier import make_classify_node, make_confidence_gate
from logistics_agent.config import Settings
from logistics_agent.llm import LLMs
from logistics_agent.router import (
    make_after_check_route, make_agent_node, make_agent_should_continue,
    make_safety_check, make_setup_node,
)
from logistics_agent.state import Session, WorkflowState
from logistics_agent.terminal_nodes import make_deliver_node, make_escalate_node
from logistics_agent.tools import build_router_tools


def build_workflow(
    *, settings: Settings, llms: LLMs, chains: Chains, vectorstore: Chroma,
    session: Session, checkpointer: Any,
):
    """Compile a workflow bound to ONE session (one run / one thread)."""
    tools = build_router_tools(session, vectorstore, chains, settings)
    router_llm = llms.fast.bind_tools(tools)

    g = StateGraph(WorkflowState)
    g.add_node("classify", make_classify_node(chains.classify))
    g.add_node("setup", make_setup_node(session, settings))
    g.add_node("agent", make_agent_node(router_llm))
    g.add_node("tools", ToolNode(tools))
    g.add_node("check", make_safety_check(session, settings))
    g.add_node("deliver", make_deliver_node(session))
    g.add_node("escalate", make_escalate_node(session, settings))

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", make_confidence_gate(settings),
                            {"setup": "setup", "escalate": "escalate"})
    g.add_edge("setup", "agent")
    g.add_conditional_edges("agent", make_agent_should_continue(session, settings),
                            {"tools": "tools", "check": "check"})
    g.add_edge("tools", "agent")
    g.add_conditional_edges("check", make_after_check_route(session, settings),
                            {"agent": "agent", "deliver": "deliver", "escalate": "escalate"})
    g.add_edge("deliver", END)
    g.add_edge("escalate", END)

    return g.compile(checkpointer=checkpointer)
