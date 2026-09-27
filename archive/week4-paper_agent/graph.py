from langgraph.graph import StateGraph, START, END
from paper_agent.state import AgentState
from paper_agent.nodes import (
    llm_node,
    tool_node,
    route_after_llm,
)


def build_graph():
    """组装 ReAct Agent 图

    图结构：
                    ┌──────── tools ←──────┐
                    ↓                       │
        START → llm ───→ route_after_llm ──┘
                           ↓
                          END
    """
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")

    builder.add_conditional_edges("llm", route_after_llm, {
        "tools": "tools",
        END: END,
    })

    builder.add_edge("tools", "llm")

    return builder.compile()


graph = build_graph()

DEFAULT_CONFIG = {"recursion_limit": 25}
