"""图组装与编译

Week 5 升级：Plan-and-Execute 模式 + 多轮对话持久化
Week 7 升级：自适应规划层（check_results + web_search_fallback）

图结构:
    START → planner → [route_after_plan]
                        ├→ executor → [route_after_executor]
                        │                ├→ synthesizer → END        (结果充足)
                        │                └→ web_search_fallback → synthesizer → END  (结果不足)
                        └→ synthesizer → END                        (闲聊等无需工具的问题)

多轮对话：通过 checkpointer 保存对话状态，thread_id 区分不同会话
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from course_agent.state import AgentState
from course_agent.nodes import (
    planner_node,
    route_after_plan,
    executor_node,
    route_after_executor,
    web_search_fallback_node,
    synthesizer_node,
    llm_node,
    tool_node,
    should_continue,
    force_search_node,
)

# 多轮对话状态持久化（内存级，重启丢失；生产环境可用 SQLite/PostgreSQL）
memory = MemorySaver()


def build_graph():
    """组装 Plan-and-Execute Agent 图（含自适应规划层 + checkpointer）"""
    builder = StateGraph(AgentState)

    builder.add_node("planner", planner_node)
    builder.add_node("executor", executor_node)
    builder.add_node("web_search_fallback", web_search_fallback_node)
    builder.add_node("synthesizer", synthesizer_node)

    builder.add_edge(START, "planner")
    builder.add_conditional_edges("planner", route_after_plan, ["executor", "synthesizer"])
    builder.add_conditional_edges("executor", route_after_executor, ["synthesizer", "web_search_fallback"])
    builder.add_edge("web_search_fallback", "synthesizer")
    builder.add_edge("synthesizer", END)

    return builder.compile(checkpointer=memory)


def build_react_graph():
    """组装 ReAct Agent 图（含 checkpointer，保留兼容）"""
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)
    builder.add_node("force_search", force_search_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", "force_search", END])
    builder.add_edge("tools", "llm")
    builder.add_edge("force_search", "llm")

    return builder.compile(checkpointer=memory)


# 默认使用 Plan-and-Execute 模式
graph = build_graph()
# ReAct 模式（备用）
react_graph = build_react_graph()
