"""图组装与编译

Week 5 升级：Plan-and-Execute 模式 + 多轮对话持久化
Week 7 升级：自适应规划层（check_results + web_search_fallback）

图结构:
    START → planner → [route_after_plan]
                        ├→ executor → [route_after_executor]
                        │                ├→ synthesizer → END        (结果充足)
                        │                └→ web_search_fallback → synthesizer → END  (结果不足)
                        └→ synthesizer → END                        (无需工具的问题)

ReAct 图结构:
    START → llm → [should_continue]
                    ├→ tools → llm  (有工具调用)
                    └→ END           (直接回答)

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
)

# 两张图**各用独立的 checkpointer**。
#
# 原来两者共用同一个模块级 MemorySaver，而命名空间只有 thread_id。
# 于是"同一个 thread_id 换个模式继续说"就会读到对方的会话状态 ——
# 实测：plan 模式跑完后，react 图用同一 thread_id 读到了 plan 的消息和 plan 字段；
# 更糟的是 plan 的 executor 会造出合成 ToolMessage(tool_call_id="plan_0")，
# 而 state 里没有声明过它的 assistant 消息，react 的 llm_node 把这串发出去
# 会被 OpenAI 兼容接口按协议拒绝（400），最后被 safe_llm_invoke 吞成兜底套话 ——
# 也就是"切换模式后模型静默变傻"。
#
# 分开之后即使前端忘了重置 thread_id，两套会话也不会互相污染。
# 保留可传入 checkpointer 的参数，方便换成 SQLite/Postgres 做持久化。
plan_memory = MemorySaver()
react_memory = MemorySaver()


def build_graph(checkpointer=None):
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

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else plan_memory)


def build_react_graph(checkpointer=None):
    """组装 ReAct Agent 图（含 checkpointer）"""
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else react_memory)


# 默认使用 Plan-and-Execute 模式
graph = build_graph()
# ReAct 模式
react_graph = build_react_graph()
