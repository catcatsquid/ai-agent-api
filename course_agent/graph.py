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

import sqlite3
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver
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

# 两张图**各用独立的 checkpointer**（各有各的库文件）。
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
# 保留可传入 checkpointer 的参数，方便测试注入。
#
# ## 为什么从 MemorySaver 换成 SQLite
#
# MemorySaver 把 state 放在**进程内存**里：服务一重启，所有 thread 的状态就没了 ——
# 学生回来变成新会话，"继续下一组 / 继续讲"全部失效。
# 之前只给刷题游标单独做了一份落盘副本（runtime/plan_cursors.json）兜这个底，
# 但那只救一条链路，还造成"内存 + 磁盘"两个真源、读写口径得手工对齐
# （1.0.1 修过一次读写不对称，1.0.2 的重启测试就是照它写的）。
# 换成 SqliteSaver 之后 state **整体**落盘，那份副本连同 _save_cursor/_load_cursor
# 一起删掉了 —— 真源只剩 state 一个。
#
# 两个注意点：
#   · check_same_thread=False —— LangGraph 会在别的线程里跑节点，FastAPI 也会并发
#   · 这里是**模块级长连接**（进程生命周期内不关），不是"每次请求开一个"
_CKPT_DIR = Path(__file__).resolve().parent.parent / "runtime"


def _open_saver(filename: str) -> SqliteSaver:
    """开一个落盘的 checkpointer（库文件放 runtime/，已被 .gitignore 排除）"""
    _CKPT_DIR.mkdir(parents=True, exist_ok=True)
    return SqliteSaver(
        sqlite3.connect(str(_CKPT_DIR / filename), check_same_thread=False))


plan_checkpointer = _open_saver("checkpoints_plan.sqlite3")
react_checkpointer = _open_saver("checkpoints_react.sqlite3")


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

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else plan_checkpointer)


def build_react_graph(checkpointer=None):
    """组装 ReAct Agent 图（含 checkpointer）"""
    builder = StateGraph(AgentState)

    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)

    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")

    return builder.compile(checkpointer=checkpointer if checkpointer is not None else react_checkpointer)


# 默认使用 Plan-and-Execute 模式
graph = build_graph()
# ReAct 模式
react_graph = build_react_graph()
