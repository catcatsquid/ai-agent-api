"""Agent 状态定义

Week 5 升级：加入 plan 字段，支持 Plan-and-Execute 模式

State 字段：
    messages: 对话历史（自动追加）
    plan: 执行计划（planner 生成，executor 消费）
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Agent 的共享状态

    messages: 对话历史，add_messages reducer 自动追加
    plan: 执行计划，planner 节点生成，格式:
        [{"tool": "search_knowledge", "args": {"query": "..."}},
         {"tool": "web_search", "args": {"query": "..."}}]
    tool_rounds: ReAct 图里已经执行过多少轮工具调用。
        用来给工具环加**业务级上限** —— 否则循环次数完全由模型决定，
        唯一的上限是 LangGraph 默认的 recursion_limit=10007
        （实测模型每轮都要求调工具时会真实发生 5004 次 LLM 调用）。
    session: **跨轮的会话游标**（刷题进度）。

        为什么需要它：Plan 图的 planner_node 组装历史时会**排除 ToolMessage**、
        并把 assistant 内容截到 200 字，所以上一轮工具结果里的
        `next_group_index` 根本传不到这一轮 —— 学生说"继续下一组"时，
        planner 既没有卷子也没有组号，只能瞎猜或去编题（实测都发生过）。

        ReAct 图天然没这个问题（它把含 ToolMessage 的整段历史给模型，
        游标"在带内"传递），但 Plan 图是**主动丢弃**的。

        ⚠️ 这是**设计，不是补偿**。别把它理解成"框架有缺陷所以打个补丁"：
        让模型从工具结果里"读一个数字出来"本来就不可靠（那是靠注意力搬状态），
        而这里是确定性的字段。框架那边不是修不好，是**显式状态本来就更可靠**。

        它由 checkpointer 落盘保管（graph.py 的 SqliteSaver），所以活得过进程重启。
        1.0.x 曾额外做一份 runtime/plan_cursors.json 的落盘副本兜底，1.2.0 删掉了 ——
        真源只剩这一个字段。

        结构: {"paper": 卷名, "type": 题型, "group_index": 刚出的组号,
               "next_group_index": 下一组, "qids": [刚出的题], "finished": bool}
    """
    messages: Annotated[list, add_messages]
    plan: list[dict]
    tool_rounds: int
    session: dict
