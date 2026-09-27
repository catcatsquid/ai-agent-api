"""Agent 状态定义

Week 5 升级：加入 plan 字段，支持 Plan-and-Execute 模式

State 字段：
    messages: 对话历史（自动追加）
    plan: 执行计划（planner 生成，executor 消费）
"""

from typing import Annotated, Optional
from typing_extensions import TypedDict
from langgraph.graph import add_messages


class AgentState(TypedDict):
    """Agent 的共享状态

    messages: 对话历史，add_messages reducer 自动追加
    plan: 执行计划，planner 节点生成，格式:
        [{"tool": "search_knowledge", "query": "..."},
         {"tool": "web_search", "query": "..."}]
    """
    messages: Annotated[list, add_messages]
    plan: list[dict]
