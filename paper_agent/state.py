"""Agent 状态定义

State 是 LangGraph 的核心概念：一个在所有节点间共享的数据结构。
每个节点读取它、更新它，图负责把更新传递给下一个节点。

对比你第3周 agent_api.py 里手动管理的 messages 列表：
  messages = [{"role": "user", "content": question}]  # 手动初始化
  messages.append(message)                             # 手动追加
  messages.append({"role": "tool", ...})               # 手动追加工具结果

LangGraph 用 Annotated + add_messages 替代了这些手动操作：
  新消息自动追加到列表，不会覆盖。
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import add_messages
from langchain_core.messages import AnyMessage


class AgentState(TypedDict):
    """Agent 的共享状态

    字段:
        messages: 对话历史，使用 add_messages reducer 自动追加新消息
    """
    messages: Annotated[list[AnyMessage], add_messages]
