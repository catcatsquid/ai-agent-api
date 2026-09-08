"""图组装与编译

把节点和边组装成 StateGraph，编译后就是可执行的 Agent。

对比第3周 agent_api.py 中的 for step in range(max_steps) 循环：
  - 循环体 = llm_node → should_continue → tool_node → llm_node → ...
  - 退出条件 = should_continue 返回 "end"

LangGraph 用 add_edge / add_conditional_edges 声明这些关系，
图自己知道怎么循环、怎么传状态、什么时候停。
"""

from langgraph.graph import StateGraph, START, END
from paper_agent.state import AgentState
from paper_agent.nodes import llm_node, tool_node, should_continue


def build_graph():
    """组装并编译 ReAct Agent 图

    图结构:
        START → llm → [should_continue] → tools → llm → ... → END
                                      ↘ END
    """
    builder = StateGraph(AgentState)

    # 添加节点
    builder.add_node("llm", llm_node)
    builder.add_node("tools", tool_node)

    # 添加边
    builder.add_edge(START, "llm")
    builder.add_conditional_edges("llm", should_continue, ["tools", END])
    builder.add_edge("tools", "llm")  # 回边：工具执行完回到 LLM

    # 编译（编译后才能 invoke）
    return builder.compile()


# 全局编译好的图实例（server.py 直接导入使用）
graph = build_graph()
