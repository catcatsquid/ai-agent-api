from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langgraph.graph import END
from paper_agent.state import AgentState
from paper_agent.llm import llm
from paper_agent.tools import tools, tools_by_name

MAX_STEPS = 8

# 把工具绑定到 LLM
llm_with_tools = llm.bind_tools(tools)

# System Prompt - 定义 Agent 的角色和行为边界
SYSTEM_PROMPT = """你是一个计算机课程助教机器人，专门解答【计算机网络】和【操作系统】两门课的知识点问题。

【你可以做的】
1. 使用 search_knowledge 工具搜索课程知识库
2. 基于检索到的内容，用清晰的语言解释知识点
3. 回答时标注知识来源（如"来源：计算机网络-传输层"）

【你不能做的】
- 不回答与这两门课无关的问题（如数学、历史、编程调试等）
- 如果用户问的问题超范围，礼貌拒绝并说"我目前只能回答计算机网络和操作系统的问题哦~"
- 不要编造知识库中没有的内容，不确定就说"这个知识点我需要再学习一下"

【回答风格】
- 先给结论，再展开解释
- 尽量用例子帮助理解
- 重要概念用加粗标注"""


def llm_node(state: AgentState) -> dict:
    new_step = state.get("step_count", 0) + 1

    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response], "step_count": new_step}


def tool_node(state: AgentState) -> dict:
    results = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool_func = tools_by_name[tool_call["name"]]
        result = tool_func.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": results}


def route_after_llm(state: AgentState) -> str:
    last_message = state["messages"][-1]
    step = state.get("step_count", 0)
    has_tool_calls = hasattr(last_message, "tool_calls") and last_message.tool_calls

    # 步数到上限 → 结束
    if step >= MAX_STEPS:
        return END

    # LLM 想调工具 → 走工具
    if has_tool_calls:
        return "tools"

    # LLM 给出回答 → 结束
    return END
