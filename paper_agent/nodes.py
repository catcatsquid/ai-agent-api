"""节点函数

LangGraph 的节点就是普通 Python 函数：
  - 接收 state（AgentState）
  - 返回 dict（对 state 的更新）

对比第3周 agent_api.py run_agent() 里 for 循环体内的逻辑：
  - 循环每轮调 requests.post → 对应 llm_node
  - 解析 tool_calls + 执行 TOOL_MAP → 对应 tool_node
  - if message.get("tool_calls") 判断 → 对应 should_continue

区别：之前全塞在一个函数里，现在拆成独立节点，图来调度。
"""

from langchain_core.messages import SystemMessage, ToolMessage
from langgraph.graph import END
from paper_agent.state import AgentState
from paper_agent.llm import llm
from paper_agent.tools import tools, tools_by_name

# 绑定工具到 LLM（让 LLM 知道有哪些工具可以调用）
llm_with_tools = llm.bind_tools(tools)

# System Prompt - 定义 Agent 的角色和行为
SYSTEM_PROMPT = """你是一个课程知识库问答助手，帮助用户解答课程相关的知识点问题。

你可以：
1. 搜索知识库（使用 search_knowledge 工具）查找课程内容
2. 后续将支持：课程大纲查询、知识点总结、练习题检索

当用户问到具体知识点时，先搜索知识库再回答。
如果问题是一般性闲聊，直接回答即可。"""


def llm_node(state: AgentState) -> dict:
    """LLM 节点：调用大模型，决定是否调用工具

    对应第3周 agent_api.py 中的:
        resp = requests.post(URL, ...)
        message = resp.json()["choices"][0]["message"]
    """
    messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """工具节点：执行大模型选择的工具

    对应第3周 agent_api.py 中的:
        func_name = tool_call["function"]["name"]
        func_args = json.loads(tool_call["function"]["arguments"])
        result = TOOL_MAP[func_name](**func_args)
    """
    results = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": results}


def should_continue(state: AgentState) -> str:
    """条件边：判断是否需要继续调用工具

    对应第3周 agent_api.py 中的:
        if message.get("tool_calls"):
            # 继续循环
        else:
            # 返回最终答案

    返回值 "tools" 或 END 对应图中的下一个节点。
    """
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END
