"""节点函数

Week 5 升级：Plan-and-Execute 模式
Week 7 升级：自适应规划层（check_results + web_search_fallback）

节点流程：
  planner → [route_after_plan]
              ├→ executor → [check_results]
              │                ├→ synthesizer → END        (结果充足)
              │                └→ web_search_fallback → synthesizer → END  (结果不足，补搜)
              └→ synthesizer → END                        (闲聊等无需工具的问题)

对比 Week 4 的 ReAct 循环：
  - ReAct: LLM 每轮决定调哪个工具，逐步试错
  - Plan-Execute: LLM 先一次性规划好用哪些工具，再批量执行
  - 自适应层: executor 执行完检查结果质量，不足时自动补 web_search
"""

import json
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import END
from course_agent.state import AgentState
from course_agent.llm import llm
from course_agent.tools import tools, tools_by_name
from course_agent.errors import safe_llm_invoke, safe_tool_invoke, LLMError
from course_agent.memory import trim_messages, summarize_old_messages

llm_with_tools = llm.bind_tools(tools)

# ============================================================
# System Prompts
# ============================================================

PLANNER_PROMPT = """你是一个课程知识库问答助手的规划器。分析用户问题，制定搜索策略。

可用工具：
1. search_knowledge - 搜索课程知识库（计网+OS），优先使用
2. web_search - 联网搜索，当问题涉及课程外内容或需要最新信息时使用

规划规则：
1. 课程知识点问题（进程、线程、调度、死锁、TCP、UDP等）→ 只用 search_knowledge
2. 课程外扩展问题（Linux内核、最新技术、具体实现等）→ 先 search_knowledge，再 web_search
3. 闲聊或非知识问题 → 返回空计划，直接回答

你必须返回JSON格式：
{"need_tools": true/false, "steps": [{"tool": "search_knowledge", "query": "搜索词"}, ...]}

注意：query 应该是精炼的搜索关键词，不是完整的问句。
最多规划 3 个步骤。"""

SYNTHESIZER_PROMPT = """你是一个课程知识库问答助手。基于检索结果回答用户问题。

回答规则：
1. 优先引用知识库内容（课件 > 课本 > 笔记），网络搜索内容作为补充
2. 标注来源：「来源：xxx（课件/课本/笔记/网络搜索）」
3. 如果知识库结果不足，可结合网络搜索结果，但需区分来源
4. 不能编造来源，只引用检索结果中真实出现的内容
5. 回答要结构化、清晰，适合学生理解"""

CHITCHAT_PROMPT = """你是一个课程知识库问答助手。你可以帮助用户解答计算机网络和操作系统课程的问题。

当用户闲聊时，友好回应，并可以引导用户提问课程相关的问题。
如果用户的问题不涉及课程知识，尽力帮助用户，但不要编造课程内容。"""

REACT_PROMPT = """你是一个课程知识库问答助手，帮助用户解答【计算机网络】和【操作系统】两门课程的知识点问题。

知识库内容覆盖：
- 计算机网络：TCP三次握手/四次挥手、TCP与UDP区别、拥塞控制等
- 操作系统：进程与线程、调度算法、死锁、同步互斥机制等

资料类型与优先级：
- 课件（ppt）：老师上课的课件，是主要学习材料，优先级最高
- 课本（textbook）：参考教材，仅作补充
- 笔记（notes）：课程笔记

重要规则：
1. 当课件内容和课本内容不一致时，以课件为准
2. 回答时优先引用课件中的内容，课本内容仅作为补充说明
3. 标注内容来源时要说明是课件还是课本

你可以使用的工具：
1. search_knowledge：搜索知识库中的课程内容（优先使用）
2. web_search：联网搜索，获取知识库之外的信息（知识库搜索结果不足时使用）
3. get_course_info：查看知识库状态

【工具使用策略】
- 第一步：总是先用 search_knowledge 搜索知识库
- 第二步：检查知识库搜索结果是否足以完整回答问题。如果结果为空、不相关、或覆盖不全，**必须调用 web_search 联网补充搜索**，不能直接用自身知识回答。
- 第三步：综合知识库和网络搜索结果，给出完整回答

【强制规则 - 必须严格遵守】
1. 只要用户问的是计算机网络或操作系统的课程知识点问题，**必须先调用 search_knowledge 搜索知识库**，绝对不能仅凭自己的记忆回答。
2. 回答必须基于搜索结果中的内容，不能编造知识库中没有的信息。
3. 绝对不能编造来源！只有在工具返回的检索结果中真实出现的来源才能标注。如果搜索结果为空或不相关，必须明确说"知识库中未找到相关内容"。
4. 如果搜索结果不足以回答问题，可以调用 web_search 联网补充，或明确告知用户"知识库中相关内容有限"，再补充通用知识，但必须明确区分哪些是知识库内容、哪些是补充说明。
5. 标注来源格式：「来源：xxx（课件/课本/笔记/网络搜索）」。

资料优先级（引用时遵守）：
- 课件内容优先于课本内容
- 两者冲突时，以课件为准
- 知识库内容优先于网络搜索内容"""


# ============================================================
# Planner Node
# ============================================================

def planner_node(state: AgentState) -> dict:
    """规划器节点：分析问题，决定用哪些工具和搜索查询"""
    user_msg = state["messages"][-1]
    user_content = user_msg.content if hasattr(user_msg, "content") else str(user_msg)

    messages = [
        SystemMessage(content=PLANNER_PROMPT),
        HumanMessage(content=user_content),
    ]
    try:
        response = safe_llm_invoke(llm, messages, context="planner")
        raw = response.content.strip()
    except LLMError:
        return {"plan": [{"tool": "search_knowledge", "query": user_content}]}

    # 解析 JSON 计划（LLM 可能包裹在 ```json ... ``` 中）
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        plan = json.loads(raw)
    except json.JSONDecodeError:
        plan = {"need_tools": True, "steps": [{"tool": "search_knowledge", "query": user_content}]}

    return {"plan": plan.get("steps", [])}


def route_after_plan(state: AgentState) -> str:
    """条件边：有计划步骤 → executor，无 → 直接回答"""
    if state.get("plan") and len(state["plan"]) > 0:
        return "executor"
    return "synthesizer"


# ============================================================
# Executor Node
# ============================================================

def executor_node(state: AgentState) -> dict:
    """执行器节点：按计划批量执行所有工具调用"""
    results = []

    for step in state["plan"]:
        tool_name = step.get("tool", "search_knowledge")
        query = step.get("query", "")

        if tool_name not in tools_by_name:
            results.append(ToolMessage(
                content=f"未知工具: {tool_name}",
                tool_call_id=f"plan_{len(results)}",
            ))
            continue

        tool = tools_by_name[tool_name]
        invoke_args = {"query": query} if tool_name != "get_course_info" else {}
        result = safe_tool_invoke(tool, invoke_args, tool_name=tool_name)
        results.append(ToolMessage(
            content=str(result),
            tool_call_id=f"plan_{len(results)}",
        ))

    return {"messages": results}


# ============================================================
# Check Results (自适应规划层)
# ============================================================

# 判断"结果不足"的关键词阈值
_INSUFFICIENT_MARKERS = [
    "知识库中未找到相关内容",
    "知识库中相关内容较少",
    "建议调用 web_search",
    "建议使用 web_search",
]


def route_after_executor(state: AgentState) -> str:
    """条件边：检查 executor 的检索结果是否充足

    判断逻辑：
      1. planner 已经规划了 web_search → 结果应该够，直接走 synthesizer
      2. 检索结果包含"未找到"/"内容较少"等标记 → 不够，走 web_search_fallback
      3. 检索结果总文本太短（<100字）→ 不够，走 web_search_fallback
      4. 其他 → 够，走 synthesizer
    """
    # 如果 planner 已经规划了 web_search，不需要再补
    has_web_search = any(
        step.get("tool") == "web_search" for step in state.get("plan", [])
    )
    if has_web_search:
        return "synthesizer"

    # 收集 executor 返回的 ToolMessage 内容
    tool_contents = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            content = msg.content
            # 如果这条 ToolMessage 是 web_search_fallback 生成的，跳过
            if "网络搜索结果" in content:
                continue
            tool_contents.append(content)

    if not tool_contents:
        return "web_search_fallback"

    combined = "\n".join(tool_contents)

    # 检查"结果不足"标记
    for marker in _INSUFFICIENT_MARKERS:
        if marker in combined:
            return "web_search_fallback"

    # 检查总文本长度（太短说明没搜到什么）
    if len(combined) < 100:
        return "web_search_fallback"

    return "synthesizer"


def web_search_fallback_node(state: AgentState) -> dict:
    """自适应补搜节点：知识库结果不足时自动联网搜索

    从用户问题中提取搜索词，调用 web_search 工具补充信息。
    """
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_msg = human_msgs[-1] if human_msgs else state["messages"][-1]
    user_content = user_msg.content if hasattr(user_msg, "content") else str(user_msg)

    web_tool = tools_by_name["web_search"]
    result = safe_tool_invoke(web_tool, {"query": user_content}, tool_name="web_search")

    return {"messages": [ToolMessage(
        content=f"[自适应补搜] 知识库结果不足，已联网搜索：\n{result}",
        tool_call_id="fallback_web_search",
    )]}


# ============================================================
# Synthesizer Node
# ============================================================

def synthesizer_node(state: AgentState) -> dict:
    """综合器节点：基于检索结果 + 历史对话生成最终回答

    多轮对话：把历史对话上下文传给 LLM，让它记得之前聊了什么。
    记忆管理：对话太长时自动裁剪（滑动窗口 + 摘要压缩），防止爆 token。
    """
    from langchain_core.messages import HumanMessage as HM
    human_msgs = [m for m in state["messages"] if isinstance(m, HM)]
    user_msg = human_msgs[-1] if human_msgs else state["messages"][-1]
    user_content = user_msg.content if hasattr(user_msg, "content") else str(user_msg)

    # 收集当前轮次的工具结果（最后一批 ToolMessage）
    tool_results = []
    for msg in state["messages"]:
        if isinstance(msg, ToolMessage):
            tool_results.append(msg.content)

    # 提取历史对话（排除系统消息和本轮工具结果）
    history_msgs = [
        m for m in state["messages"]
        if not isinstance(m, SystemMessage)
        and not isinstance(m, ToolMessage)
    ]

    # 历史对话裁剪（最多保留最近 10 条，约 5 轮）
    if len(history_msgs) > 10:
        history_msgs = trim_messages(history_msgs, max_messages=10, keep_system=False)

    if tool_results:
        context = "\n\n---\n\n".join(tool_results[-3:])  # 最多取最近3条工具结果
        messages = [
            SystemMessage(content=SYNTHESIZER_PROMPT),
        ]
        # 加上历史对话（让 LLM 有上下文）
        if len(history_msgs) > 1:  # 不止一条用户消息才加历史
            messages.extend(history_msgs[:-1])  # 除了最后一条（就是当前问题）
        messages.append(HumanMessage(content=f"用户问题: {user_content}\n\n检索结果:\n{context}"))
    else:
        messages = [
            SystemMessage(content=CHITCHAT_PROMPT),
        ]
        if len(history_msgs) > 1:
            messages.extend(history_msgs[:-1])
        messages.append(HumanMessage(content=user_content))

    try:
        response = safe_llm_invoke(llm, messages, context="synthesizer")
        return {"messages": [response]}
    except LLMError:
        from langchain_core.messages import AIMessage
        fallback = AIMessage(content="抱歉，生成回答时遇到服务问题，请稍后重试。")
        return {"messages": [fallback]}


# ============================================================
# Legacy ReAct nodes (保留兼容)
# ============================================================

def llm_node(state: AgentState) -> dict:
    """ReAct LLM 节点"""
    # 对话太长时裁剪一下，防止爆 token
    messages = [SystemMessage(content=REACT_PROMPT)] + trim_messages(
        state["messages"], max_messages=15, keep_system=False
    )
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """ReAct 工具节点"""
    results = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": results}


FORCE_SEARCH_MSG = SystemMessage(
    content="【系统纠正】你刚才没有调用 search_knowledge 工具就直接回答了。"
    "这是课程知识点问题，你必须先调用 search_knowledge 搜索知识库，"
    "然后基于搜索结果回答。请立即调用 search_knowledge 工具。"
)


def force_search_node(state: AgentState) -> dict:
    """守卫节点：LLM 未调工具就回答时，注入提醒消息让其重试"""
    return {"messages": [FORCE_SEARCH_MSG]}


def should_continue(state: AgentState) -> str:
    """ReAct 条件边"""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"

    ai_count = sum(1 for msg in state["messages"] if isinstance(msg, AIMessage))
    has_tool_msg = any(isinstance(msg, ToolMessage) for msg in state["messages"])

    if not has_tool_msg and ai_count <= 1:
        return "force_search"

    return END
