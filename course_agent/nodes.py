"""节点函数

Week 5 升级：Plan-and-Execute 模式
Week 7 升级：自适应规划层（check_results + web_search_fallback）

节点流程：
  planner → [route_after_plan]
              ├→ executor → [check_results]
              │                ├→ synthesizer → END        (结果充足)
              │                └→ web_search_fallback → synthesizer → END  (结果不足，补搜)
              └→ synthesizer → END                        (无需工具的问题)

特点：
- 以用户需求为中心，灵活选择工具，不死板
- 支持知识库问答、联网搜索、刷题训练等多种场景
- 工具参数由 planner 动态决定，executor 直接执行
"""

import json
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import END
from course_agent.state import AgentState
from course_agent.llm import llm
from course_agent.tools import tools, tools_by_name
from course_agent.errors import safe_llm_invoke, safe_tool_invoke, LLMError
from course_agent.memory import trim_messages, compress_history, SUMMARY_MARK
from course_agent.logger import logger

llm_with_tools = llm.bind_tools(tools)

# ============================================================
# System Prompts
# ============================================================

PLANNER_PROMPT = """你是「知识寻回BOT」的规划师。分析用户当前的需求，决定是否需要调用工具，以及调用哪些工具。

【重要概念区分】
- 「知识库」= 课程课件、课本、笔记，用于学习知识点（search_knowledge 搜索）
- 「习题库」= 往年试卷、练习题，用于刷题训练（list_papers / get_paper_questions / grade_answers 操作）
- 用户说"刷题/做题/写题/答题/练习/做卷子/考试/自测/找题/出题目" → 都是要从习题库抽题做，用 list_papers
- 用户说"知识库里的题/题库里的题" → 也是指习题库，用 list_papers
- 用户问"什么是XX/解释XX/XX原理" → 是知识点问题，用 search_knowledge

你有 10 个工具可用：
1. search_knowledge(query) — 搜索课程知识库（计算机网络+操作系统），用于回答知识点问题
2. web_search(query) — 联网搜索，知识库不够或涉及课外内容时使用
3. get_course_info() — 查看知识库统计信息
4. list_papers(course?) — 列出习题库中的试卷，可选按课程筛选（操作系统/计算机网络）
5. get_paper_questions(paper_source, group_index?) — 从某份试卷里取**一组**题
   注意：一次只返回一组（选择题/填空题/判断题 5 题一组，综合题 1 题一组），
   **不是整卷**。继续做下一组时，必须把上一轮工具结果里的 next_group_index 传进 group_index；
   不传的话只会反复给第 1 组。
6. explain_question(paper_source, no?) — 取某道题的**题面原文 + 参考答案 + 评分要点**
   学生说"不会 / 没思路 / 讲讲这道题 / 为什么选 X"，或者要讲某道错题时必须调它。
   出题接口刻意不返回题面和答案，所以**没有它就讲不了题**，不要凭图片瞎猜。
7. grade_answers(paper_source, user_answer_text) — 批量批改学生答案
8. prepare_custom_quiz(topic, course?, qtype?, count?, difficulty?) — 为「临时出题」取素材
   学生说"不想做这套了 / 另外出几道题 / 针对XX多练几道" → 用这个，**不要**去翻题库。
   它不返回题目，只返回素材，题目由你自己写（并在题前说明是自拟题）。
9. get_practice_stats(paper_source) — 获取试卷练习次数统计
10. record_practice(paper_source) — 记录完成一次试卷练习

【强制规则 - 必须严格遵守】
1. 用户说"刷题/做题/写题/答题/练习/做卷子/考试/自测/找题/出题目/题库/卷子/试卷" → 必须调用 list_papers，绝对不能用 search_knowledge
2. 用户选了某套试卷（说"第X套"或试卷名）→ 调用 get_paper_questions；
   说"继续/下一组"→ 也要调 get_paper_questions，并把上一组的 next_group_index 传进去
3. 用户提交答案（字母串如ABCDE、或1.A 2.B格式）→ 调用 grade_answers
4. 用户说"不会/没思路/讲讲这道题/为什么" → 调用 explain_question 取题面和答案
5. 用户说"不想做这套了/另外出几道题/额外练几道" → 调用 prepare_custom_quiz，不要用题库工具
6. 用户问知识点问题 → 用 search_knowledge
7. 闲聊/问候/不需要工具 → 直接回答

你必须返回JSON格式：
{"need_tools": true/false, "steps": [{"tool": "工具名", "args": {"参数名": "参数值"}}, ...]}

注意：
- steps 最多 3 个步骤
- args 是工具的参数字典，根据工具定义填写正确的参数名
- 如果不需要工具，steps 为空数组，need_tools 为 false
- 只规划当前这一轮需要做的事，不要规划未来多轮的流程
- **题目和图片只能来自工具返回**：绝不允许规划"自己出题"或"编一道题"，
  也不允许编造图片链接（出题只能通过 list_papers / get_paper_questions 从题库取）"""

SYNTHESIZER_PROMPT = """你是「知识寻回BOT」，一位耐心、专业的课程学习助手。

你的能力：
- 答疑解惑 — 基于知识库内容回答学生问题
- 教学讲解 — 分章节、分节给学生讲课
- 刷题训练 — 从习题库抽题让学生做，批改并讲解
- 复习总结 — 帮学生梳理知识点、回顾错题

交互原则：
- 自然、像真人老师一样对话，不要太机械
- 基于工具返回的结果回答，不要编造内容
- 学生问知识就讲知识，学生想刷题就出题，学生提交答案就批改
- 重要知识点用加粗或列表突出显示
- 讲题时先说答案，再讲思路，最后关联知识点

刷题相关规则：
- 出题展示时，只显示「题号 + 知识点 + 图片」，不要输出题目文字内容
- 题目格式：Q1 [进程调度] 然后贴图片（![题目](图片URL)）
- 选择题 5 道一组出，其他题型 1 道一组出
- 批改结果来自 grade_answers 工具，不要自己判断对错
- 学生做完一组后，给出批改结果和讲解，然后问是否继续下一组

【最重要的一条：不许编题】
- **题目和图片只能来自工具返回**。工具没返回题目时，就如实说"我没能取到题目"，
  绝不允许自己编一道题、也不允许编造图片链接（尤其不许用 example.com 之类的占位地址）。
- 出题工具只给「题号 + 知识点 + 图片」，**题干和答案是故意不给的**（防泄题）。
  学生要讲解时用 explain_question 取回来，不要凭知识点名字去猜题目内容。
- 讲题时如果手里只有知识点名字、没有题面，就说"我先把题面取出来"并调 explain_question，
  而不是围绕那个知识点泛泛而谈。

输出格式：
- 用 Markdown 格式
- 题目图片用 ![题目](图片URL) 格式
- 保持自然流畅，不要输出 JSON 或代码块
- **不要把工具调用本身写进回复**（比如 `get_paper_questions {"paper_source": ...}` 这种）"""

CHITCHAT_PROMPT = """你是「知识寻回BOT」，一位课程学习助手，擅长操作系统和计算机网络。

友好回应用户，可以引导用户提问课程问题或开始刷题。
如果用户的问题不涉及课程知识，尽力帮助用户，但不要编造课程内容。"""

REACT_PROMPT = """你是「知识寻回BOT」，一位耐心、专业的课程学习助手，擅长操作系统和计算机网络两门课程。

你的能力：
1. 教学讲解 — 根据知识库内容，分章节、分节给学生讲课
2. 刷题训练 — 从习题库中抽题让学生做，批改并讲解
3. 答疑解惑 — 学生提问时，检索知识库后给出清晰解答
4. 复习总结 — 帮学生梳理知识点、回顾错题

可用工具：
- search_knowledge(query) — 搜索课程知识库，优先使用
- web_search(query) — 联网搜索，知识库不够时补充
- get_course_info() — 查看知识库信息
- list_papers(course?) — 列出习题库试卷
- get_paper_questions(paper_source) — 获取试卷题目
- grade_answers(paper_source, user_answer_text) — 批改答案
- get_practice_stats(paper_source) — 练习统计
- record_practice(paper_source) — 记录练习

交互原则：
- 自然、像真人老师一样对话，不要太机械
- 学生说"学XX"就检索知识库开始教学
- 学生说"刷题/做题"就调用 list_papers 列试卷
- 学生提交答案后，调用 grade_answers 批改，不要自己判
- 出题只显示题号+知识点+图片，不要输出题目文字
- 学生想换内容、问问题、闲聊时，灵活响应，不要死板
- 不能编造答案，所有题目答案必须来自工具

输出格式：
- 用 Markdown 格式
- 题目图片用 ![题目](图片URL) 格式
- 保持自然流畅"""


# ============================================================
# Planner Node
# ============================================================

def planner_node(state: AgentState) -> dict:
    """规划器节点：分析用户需求，决定用哪些工具"""
    # 历史口径必须和 synthesizer 一致（都排除系统提示和工具结果），
    # 这样 compress_history 的摘要能在两个节点间命中缓存，不会一轮调两次摘要 LLM。
    # 之前 planner 用的是含 ToolMessage 的全量窗口，既不共享缓存，
    # 也看不到摘要 —— 于是"刚才第一个问题问的是什么"这类指代型追问，
    # 规划阶段就丢了上下文，只会拿上一轮的话题去检索。
    history_msgs = [
        m for m in state["messages"]
        if not isinstance(m, SystemMessage) and not isinstance(m, ToolMessage)
    ]
    summary_text, recent_msgs = compress_history(history_msgs, llm)

    history_lines = []
    if summary_text:
        history_lines.append(f"（更早的对话摘要）{summary_text}")
    for m in recent_msgs[-4:]:
        if isinstance(m, HumanMessage):
            history_lines.append(f"用户: {m.content}")
        elif isinstance(m, AIMessage):
            history_lines.append(f"助手: {m.content[:200]}")
    history_text = "\n".join(history_lines) if history_lines else ""

    user_msg = state["messages"][-1]
    user_content = user_msg.content if hasattr(user_msg, "content") else str(user_msg)

    prompt = PLANNER_PROMPT
    if history_text:
        prompt += f"\n\n--- 对话上下文 ---\n{history_text}"

    # 把跨轮会话游标显式交给 planner。
    # 不加这段的话，学生说"继续下一组"时 planner 手里既没有卷子也没有组号
    # （上一轮的工具结果被上面的 history_msgs 过滤掉了），实测会瞎猜卷子、甚至编题。
    drill = state.get("session") or {}
    if drill.get("paper"):
        prog = [f"卷子：{drill['paper']}"]
        if drill.get("type"):
            prog.append(f"题型：{drill['type']}")
        if drill.get("group_index"):
            prog.append(f"刚出到第 {drill['group_index']} 组")
        if drill.get("finished"):
            prog.append("这份卷子已经出完了")
        elif drill.get("next_group_index"):
            prog.append(f"**下一组要传 group_index={drill['next_group_index']}**")
        if drill.get("qids"):
            prog.append("刚出的题 qids：" + ",".join(drill["qids"]))
        prompt += ("\n\n--- 当前刷题进度（来自上一轮，务必据此填参数）---\n"
                   + "\n".join(f"- {x}" for x in prog)
                   + "\n学生说「继续/下一组」时，直接用上面的卷子和 group_index 调 "
                     "get_paper_questions；学生提交答案时用上面的 qids 调 grade_answers。")

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=user_content),
    ]
    try:
        response = safe_llm_invoke(llm, messages, context="planner")
        raw = response.content.strip()
    except LLMError:
        return {"plan": [{"tool": "search_knowledge", "args": {"query": user_content}}]}

    # 解析 JSON 计划
    if "```json" in raw:
        raw = raw.split("```json")[1].split("```")[0].strip()
    elif "```" in raw:
        raw = raw.split("```")[1].split("```")[0].strip()

    try:
        plan_data = json.loads(raw)
        # 模型经常返回"合法 JSON 但结构不对"的东西（顶层是数组、
        # steps 是对象、steps 里混进字符串）。原来只 catch JSONDecodeError，
        # 这些情况会抛 AttributeError 冒到路由层，整个请求 500。
        if not isinstance(plan_data, dict):
            raise ValueError(f"计划顶层不是对象: {type(plan_data).__name__}")
        steps = plan_data.get("steps", [])
        if not isinstance(steps, list):
            raise ValueError(f"steps 不是数组: {type(steps).__name__}")
        # 统一格式：每个 step 必须有 tool 和 args
        normalized = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool_name = step.get("tool", "")
            if not tool_name:
                continue
            args = step.get("args", {})
            if not isinstance(args, dict):
                args = {}
            # 兼容旧格式：如果有 query 字段但没有 args
            if not args and step.get("query"):
                args = {"query": step["query"]}
            normalized.append({"tool": tool_name, "args": args})
        # steps 有内容但一条可用都没有（例如里面全是字符串）：
        # 不能当成"不需要工具"，否则会绕过检索直接作答
        if not normalized and steps:
            raise ValueError("steps 里没有可用的工具调用")
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("planner 返回的计划无法解析，回退默认计划: %s", e)
        normalized = [{"tool": "search_knowledge", "args": {"query": user_content}}]

    return {"plan": normalized}


def route_after_plan(state: AgentState) -> str:
    """条件边：有计划步骤 → executor，无 → 直接回答"""
    if state.get("plan") and len(state["plan"]) > 0:
        return "executor"
    return "synthesizer"


# ============================================================
# Executor Node
# ============================================================

def executor_node(state: AgentState) -> dict:
    """执行器节点：按计划批量执行所有工具调用

    返回时**必须同时给出声明这些 tool_call 的 AIMessage**。
    原来只往 messages 里塞 ToolMessage(tool_call_id="plan_0"...)，
    而 state 里没有任何 assistant 消息声明过 plan_0 —— 这是一条"孤儿 tool 消息"，
    违反 OpenAI 消息协议：一旦这份 state 被整段送给模型（换模式续聊、
    或者将来任何调用方直接 dump state["messages"]），接口会 400，
    再被 safe_llm_invoke 吞成兜底套话，表现为"模型突然只会说套话"。
    """
    results = []
    tool_calls = []

    for i, step in enumerate(state["plan"]):
        tool_name = step.get("tool", "")
        args = step.get("args", {})
        call_id = f"plan_{i}"
        tool_calls.append({"id": call_id, "name": tool_name, "args": args})

        if tool_name not in tools_by_name:
            results.append(ToolMessage(
                content=f"未知工具: {tool_name}",
                tool_call_id=call_id,
                name=tool_name,
            ))
            continue

        tool = tools_by_name[tool_name]
        try:
            result = safe_tool_invoke(tool, args, tool_name=tool_name)
        except Exception as e:
            result = f"工具调用失败: {str(e)[:100]}"

        results.append(ToolMessage(
            content=str(result),
            tool_call_id=call_id,
            name=tool_name,
        ))

    if not results:
        return {"messages": []}

    # 内容留成可读的"我调用了哪些工具"：它对 planner/synthesizer 的历史是有用上下文，
    # 前端则不会看到它（sse_content_event 会跳过带 tool_calls 的消息）。
    declaration = AIMessage(
        content="（调用了工具：" + "、".join(tc["name"] for tc in tool_calls) + "）",
        tool_calls=tool_calls,
    )
    out = {"messages": [declaration] + results}

    # 从工具结果里抽出"刷题进度"存进 state —— 这是 Plan 图里游标唯一的落脚点
    # （planner 会把 ToolMessage 从历史里剔掉，所以游标没法"在带内"传递）。
    new_session = _extract_drill_session(state.get("session") or {},
                                        state.get("plan") or [], results)
    if new_session is not None:
        out["session"] = new_session
    return out


def _extract_drill_session(current: dict, plan: list, results: list):
    """看这一轮的工具结果里有没有刷题进度，有就更新游标

    只认我们自己生成的这几个字段（`get_paper_questions` 的 JSON 输出），
    解析不出来就原样返回 None（不动游标）。
    """
    import json as _json

    updated = None
    for step, msg in zip(plan or [], results or []):
        if step.get("tool") != "get_paper_questions":
            continue
        try:
            data = _json.loads(msg.content)
        except (TypeError, ValueError, AttributeError):
            continue
        if not isinstance(data, dict) or "paper" not in data:
            continue
        updated = dict(current or {})
        updated.update({
            "paper": data.get("paper"),
            "type": data.get("type") or step.get("args", {}).get("qtype"),
            "group_index": data.get("group_index"),
            "next_group_index": data.get("next_group_index"),
            "qids": list(data.get("qids") or []),
            "finished": data.get("next_group_index") is None and not data.get("has_more", True),
        })
    return updated


# ============================================================
# Check Results (自适应规划层)
# ============================================================

_INSUFFICIENT_MARKERS = [
    "知识库中未找到相关内容",
    "知识库中相关内容较少",
    "没有找到符合条件的试卷",
    "找不到试卷",
    "习题库为空",
    "习题库不存在",
]


def _current_turn_tool_messages(state: AgentState) -> list:
    """只取"本轮"的工具结果

    checkpointer 会把历史累积在 state["messages"] 里，
    原来 route_after_executor / synthesizer 遍历全部消息，
    于是：① 上一轮的"知识库未找到"标记会让本轮凭空多跑一次联网搜索；
    ② 用户这轮只是闲聊，synthesizer 却拿上一轮的检索内容当本轮结果作答。
    以最后一条 HumanMessage 为界切分即可。
    """
    msgs = state["messages"]
    last_human = -1
    for i, m in enumerate(msgs):
        if isinstance(m, HumanMessage):
            last_human = i
    if last_human < 0:
        return [m for m in msgs if isinstance(m, ToolMessage)]
    return [m for m in msgs[last_human + 1:] if isinstance(m, ToolMessage)]


def route_after_executor(state: AgentState) -> str:
    """条件边：检查检索结果是否充足，不足时补搜

    只对知识库问答场景做补搜判断，刷题场景不需要。
    """
    # 如果 planner 已经规划了 web_search，不需要再补
    has_web_search = any(
        step.get("tool") == "web_search" for step in state.get("plan", [])
    )
    if has_web_search:
        return "synthesizer"

    # 如果调用了刷题相关工具，不需要补搜。
    # explain_question 必须在名单里：它是讲题路径上的工具，结果本来就短，
    # 不列进来的话 route_after_executor 会把它当成"检索结果不足"而多跑一次联网搜索。
    has_paper_tools = any(
        step.get("tool") in ("list_papers", "get_paper_questions", "grade_answers",
                             "explain_question", "get_practice_stats", "record_practice")
        for step in state.get("plan", [])
    )
    if has_paper_tools:
        return "synthesizer"

    # 收集本轮 ToolMessage 内容（"知识库有多少条"这类元信息不参与"够不够"的判断，
    # 否则一句"知识库当前有 N 条分块"这种短回复会被判成结果不足）
    tool_contents = []
    for msg in _current_turn_tool_messages(state):
        content = msg.content
        if "自适应补搜" in content or "网络搜索" in content:
            continue
        if "条文档分块" in content or "条分块" in content:
            continue
        tool_contents.append(content)

    if not tool_contents:
        return "web_search_fallback"

    combined = "\n".join(tool_contents)

    # 检查结果不足标记
    for marker in _INSUFFICIENT_MARKERS:
        if marker in combined:
            return "web_search_fallback"

    # 文本太短
    if len(combined) < 100:
        return "web_search_fallback"

    return "synthesizer"


def web_search_fallback_node(state: AgentState) -> dict:
    """自适应补搜节点：知识库结果不足时自动联网搜索"""
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
    """综合器节点：基于工具结果 + 历史对话生成最终回答"""
    human_msgs = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    user_msg = human_msgs[-1] if human_msgs else state["messages"][-1]
    user_content = user_msg.content if hasattr(user_msg, "content") else str(user_msg)

    # 收集本轮工具结果（只看最后一条用户消息之后，避免把上一轮的检索内容
    # 当成这一轮的结果 —— 用户说"谢谢"时那种情况最明显）
    tool_results = [msg.content for msg in _current_turn_tool_messages(state)]

    # 提取历史对话（排除系统消息和工具结果）
    history_msgs = [
        m for m in state["messages"]
        if not isinstance(m, SystemMessage)
        and not isinstance(m, ToolMessage)
    ]

    # 历史压缩：超过阈值时用 LLM 把老对话总结成摘要，否则只做滑动窗口。
    # 摘要文本拼进 system prompt（而不是作为第二条 system 消息塞进去）——
    # 多段 system 消息在部分模型上行为不一致，合并成一条最稳。
    summary_text, history_msgs = compress_history(history_msgs, llm)
    base_prompt = SYNTHESIZER_PROMPT if tool_results else CHITCHAT_PROMPT
    if summary_text:
        base_prompt = f"{base_prompt}\n\n{SUMMARY_MARK}\n{summary_text}"

    if tool_results:
        context = "\n\n---\n\n".join(tool_results[-5:])
        messages = [
            SystemMessage(content=base_prompt),
        ]
        if len(history_msgs) > 1:
            messages.extend(history_msgs[:-1])
        messages.append(HumanMessage(content=f"用户: {user_content}\n\n工具返回结果:\n{context}"))
    else:
        messages = [
            SystemMessage(content=base_prompt),
        ]
        if len(history_msgs) > 1:
            messages.extend(history_msgs[:-1])
        messages.append(HumanMessage(content=user_content))

    try:
        response = safe_llm_invoke(llm, messages, context="synthesizer")
        return {"messages": [response]}
    except LLMError:
        fallback = AIMessage(content="抱歉，生成回答时遇到服务问题，请稍后重试。")
        return {"messages": [fallback]}


# ============================================================
# ReAct 节点
# ============================================================

# ReAct 工具环的业务级轮数上限。
#
# 原来 should_continue 只看"最后一条消息有没有 tool_calls"、tool_node 也不计数，
# 也就是说**循环次数完全由模型决定**，唯一的上限是 LangGraph 默认的
# recursion_limit=10007 —— 实测模型每轮都要求调工具时，抛错前真实发生
# LLM 调用 5004 次、工具调用 5003 次（线上每次都是一次 HTTP，超时 120s）。
# 手写那条循环有自己的 _MAX_TOOL_ROUNDS = 6，这里对齐。
MAX_TOOL_ROUNDS = 6


def llm_node(state: AgentState) -> dict:
    """ReAct LLM 节点

    轮数用尽时改用**不带工具**的 llm 再答一次：这样模型必须给最终答案，
    而不是让循环停在一个 tool 消息上、把工具原文当回复。
    """
    rounds = int(state.get("tool_rounds", 0) or 0)
    exhausted = rounds >= MAX_TOOL_ROUNDS

    messages = [SystemMessage(content=REACT_PROMPT)] + trim_messages(
        state["messages"], max_messages=15, keep_system=False
    )
    if exhausted:
        logger.warning("ReAct 工具轮数已达上限 %d，改为直接作答", MAX_TOOL_ROUNDS)
        messages.append(HumanMessage(
            content="（工具调用次数已达上限，请基于已经拿到的信息直接作答，不要再调用工具。）"
        ))

    try:
        response = safe_llm_invoke(llm if exhausted else llm_with_tools,
                                   messages, context="react_llm")
        return {"messages": [response]}
    except LLMError:
        return {"messages": [AIMessage(content="抱歉，处理请求时遇到服务问题，请稍后重试。")]}


def tool_node(state: AgentState) -> dict:
    """ReAct 工具节点（同时累加工具轮数）"""
    results = []
    last_message = state["messages"][-1]

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool = tools_by_name.get(tool_name)
        if tool is None:
            results.append(ToolMessage(
                content=f"未知工具: {tool_name}",
                tool_call_id=tool_call["id"],
                name=tool_name,
            ))
            continue
        result = safe_tool_invoke(tool, tool_call["args"], tool_name=tool_name)
        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"], name=tool_name)
        )

    return {
        "messages": results,
        "tool_rounds": int(state.get("tool_rounds", 0) or 0) + 1,
    }


def should_continue(state: AgentState) -> str:
    """ReAct 条件边：有工具调用就去执行，没有就结束

    轮数用尽是硬性收敛条件：即使模型还在要求调工具也直接结束，
    避免只靠框架默认的 recursion_limit 兜底（那意味着几千次调用）。
    """
    rounds = int(state.get("tool_rounds", 0) or 0)
    if rounds >= MAX_TOOL_ROUNDS:
        return END

    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END
