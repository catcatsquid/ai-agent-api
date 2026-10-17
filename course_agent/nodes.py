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
import re
from langchain_core.messages import SystemMessage, ToolMessage, AIMessage, HumanMessage
from langgraph.graph import END
from course_agent.state import AgentState
from course_agent.llm import llm
from course_agent.tools import tools, tools_by_name
from course_agent.errors import safe_llm_invoke, safe_tool_invoke, LLMError
from course_agent.memory import trim_messages, compress_history, SUMMARY_MARK
from course_agent.logger import logger
from course_agent import quiz_service
from course_agent import progress

llm_with_tools = llm.bind_tools(tools)

# ============================================================
# 跨轮会话游标（AgentState["session"]）
# ============================================================
#
# 游标只存在 `AgentState["session"]` 里，而 state 由 checkpointer 保管 ——
# 1.2.0 起是**落盘的** SqliteSaver（见 graph.py），所以它天然活得过进程重启。
#
# 历史：1.0.x 时 checkpointer 还是 MemorySaver（进程内存），服务一重启游标就没了，
# 学生说"继续下一组"只能听到"找不到题目"。当时给游标单独做了一份落盘副本
# （runtime/plan_cursors.json + _save_cursor/_load_cursor）。那份副本带来了
# "内存 + 磁盘"两个真源，还引出过读写不对称的 bug（1.0.1 修过、1.0.2 补了回归测试）。
# 换成 SqliteSaver 之后 state 整体落盘，**那份副本连同一整套读写函数都删掉了** ——
# 真源只剩 state 一个。


# ============================================================
# 工具调用泄漏成正文的兜底清理
# ============================================================
#
# 实测：synthesizer 手里没有工具，但当它觉得"还差一步"时，
# 会把工具调用**当正文写出来**给学生看：
#     请稍等，我马上为你获取题目。get_paper_questions {"paper_source": "..."}
# 根因是 Plan 图单向、synthesizer 无法再取数据（结构问题，正解是允许一次重规划）。
# 这里先做兜底：把这种片段从回答里摘掉，别让学生看到内部协议。
# 注意**不要**用"已知工具名白名单"来匹配 —— 实测模型会**编造不存在的工具名**
# （学生答 ACDBA 时它输出了 `get_paper_answers {...}`，而白名单里没有这个名字，
#  于是整段泄漏给了学生）。改成通用的"标识符 + JSON 对象"：
# 蛇形命名的标识符紧跟一个 JSON 对象，正常中文回复里不会长这样。
_TOOL_CALL_TEXT_RE = re.compile(
    r"(?:\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,4}\s*\{[^{}]*\})"
    r"|(?:\b[a-z][a-z0-9]*(?:_[a-z0-9]+){1,4}\(\s*\{[^{}]*\}\s*\))")


def strip_tool_call_text(text: str) -> str:
    """摘掉回答里泄漏出来的工具调用片段"""
    if not text:
        return text
    cleaned = _TOOL_CALL_TEXT_RE.sub("", text)
    # 摘掉后可能留下多余空行与结尾的悬空提示
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


# ============================================================
# 会话状态注入 + 两个守卫（从手写助教那侧移植过来）
# ============================================================
#
# 起因：同题对比 BOT 与手写助教时发现 BOT 在"学习/复习"流程上明显吃亏，而原因不是
# 范式，是**状态注入放的位置不同** —— 助教把状态注进"写回答的模型"的 system prompt，
# 而 BOT 只注进了 planner（写回答的 synthesizer 根本看不到）。实测后果：
#   · 学生交小测答案 → BOT 说"请提供具体的问题和选项"（它手里没有书签，也没题面）
#   · 讲了课但不说"本节结束"、不出小测（那是写给 synthesizer 的契约）
#
# 另外把助教的两个守卫也补过来（这两条 BOT 一直没有）：
#   · 学生说"没看到题目" → 确定性重发（不许只口头道歉）
#   · 历史/工具结果的裁剪与压缩（BOT 的 MemorySaver 原来**没有任何上限**，审查文档 F2）

_MISSED_QUESTIONS = re.compile(
    r"没看到|没有题|题目呢|题呢|没给|没发|没显示|没出来|漏了|再发一遍|重新发|再看一遍|重复一遍"
)
_BOT_KEEP_TOOL_TURNS = 2          # 只保留最近 N 轮的工具结果原文
_BOT_MAX_MESSAGES = 40            # 与手写助教同口径


def _missed_questions(text: str) -> bool:
    """学生是不是在说「题目没给我」"""
    return bool(_MISSED_QUESTIONS.search(str(text or "")))


_QUIZ_ANSWER_HINT = re.compile(
    r"(\d\s*[.、)]\s*[A-Ha-h对错√×]|(?<![A-Za-z])[A-H]{3,}(?![A-Za-z])|我答|答案[:：])"
)


_STUDY_REQ_RE = re.compile(r"复习|预习|我要学|带我|过一遍|串一遍|讲讲|讲解|梳理|学一下")
_CONT_RE = re.compile(r"继续|接着|下一节|往下")


def _looks_like_study_request(text: str) -> bool:
    """学生是不是在要求「系统讲这门课/这一章」或「接着讲下去」

    包含「继续/接着/下一节」这类**延续信号** —— 实测「继续讲」里没有
    「讲讲/讲解」这种词，只按请求词判断会漏掉它（而它恰恰是最常见的一句）。
    漏判的代价：不补 read_section → 模型凭记忆接着讲。
    """
    t = str(text or "")
    if not (_STUDY_REQ_RE.search(t) or _CONT_RE.search(t)):
        return False
    # 别再拦「学生交答案」那一轮（那轮由另一条守卫处理成不调工具）
    return not _looks_like_quiz_answer(t)


def _course_from_text(text: str) -> str:
    """从学生这句话里认出课程名（认不出就返回空，**不乱猜**）"""
    t = str(text or "")
    for c in ("操作系统", "计算机网络"):
        if c in t:
            return c
    return ""


def _looks_like_quiz_answer(text: str) -> bool:
    """学生像在提交一份答案（"1.B 2.D"、"ACDBE"、"我答一下…"）

    用于"讲课/复习中交自拟小测答案"这个场景 —— 那种答案题库里没有，
    只能由模型自己批；实测模型会习惯性去调 grade_answers，
    然后反问学生"你在做哪份卷子"，把学习流程硬拽回刷题流程。
    """
    return bool(_QUIZ_ANSWER_HINT.search(str(text or "")))


def _session_block(session: dict) -> str:
    """把会话状态整理成"写给**写回答的模型**"的一段提示

    这段必须同时给 synthesizer 和 chitchat 用 —— 因为学生交小测答案那一轮
    **一个工具都不会调**（走的是 CHITCHAT_PROMPT），而"小测由你自己批"这条
    恰恰要在那一轮生效。
    """
    if not session:
        return ""
    bits = []
    study = session.get("study") or {}
    if study.get("course"):
        line = (f"【当前学习位置】{study.get('mode', '学习')} · {study['course']}"
                + (f" · {study['chapter']}" if study.get("chapter") else "")
                + f" · 上一节《{study.get('section', '')}》"
                  f"({study.get('status', '讲解中')})")
        if study.get("last_source"):
            line += f" · 刚读到《{study['last_source']}》的 {study.get('last_read', '')}"
            line += (f"，继续讲要用 offset={study['next_offset']}"
                     if study.get("next_offset") is not None else "（已读完）")
        bits.append(line + "\n"
                    "· **本节的测验题是你自己出的，所以由你自己批**：逐题给 ✓/✗ 与讲解，"
                    "说清**错在哪、为什么错、正确答案是什么**，错的知识点要重讲一遍。\n"
                    "· **不要**调 grade_answers / get_paper_questions —— 那是题库真题的"
                    "批改通道；学生答的是你刚出的小测，题库里没有它。\n"
                    "· 学生说「继续讲」＝接着读**同一份**课件；说「下一节」＝进下一份。\n"
                    "· 学生说「没听懂 / 换个讲法」→ 换角度重讲（类比、画流程、举具体例子），"
                    "**不要**让学生报题号、也不要复读上一轮的原话。\n"
                    "· 讲完这一节要明确说「本节结束」，然后出覆盖本节核心考点的小测题；"
                    "**不要**提前讲后面的内容。")
    drill = session.get("paper")
    if drill:
        bits.append(f"【当前试卷】{drill}"
                    + (f" · 刚出到第 {session['group_index']} 组"
                       if session.get("group_index") else "")
                    + (f" · 下一组传 group_index={session['next_group_index']}"
                       if session.get("next_group_index") else ""))
    return ("\n\n--- 当前会话状态（务必据此回答）---\n" + "\n".join(bits)) if bits else ""


# 什么时候把"全局进度 + 开场建议"塞进提示词：
#   · 新会话（还没有任何会话状态）—— 这就是这个功能的本意：开窗就能接着上次
#   · 学生明确要学/要复习/问进度/说继续 —— 这时不注入，它就得反问"你学到哪了"
_PROGRESS_INTENT_RE = re.compile(
    r"学习|复习|学到哪|学到第|进度|继续|接着|上次|建议|开始学|想学|学过|错题|重做")

# 「只说想学/想复习」的开场（还没选模式、也没说具体章节）
_STUDY_KICKOFF_RE = re.compile(r"(想|要|准备|开始)\s*(学习|复习)|^学(习|一下)")
# 说了具体目标就不算开场：这类应该照常规划（直接讲那一节 / 出那几道题）
_STUDY_TARGET_RE = re.compile(
    r"题|卷|章|节|课件|课本|笔记|继续|接着|讲|解释|为什么|是什么|怎么|"
    r"第[一二三四五六七八九十\d]|死锁|调度|进程|线程|内存|传输层|网络层|握手")


def _is_study_kickoff(text: str) -> bool:
    """学生是不是"刚开口说想学/想复习"，还没说学什么、怎么学

    这种轮次**不该直接开讲**（实测：它一上来就 read_section 讲了一整节），
    而应该先把已知进度摆出来、让他在四种模式里选一条。
    """
    t = str(text or "").strip()
    if not t or len(t) > 20:
        return False
    return bool(_STUDY_KICKOFF_RE.search(t) and not _STUDY_TARGET_RE.search(t))


def _progress_block(user_content: str, session: dict) -> str:
    """取"已知进度 + 四选一建议"的提示块（无进度时返回空串）

    只在**相关**的轮次注入。原来还允许"新会话一律注入"，实测立刻踩坑：
    新窗口里问一句**概念题**（"死锁的四个必要条件是什么"），进度块把
    "卷子《…》做到第 1 组"也带进去了 → planner 以为要刷题，
    真的调了 get_paper_questions + grade_answers（评测用例
    `knowledge_answer_streams` 抓到的回归）。

    所以现在**只按意图**注入：说想学/想复习/问进度/要续做/要重做错题，
    或者开头那句"我想学习"（`_is_study_kickoff`）。空会话本身不是理由。
    """
    try:
        text = str(user_content or "")
        if not (_PROGRESS_INTENT_RE.search(text) or _is_study_kickoff(text)):
            return ""
        return progress.brief()
    except Exception:                                   # noqa: BLE001
        logger.warning("读全局学习进度失败（不影响本轮）", exc_info=True)
        return ""


# ============================================================
# 学习/复习相关工具（先定义给 planner 用）
# ============================================================

# ============================================================
# 行为规范（B 类：话术级行为，**不进硬判据**，只做观察项）
#
# 为什么写成模块级常量而不是散在提示词里：
#   1. 评测要能**从生产代码 import**（判据与规范不许各写一份，
#      同一事实写两遍必然有一处过期 —— 这条在 CHANGELOG 里写过）；
#   2. 三个提示词（planner / synthesizer / react）要复用同一套说法，
#      各抄一份必然走形。
# 划分标准见 `ai-agent-notes/决策记录.md`：需要人读才知道对错的行为 → 写在这里。
# ============================================================

QUIZ_BEHAVIOR_RULES = """【答题与批改的行为规范】
1. 学生**只答了部分题**（如"1.C 2.B 3.D"）→ 只批这几道，并**提醒还剩哪几道没答**
   （题号从本组 qids 里数出来，别凭印象）。
2. 学生**部分作答 + 其余说"不会/教教我"** → 批已答的，再**把那几道不会的答案给出来并讲解**。
3. 学生**直接要答案**（"答案是什么"）→ 调 **get_answers**，把返回的答案**原样转述**，
   不要改写字母、**不要顺带批改** —— **编一个答案比不给更糟**。
4. 学生说的**明显不是作答**（答非所问、突然聊别的）→ 先轻轻指出"这看起来不像答案"，
   用聊天方式接住他，再引回做题；**这一轮不要调 grade_answers**（别浪费一次批改）。
5. 一题一题答（多轮）→ 每轮**只批新增的那道**，不要每轮重批一遍。
6. 整套卷子做完 → **不报总正确率**；调 record_practice 记一次，然后问
   「重做错题，还是下一套？」—— 学生选重做就调 **practice_wrong**（重出原题）。
7. 列完试卷清单后，顺带问一句「还是给你出几道题练手？」——
   有些学生其实不想做整套卷子。**按知识点**的练习（"练几道死锁的题"）用
   **practice_by_topic**，不要拉试卷清单。
8. 自己出的题**必须自带答案**：题目 + 标准答案 + 评分要点一起给出，批改时以你给的答案为依据。"""

STUDY_BEHAVIOR_RULES = """【学习/复习的行为规范】
1. 学生中途岔开提了个问题 → **先把他的问题答完**，然后主动说一句"我们接着说刚才那一节"，
   把它拉回主线（不要答完就停在岔开的话题上）。
2. 讲课时插播做题 → 做完题**回到原来的学习主线**，别把学习流程丢在刷题上。
3. 一整章讲完 → 主动出**覆盖本章考点的小测**；批改后问「下一章，还是先休息？」。
4. 学生连续闲聊到第 3 轮还没进入学习 → **轻量引导一次**（"要不要开始学点什么？"）；
   他仍不学就**不再推**，正常陪聊。
5. 任何时刻学生说「不学了 / 算了 / 换个话题」→ **无条件清空当前任务态**，回到自由对话；
   不要追问"确定吗"、也不要继续讲。
6. 有多个"没做完"的断点（多套卷子/多门课）时，「继续」有歧义 → **把断点列出来让他选**，
   不要自己挑一个。"""


PLANNER_PROMPT = """你是「知识寻回BOT」的规划师。分析用户当前的需求，决定是否需要调用工具，以及调用哪些工具。

【重要概念区分】
- 「知识库」= 课程课件、课本、笔记，用于学习知识点（search_knowledge 搜索）
- 「习题库」= 往年试卷、练习题，用于刷题训练（list_papers / get_paper_questions / grade_answers 操作）
- 用户说"刷题/做题/写题/答题/练习/做卷子/考试/自测/找题/出题目" → 都是要从习题库抽题做，用 list_papers
- 用户说"知识库里的题/题库里的题" → 也是指习题库，用 list_papers
- 用户问"什么是XX/解释XX/XX原理" → 是知识点问题，用 search_knowledge
- 用户说"复习/学习/讲解/梳理/带我过一遍/讲讲这门课/开始第一天的复习" →
  这是**学知识点**，先 `list_course_sections` 拿到该课的资料地图，
  再用 `read_section`（系统讲一节）或 `search_knowledge(course=…)`（答具体问题）取资料。
  **不要**用 list_papers / get_paper_questions —— 那是刷题，不是复习。
  （真实踩过：学生说"想复习操作系统"，结果被当成了刷题，直接甩卷子清单；
   说"开始第一天的复习"，直接被塞了一组题。学生要的是讲解，不是做题。）
- 用户说"我要学/我要复习**某一章**"（如"我要学死锁那一章"）→ 同样先
  `list_course_sections`，找到对应课件后用 `read_section` 读它，**一次只讲一节**
- 用户说"下一节/继续"且上一轮在学课程 → 用 record_study_position 的书签定位，
  继续讲**下一节**（不要重新列章节清单，也不要跳着讲）

你有 17 个工具可用：
1. search_knowledge(query, course?, source?) — 搜索课程知识库，用于回答知识点问题
   **讲解课程内容时务必带 course**（知道具体哪份课件时带 source），这样检索才不会串课
2. list_course_sections(course?) — 列出某门课**实际有哪些课件/课本/笔记**（带章节名与内容量）
   学生说"我要复习/我要学 某门课（或某个章节）"时，**先调它**拿到真实的资料地图，
   再决定怎么分节 —— 不要凭记忆编章节（如"第一天：操作系统概述"）
3. read_section(course, chapter?, section?, offset?, limit?) — 按**课件原始顺序**读某一节
   **系统地讲一节时优先用它**。**不需要你记文件名**：
   · 学生说"**某一章**"（"我要学死锁那一章"）→ 用章名：read_section(course="操作系统", chapter="死锁")
   · 学生说"**第几节/接着讲**"→ 用节号：read_section(course="操作系统", section=2)
   想知道有哪些章，先调 list_course_sections。
   一份课件很长时用 offset 续读；一次读多少由你判断。
   ⚠️ **不要用课本或笔记来"讲课程"** —— 它们的章节顺序不代表本课顺序；
   真去读了，工具会自动换成课件并告诉你。
4. record_study_position(course, section, mode?, chapter?, source?, status?) —
   记下"学到哪一节了"的书签（**不是状态机**，只是个记忆）
   开始讲一节时记一次；本节讲完小测通过或学生卡住时，用 status 更新
5. web_search(query) — 联网搜索，知识库不够或涉及课外内容时使用
6. get_course_info() — 查看知识库统计信息
7. list_papers(course?) — 列出习题库中的试卷，可选按课程筛选（操作系统/计算机网络）
8. get_paper_questions(paper_source, group_index?) — 从某份试卷里取**一组**题
   注意：一次只返回一组（选择题/填空题/判断题 5 题一组，综合题 1 题一组），
   **不是整卷**。继续做下一组时，必须把上一轮工具结果里的 next_group_index 传进 group_index；
   不传的话只会反复给第 1 组。
9. explain_question(paper_source, no?) — 取某道题的**题面原文 + 参考答案 + 评分要点**
   学生说"不会 / 没思路 / 讲讲这道题 / 为什么选 X"，或者要讲某道错题时必须调它。
   出题接口刻意不返回题面和答案，所以**没有它就讲不了题**，不要凭图片瞎猜。
10. grade_answers(paper_source, user_answer_text) — 批量批改学生答案
11. prepare_custom_quiz(topic, course?, qtype?, count?, difficulty?) — 为「临时出题」取素材
    学生说"不想做这套了 / 另外出几道题 / 针对XX多练几道" → 用这个，**不要**去翻题库。
    它不返回题目，只返回素材，题目由你自己写（并在题前说明是自拟题）。
12. get_practice_stats(paper_source) — 获取试卷练习次数统计
13. record_practice(paper_source) — 记录完成一次试卷练习
14. get_learning_progress() — 查学生**跨会话**的学习/刷题进度
    学生问「我学到哪了 / 我哪套卷子没做完 / 上次复习到哪」→ 调它，照着说；
    **不要**反问"你之前学到哪了"（进度已经注入在你的提示里了，没有才调它）
15. get_answers(paper_source, no?, qids?) — **只取答案**（题库原文）
    学生**直接索要答案**时用它；返回的答案**原样转述**，不要改写字母 ——
    **编一个答案比不给更糟**。这一轮**不要**调 grade_answers。
16. practice_wrong(paper_source) — 把这份卷子**做错过的题**取回来重做
    整套卷子做完、学生说"重做错题"时用；**重出原题**（不是让模型编新题）。
17. practice_by_topic(topic, course?, count?, qtype?) — 按**知识点**从题库抽几道题
    学生说"想练几道**死锁**的题"这种**按知识点**的练习时用它（不要拉试卷清单）；
    匹配不到会返回 count=0，那就改用 prepare_custom_quiz 自己出题（自带答案）。

【强制规则 - 必须严格遵守】
1. 用户说"刷题/做题/写题/答题/练习/做卷子/考试/自测/找题/出题目/题库/卷子/试卷" → 必须调用 list_papers，绝对不能用 search_knowledge
2. 用户**指名**某份试卷（"2026卷"/"2025那套"/"2019模拟题"/"第X套"等）→ 必须调用 get_paper_questions
   取题。**不要**只调 list_papers 把清单再念一遍 —— 学生已经选好了。
   说"继续/下一组"→ 也要调 get_paper_questions，并把上一组的 next_group_index 传进去
3. 用户提交答案（字母串如ABCDE、或1.A 2.B格式）→ 调用 grade_answers
4. 用户说"不会/没思路/讲讲这道题/为什么" → 调用 explain_question 取题面和答案
5. 用户说"不想做这套了/另外出几道题/额外练几道" → 调用 prepare_custom_quiz，不要用题库工具
   ⚠️ **也要看"刷题"的宾语**：学生说"刷几道**传输层**的题""练练**死锁**"这类
   **按知识点**的练习（不是整份卷子）→ 先用 **practice_by_topic(topic=…)** 从题库抽题
   （它是跨所有卷子按知识点匹配的），**不要** list_papers —— 题库没有按知识点组卷，
   给菜单等于没满足需求（实测三轮里 BOT 全这么错了）。
   只有 practice_by_topic 返回 `count: 0`（题库里没有匹配的题）时，才改用
   prepare_custom_quiz 自己出题（并**自带标准答案**）。
6. 用户说"复习/学习/讲解/梳理/过一遍/带我学"或问知识点问题 → 调用 search_knowledge，
   并且只讲检索到的课程资料里的内容（资料没覆盖就直说，再用通用知识补充）
   ⚠️ 学生说"**没听懂 / 换个讲法 / 再讲一遍**"且当前在讲课（见下方学习位置）→
   必须调 read_section **把同一份课件的原文再取一次**再讲，换个角度；
   **不要**不调工具就凭记忆复述，也不要让学生自己报题号。
7. 闲聊/问候/不需要工具 → 直接回答

你必须返回JSON格式：
{"need_tools": true/false, "steps": [{"tool": "工具名", "args": {"参数名": "参数值"}}, ...]}

注意：
- steps 最多 3 个步骤
- args 是工具的参数字典，根据工具定义填写正确的参数名
- 如果不需要工具，steps 为空数组，need_tools 为 false
- 只规划当前这一轮需要做的事，不要规划未来多轮的流程
- **题目和图片只能来自工具返回**：绝不允许规划"自己出题"或"编一道题"，
  也不允许编造图片链接（出题只能通过 list_papers / get_paper_questions 从题库取）

""" + QUIZ_BEHAVIOR_RULES + "\n\n" + STUDY_BEHAVIOR_RULES

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

【学习/复习模式】学生说"我要学/我要复习某门课（或某一章）"时，是**带你系统过一遍**，
不是甩一堆知识点让他自己看。下面是要把握的原则，不是必须逐条执行的清单 ——
**节奏和粒度由你按内容和学生的反应判断**：

- **优先用本课课件**（read_section 读出来的原文），而不是你记忆里的通用教材。
  课件没覆盖的地方可以补充，但要说明那是补充。
- **讲解节奏照 read_section 返回里的契约走**（它会按 mode 给出：复习＝快速过＋多测、
  学习＝细讲＋少测；讲完说「本节结束」、出小测、逐题批改并重讲错的知识点、
  新符号先给全称、确认懂了再进下一节、不串课）。"快慢多少"由你按内容判断。
- **一次只讲一节**：本节讲透再往下，**绝不提前把后面的内容也倒出来**。
  一节的信息量按"学生一节课能消化"来把握（大约半小时到一小时），
  内容多就拆成多节 —— **怎么切由你判断**（同一份课件也可以分几次讲）。
  讲完这一节明确说一句「本节结束」，让学生知道可以歇一下或问问题。
- **讲完就小测**：出 3~5 道覆盖**本节核心考点**的题，题型混着来
  （选择/填空/简答/计算）。题目数量与题型按本节内容定，不必凑固定数字。
- **逐题批改**：每题给 ✓/✗ 和一句讲解；错的要说清**错在哪、为什么错、正确答案是什么**，
  并把相关知识点**重讲一遍** —— 只报对错，学生学不到东西。
- **确认懂了再走**：小测过了、或学生明确说懂了，才进下一节；
  学生卡住就停下来解决那一点，**不要硬推**。
- **新符号必须解释**：任何缩写或符号第一次出现，都要给全称和含义
  （如 TLB＝Translation Look-aside Buffer，快表），别默认学生认识。
- **作业辅导**：先教涉及的知识点 → 让学生自己写 → 批改 → 最后给完整答案和讲解，
  不要一上来就把答案抖出来。
- **不串课**：切换课程/章节时，明确说清"现在在讲哪门课、哪一章、进到哪了"；
  两门课的内容不要混在一起讲。

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
- **不要把工具调用本身写进回复**（比如 `get_paper_questions {"paper_source": ...}` 这种）

""" + QUIZ_BEHAVIOR_RULES + "\n\n" + STUDY_BEHAVIOR_RULES + "\n\n" + STUDY_BEHAVIOR_RULES

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
- 保持自然流畅

⚠️ 工具调用必须走真正的 function calling，**绝不要**把调用过程写进正文：
不许输出 `工具名(...)`、不许输出 `{"query": ...}` 这种参数 JSON、
不许输出 `finish_reason` / `delta` / `tool_calls` 这类协议字段。
如果你发现自己在正文里"打算调用"某个工具，那就**直接调用它**，正文只写给学生看的话。

""" + QUIZ_BEHAVIOR_RULES + "\n\n" + STUDY_BEHAVIOR_RULES


# ============================================================
# Planner Node
# ============================================================

def _grade_call_template(drill: dict) -> str:
    """给 planner 一句**可直接照抄**的批改调用

    为什么明写而不是描述规则：实测只说「用上面的 qids 调 grade_answers」时，
    planner 把那串 qid **填进了 paper_source**，于是批改直接失败
    （`没能确定试卷: 2025OS_Q1,2025OS_Q2,…`）。给可照抄的一句它才会照做 ——
    这套「照抄」办法在 `read_section` 的「继续讲」上已经验证过。
    """
    paper = drill.get("paper", "")
    qids = json.dumps(drill.get("qids") or [], ensure_ascii=False)
    return (f'```\ngrade_answers(paper_source="{paper}", '
            f'user_answer_text="<学生这段答案原文>", qids={qids})\n```\n')


def planner_node(state: AgentState, config=None) -> dict:
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
    #
    # 游标就在 state 里，而 state 由落盘的 checkpointer 保管 —— 所以"进程重启后
    # 还能继续"不再需要额外的副本（那份 runtime/plan_cursors.json 已随 1.2.0 删除）。
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
            prog.append("刚出的题 qids（**这是数组，不是卷名**）："
                        + json.dumps(drill["qids"], ensure_ascii=False))
        prompt += ("\n\n--- 当前刷题进度（来自上一轮，务必据此填参数）---\n"
                   + "\n".join(f"- {x}" for x in prog)
                   + "\n学生说「继续/下一组」时，用 **group_index** 调 get_paper_questions。\n"
                     "学生提交答案时，**原样照抄这一句**（不要改参数名）：\n"
                     + _grade_call_template(drill)
                   + "\n⚠️ 上面那串 `xxx_Q1` 是 **qids（数组）**，"
                     "**绝对不要把它填进 paper_source** —— paper_source 要的是"
                     "「试卷来源名称」这种中文卷名（见上一行）。"
                     "实测填错会导致批改直接失败。")

    # 全局进度（跨会话）：新会话或学生说想学/复习/问进度时给 planner 看。
    # 不注入的话，新窗口说"我想学习"时它手里什么都没有，只能反问"你学到哪了"。
    _pblock = _progress_block(user_content, state.get("session") or {})
    if _pblock:
        prompt += "\n\n" + _pblock

    # 学习/复习书签（与刷题游标同一套机制，互不干扰）。
    # 有了它，学生说"下一节"时才知道该从哪继续；也顺带解决"两门课切换串味"。
    study = drill.get("study") or {}
    if study.get("course"):
        prompt += (
            "\n\n--- 当前学习位置（上一轮记下的书签）---\n"
            f"- 模式：{study.get('mode', '学习')}\n"
            f"- 课程：{study['course']}\n"
            + (f"- 章：{study['chapter']}\n" if study.get("chapter") else "")
            + f"- 上一节：《{study.get('section', '')}》　状态：{study.get('status', '')}\n"
            + (f"- 依据课件：{study['source']}\n" if study.get("source") else "")
            + (f"- 刚读到：{study.get('last_source')} 的 {study.get('last_read')}\n"
               if study.get("last_source") else "")
            + (
                # 给**可直接照抄的一句调用**。实测"接着读同一份课件、用上面的 offset"
                # 这种描述式提示模型不跟：它还是去调 section=1 读了另一份课件，
                # 然后说"课程材料没覆盖这个内容"。照抄一句它反而会照做。
                ("学生说「继续讲」时，**原样照抄这一句**（不要改参数）：\n"
                 f"```\nread_section(course=\"{study['course']}\", "
                 f"source=\"{study.get('last_source')}\", "
                 f"offset={study.get('next_offset')}, limit=15)\n```\n"
                 if study.get("last_source") and study.get("next_offset") is not None
                 else "这份课件已经读完了。学生说「继续/下一节」时，用\n"
                      f"```\nread_section(course=\"{study['course']}\", section=编号+1)\n```\n"
                      "进入下一节。\n")
            )
            + "**不要**在「继续讲」时回到 section=1 重新读第一份课件。\n"
              "**本节的测验题是你自己出的，所以由你自己批**（逐题 ✓/✗ + 讲解，"
              "说清错在哪、为什么、正确答案），**不要**调 grade_answers —— "
              "那是题库真题的批改通道。\n"
              "学另一门课或另一章时，先确认新位置再调 record_study_position 更新书签，"
              "**不要**把两门课的内容混在一起讲。"
        )

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
        # 【1.7.3】合法 JSON 但**没明说"不用工具"**时，也不能当"不需要工具"。
        # 实测（离线打桩 planner）：模型回一个 `{}` 或漏掉 need_tools 时，
        # steps 是空数组、上面那条判据不触发 → normalized=[] →
        # route_after_plan 直接进 synthesizer → **一个工具都不调就凭记忆作答**。
        # prompt 里写死了"必须返回 need_tools"，所以"缺失"= 不合规，不是"不用工具"；
        # 只有**显式** `need_tools: false` 才是真的不需要工具（闲聊、学生交答案
        # 那两轮由守卫 2 单独处理，不受影响）。照旧回退到检索。
        if not normalized and plan_data.get("need_tools", True) is not False:
            raise ValueError("计划里没有任何可用的工具调用，且没有明确 need_tools=false")
    except (json.JSONDecodeError, ValueError, TypeError, AttributeError) as e:
        logger.warning("planner 返回的计划无法解析，回退默认计划: %s", e)
        normalized = [{"tool": "search_knowledge", "args": {"query": user_content}}]

    # 【守卫 1】学生说"没看到题目" → **确定性重发**，不许只口头道歉。
    # 移植自手写助教（那边有这条守卫，BOT 一直没有）。实测模型会回一句
    # 「很抱歉…我将重新发送这组题目」，然后一条题都没发、也没调工具。
    if _missed_questions(user_content) and drill.get("next_group_index") is not None \
            and not any(s.get("tool") == "get_paper_questions" for s in normalized):
        resend_index = max(1, int(drill.get("group_index") or 1))
        logger.info("学生说没收到题目 → 确定性重发第 %s 组（覆盖原计划）", resend_index)
        normalized = [{"tool": "get_paper_questions",
                       "args": {"paper_source": drill.get("paper", ""),
                                "group_index": resend_index,
                                "qtype": drill.get("type") or ""}}]

    # 【守卫 2】讲课/复习中学生交**自拟小测**的答案 →
    # 这一轮不规划任何工具，直接让 synthesizer 自己逐题批。
    # 为什么用代码拦：实测 planner 在"学生提交答案"这种输入上**总是**想到题库
    # （grade_answers / get_paper_questions），而题库里没有这道题，
    # 于是它会反问"你在做哪份卷子"。system prompt 里写了不要用，它不跟。
    study = drill.get("study") or {}
    if study.get("course") and not drill.get("paper") \
            and _looks_like_quiz_answer(user_content):
        logger.info("讲课/复习中收到小测答案 → 不走题库工具，交给 synthesizer 自批")
        normalized = []

    # 【守卫 3】学生在听课/复习，但计划里**没有任何"取材料"的动作** →
    # 这一轮它一定会凭记忆讲（实测：只调 list_course_sections 就开讲，
    # 讲的内容不在课件里，还自己编了题和 example.com 图片）。
    # 这里确定性地补一步 read_section，把课件原文塞到它手里。
    _material = ("read_section", "search_knowledge", "search_knowledge_base")
    if not any(s.get("tool") in _material for s in normalized) \
            and _looks_like_study_request(user_content):
        _study = drill.get("study") or {}
        _course = _study.get("course") or _course_from_text(user_content)
        if _course:
            if _study.get("last_source") and _study.get("next_offset") is not None:
                # 有书签就接着上次读（别把学生拉回第一节）
                _args = {"course": _course, "source": _study["last_source"],
                         "offset": _study["next_offset"]}
            else:
                _args = {"course": _course, "section": 1}
            logger.info("听课/复习但计划里没有取材料 → 补一步 read_section(%s)", _args)
            normalized = [{"tool": "read_section", "args": _args}] + normalized

    # 【守卫 4·1.8.0】学生只是"刚开口说想学/想复习"、还没选模式 → **这一轮不调工具**
    #
    # 本意（用户明确要求）：新窗口说"我想学习"时，先展示已知进度 + 给出四种模式建议，
    # 让他在"接着学 / 自拟练习 / 刷整套卷子 / 复习"里挑一个。
    # 实测不加这条：它直接 read_section 讲了一整节（《03-OS调度.pdf》），
    # 学生根本没机会选。有进度才这么收，没有进度时保持原来的行为（先给资料地图）。
    if progress.has_any() and _is_study_kickoff(user_content):
        logger.info("学生只说想学/想复习 → 先展示进度并让他选模式，本轮不规划工具")
        normalized = []

    out: dict = {"plan": normalized}

    # 【守卫 2】状态上限：MemorySaver 原来**没有任何裁剪**（审查文档 F2），
    # 长会话下 messages 单调增长。这里按 user 消息边界裁剪，并在边界之外
    # 把较早的工具结果正文替换成占位符 —— 与手写助教的 _trim_history +
    # _compact_tools 同口径，只是用 RemoveMessage / 同 id 覆盖来实现。
    pruned = _prune_old_messages(state["messages"])
    if pruned:
        out["messages"] = pruned
    return out


def _prune_old_messages(messages: list) -> list:
    """返回需要写回 state 的"裁剪结果"（RemoveMessage 列表）

    两条规则，都保护消息配对完整性（assistant 的 tool_calls 与 tool 消息必须成对，
    否则接口会 400）：
      1. 超过 _BOT_MAX_MESSAGES 时，在 **user 消息处**切开，丢掉更早的整段；
      2. 较早的工具结果正文压成占位符（保留消息本身与 id，只换 content）。
    返回空列表表示"不需要动"。
    """
    from langgraph.graph.message import RemoveMessage

    out = []
    msgs = messages or []
    if len(msgs) > _BOT_MAX_MESSAGES:
        cut = len(msgs) - _BOT_MAX_MESSAGES
        while cut < len(msgs) and not isinstance(msgs[cut], HumanMessage):
            cut += 1
        if cut < len(msgs):
            for m in msgs[:cut]:
                mid = getattr(m, "id", None)
                if mid:
                    out.append(RemoveMessage(id=mid))

    # 较早的工具结果压缩：只保留最近 _BOT_KEEP_TOOL_TURNS 轮的 ToolMessage 原文
    tool_msgs = [m for m in msgs if isinstance(m, ToolMessage)]
    if len(tool_msgs) > _BOT_KEEP_TOOL_TURNS:
        for m in tool_msgs[:-_BOT_KEEP_TOOL_TURNS]:
            text = m.content if isinstance(m.content, str) else ""
            if len(text) > 200 and getattr(m, "id", None):
                out.append(ToolMessage(
                    content=json.dumps(
                        {"note": "（这一步的工具结果已省略，避免上下文过长）"},
                        ensure_ascii=False),
                    tool_call_id=m.tool_call_id, name=getattr(m, "name", None),
                    id=m.id))
    return out


def route_after_plan(state: AgentState) -> str:
    """条件边：有计划步骤 → executor，无 → 直接回答"""
    if state.get("plan") and len(state["plan"]) > 0:
        return "executor"
    return "synthesizer"


# ============================================================
# Executor Node
# ============================================================

def executor_node(state: AgentState, config=None) -> dict:
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
    #
    # 合并基线只能是 state：它由落盘的 checkpointer 保管，**不存在第二个副本**了。
    # （1.0.1 修过一次"读写不对称" —— 那时还有 runtime/plan_cursors.json，
    #   写入时不认它，于是重启后只更新书签的那一轮会把它抹掉。
    #   1.2.0 换成 SqliteSaver 之后那份副本整个删了，这类不对称从根上没有了。）
    base_session = state.get("session") or {}
    new_session = _extract_drill_session(base_session,
                                        state.get("plan") or [], results)
    quiz_changed = new_session is not None
    study = _extract_study_session(new_session if new_session is not None
                                   else base_session,
                                   state.get("plan") or [], results)
    if study is not None:
        new_session = study
    if new_session is not None:
        out["session"] = new_session

    # 把这一轮的进度**镜像到全局账本**（跨会话，见 course_agent/progress.py）。
    # 只在"这一轮真的更新了书签/游标"时写 —— 否则每轮都写会把 visits 刷成天文数字。
    _mirror_to_global_progress(
        new_session if new_session is not None else base_session,
        results,
        thread_id=_thread_id(config),
        study_changed=study is not None,
        quiz_changed=quiz_changed,
    )
    return out


def _thread_id(config) -> str:
    return str(((config or {}).get("configurable") or {}).get("thread_id") or "")


def _mirror_to_global_progress(session, results, thread_id, study_changed, quiz_changed):
    """把会话状态镜像到全局进度账本（跨会话）

    为什么镜像"会话状态"而不是重新解析工具返回：`context`/`state` 里的 session
    已经是**合并归一后**的唯一真源（游标合并、空值不覆盖都在里面做过了）。
    再解析一遍工具返回就会出现两套判据。

    **任何异常都不许影响这一轮问答** —— 进度是附带收益，不是主链路。
    """
    try:
        study = (session or {}).get("study") or {}
        if study_changed and study.get("course"):
            progress.record_study({
                "mode": study.get("mode"),
                "course": study.get("course"),
                "chapter": study.get("chapter"),
                "section": study.get("section"),
                "source": study.get("last_source") or study.get("source"),
                "status": study.get("status"),
            }, thread_id=thread_id)
        if study.get("course") and study.get("last_source"):
            progress.record_read(study["course"], study["last_source"],
                                 study.get("next_offset"), thread_id=thread_id)
        if quiz_changed and (session or {}).get("paper"):
            progress.record_quiz({
                "paper": session.get("paper"),
                "type": session.get("type"),
                "group_index": session.get("group_index"),
                "next_group_index": session.get("next_group_index"),
                "qids": session.get("qids") or [],
            }, thread_id=thread_id)
        paper = (session or {}).get("paper") or ""
        for msg in results or []:
            try:
                data = json.loads(getattr(msg, "content", "") or "")
            except (TypeError, ValueError):
                continue
            if isinstance(data, dict) and "total" in data and "correct" in data:
                progress.record_grade(paper, data, thread_id=thread_id)
    except Exception:                                   # noqa: BLE001
        logger.warning("写全局学习进度失败（不影响本轮问答）", exc_info=True)


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


def _extract_study_session(current: dict, plan: list, results: list):
    """看这一轮有没有"学习书签"或"读到哪了"要记，有就更新

    书签只是**记忆**：讲什么、怎么分节、何时进下一节都由模型判断
    （刻意不做成状态机 —— 用户明确说过不要）。

    除了显式的 record_study_position，这里还会从 read_section 的返回里
    偷偷记住"刚读的是哪份课件、下一段从 offset 几开始" ——
    因为实测模型**记不住**返回值里的"下次用 offset=15 继续"，
    学生说"继续讲"时它会跳到下一份课件、然后声称"没找到内容"再退回凭记忆讲。
    把这件事交给状态，它就不用记了。
    """
    import json as _json

    updated = None
    for step, msg in zip(plan or [], results or []):
        tool = step.get("tool")

        if tool == "record_study_position":
            try:
                data = _json.loads(msg.content)
            except (TypeError, ValueError, AttributeError):
                continue
            pos = (data or {}).get("recorded") if isinstance(data, dict) else None
            if not isinstance(pos, dict) or not pos.get("course"):
                continue
            updated = dict(updated if updated is not None else (current or {}))
            merged_pos = dict((updated.get("study") or {}))
            # 空值不覆盖已有的非空值：实测模型会传 chapter="" 把上一轮记住的章抹掉
            merged_pos.update({k: v for k, v in pos.items() if str(v or "").strip()})
            updated["study"] = merged_pos

        elif tool == "list_course_sections":
            # 只调了地图、没调 read_section 的情况：至少把"在学哪门课"记下来，
            # 否则学习契约（以课件原文为准 / 收尾 / 不要编题编图）全都不会注入 ——
            # 实测那一轮模型就凭记忆讲课、还编了题和 example.com 图片。
            text = msg.content if isinstance(msg.content, str) else ""
            m = re.search(r"\[study-course\]\s*(.+)", text)
            if not m:
                continue
            kv = dict(re.findall(r"(\w+)=(\S*)", m.group(1)))
            if not kv.get("course"):
                continue
            updated = dict(updated if updated is not None else (current or {}))
            study = dict(updated.get("study") or {})
            study.setdefault("course", kv["course"])
            study["mode"] = kv.get("mode") or study.get("mode") or "学习"
            updated["study"] = study

        elif tool == "read_section":
            # 解析工具显式给出的 [study-state] 行（不要解析自然语言措辞 —— 太脆：
            # 我第一版就是按措辞解析，结果 study 里没有 course，
            # planner 的"当前学习位置"整块被跳过，模型照样不知道读到哪了）
            text = msg.content if isinstance(msg.content, str) else ""
            m = re.search(r"\[study-state\]\s*(.+)", text)
            if not m:
                continue
            kv = dict(re.findall(r"(\w+)=(\S*)", m.group(1)))
            if not kv.get("course"):
                continue          # 课程不明就不写 —— 免得覆盖掉已有的书签
            updated = dict(updated if updated is not None else (current or {}))
            study = dict(updated.get("study") or {})
            study["course"] = kv["course"]
            if kv.get("chapter"):
                study["chapter"] = kv["chapter"]
            study["last_source"] = kv.get("source", "")
            study["last_read"] = f"第 {kv.get('read', '?')} 块"
            nxt = kv.get("next_offset", "none")
            study["next_offset"] = None if nxt in ("none", "", "None") else int(nxt)
            updated["study"] = study

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
# 回答的收尾清理（**两条路共用**）
# ============================================================

def strip_protocol_frames(text: str) -> str:
    """摘掉正文里混进来的**原始协议帧**（模型把工具调用协议当普通文本打出来）

    ## 实测长什么样（2026-10-16，ReAct 模式，3 次里命中 1 次）

    这一轮模型**没有真的调工具**（`tool_calls` 是空的），而是把协议当正文写了：

    ```
    好的，我们来复习死锁……我会调用一下知识库来获取更详细的信息。
    {"course": "操作系统", "query": "死锁的四个必要条件"}      ← 工具参数被写成普通文本
    ```{"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant",…
    ```

    最后那段是**原始 API 响应**（`finish_reason` / `delta` / `tool_calls`），
    它是模型的"幻觉续写"，不是我们漏播的——但学生看到的就是一坨协议。

    ## 判据：**解析出来的 JSON 必须命中"协议字段"**，不是关键词匹配

    只按字面量（如 `"tool_calls" in text`）会误伤教学文本（"这个 JSON 里的
    tool_calls 字段…"）。所以这里扫 `{` 起点、逐层配对括号取出完整片段、
    `json.loads` 成功、且顶层/一层里命中 ≥2 个协议字段，才动手摘。
    配不成对的（文本里本来就有个孤立花括号）一律不动。
    """
    if not text:
        return text
    out, i, n = [], 0, len(text)
    while i < n:
        if text[i] != "{":
            out.append(text[i])
            i += 1
            continue
        end = _json_object_end(text, i)
        candidate = text[i:end] if end else ""
        if candidate and _is_protocol_frame(candidate):
            # 连同紧邻的 ``` 围栏与空行一起清掉（实测那一段就裹在围栏里）
            while out and out[-1] in ("`", "\n", " ", "\t"):
                out.pop()
            i = end
            if text.startswith("```", i):
                i += 3
            logger.warning("摘掉正文里的原始协议帧（%d 字符）", len(candidate))
            continue
        out.append("{" if not candidate else candidate)
        i = (i + 1) if not candidate else end
    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()


_PROTOCOL_FRAME_KEYS = frozenset({
    "delta", "finish_reason", "tool_calls", "tool_call_id", "reasoning_content",
})


def _json_object_end(text: str, start: int):
    """从 `start` 处的 `{` 开始配对括号，返回**含**结尾 `}` 的下标；配不成对返回 None

    字符串字面量与转义要跳过 —— 否则正文里一个 `"}"` 就会让配对错位，
    把一个正常句子吞掉。
    """
    depth, i, n = 0, start, len(text)
    in_str = esc = False
    while i < n:
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i + 1
        i += 1
    return None


def _is_protocol_frame(candidate: str) -> bool:
    """这段 JSON 是不是"原始协议帧"（顶层或一层嵌套里命中 ≥2 个协议字段）"""
    if len(candidate) > 20000:
        return False
    try:
        obj = json.loads(candidate)
    except (ValueError, TypeError):
        return False
    if not isinstance(obj, dict):
        return False
    keys = set(obj)
    for value in obj.values():
        if isinstance(value, dict):
            keys |= set(value)
    return len(keys & _PROTOCOL_FRAME_KEYS) >= 2


def clean_answer_text(text: str, state: dict = None) -> str:
    """摘掉回答里不该给学生看的东西（工具调用原文 / 编造的图片链接 / 学习收尾标记）

    ## 为什么必须是共用函数（1.7.3 修的真问题）

    原来这三件事只写在 `synthesizer_node` 里（只有 plan 模式那条路），
    ReAct 的 `llm_node` **只筛选文本、什么都不清** —— 于是 ReAct 模式下
    `explain_question {"question_index": 1}` 这种工具的调用原文会**直接进学生气泡**
    （评测用例 `react_mode_streams` 抓到过一次）。清理逻辑分散写就会出现这种
    "只修好一条路"的分叉，所以这里提成一个函数，两条路都调。

    ## 与流式的关系（必须知情）

    清理作用于**节点返回的那条消息**。流式已经把逐字 chunk 发出去了，
    所以清理**救不了打字机过程中闪过去的那一段**；但 `done.answer` 是清理后的
    权威值，前端在 done 时会用 answer 覆盖一次，最终气泡是干净的（这条兜底
    见 `server.py::qa_stream` 的注释）。要彻底不让它出现，得在生成时就拦住
    （提示词实测拦不住）。

    Args:
        text: 模型写出来的回答
        state: 会话状态；给了才做"学习模式补「本节结束」"这一步
    """
    cleaned = strip_tool_call_text(str(text or ""))
    # 再摘"原始协议帧"（模型把工具调用协议当正文打出来，实测 ReAct 3 次里 1 次）
    cleaned = strip_protocol_frames(cleaned)
    # 编造的图片链接（合法图片只有题库题图 /qb_images/…）：提示词写了不许编、
    # 实测照样编，所以这里用代码兜住
    cleaned, dropped = quiz_service.sanitize_images(cleaned)
    if dropped:
        logger.warning("摘掉回答里编造的图片链接 %d 个: %s", len(dropped), dropped[:3])
    # 学习/复习时，收尾标记由代码保证（模型只有 3/4 会写，而它必须可靠）
    if (state or {}).get("session", {}).get("study", {}).get("course"):
        cleaned, added = quiz_service.ensure_closing(cleaned)
        if added:
            logger.info("模型没写「本节结束」→ 已由代码补上")
    if cleaned != text:
        logger.info("清理回答里不该给学生看的内容 (%d → %d 字符)", len(str(text)), len(cleaned))
    return cleaned


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
    # 会话状态（学习书签 / 刷题游标）必须给**写回答的这个模型**看。
    # 原来只注入了 planner —— 而"本节结束 / 出小测 / 小测自己批"这些是写给
    # 最终回答的契约，学生交小测答案那一轮更是一个工具都不调（走 CHITCHAT_PROMPT）。
    base_prompt += _session_block(state.get("session") or {})
    # 「全局进度 + 四选一建议」：新会话或学生说想学/复习/问进度时才注入。
    # 这就是"开窗不用再说一遍学到哪"的落点（文案由 progress.brief() 用代码生成）。
    _pblock = _progress_block(user_content, state.get("session") or {})
    if _pblock:
        base_prompt += "\n\n" + _pblock
    if summary_text:
        base_prompt = f"{base_prompt}\n\n{SUMMARY_MARK}\n{summary_text}"
    # 收尾动作放在**最末尾**（连摘要都在它前面）。
    # 实测三档：写在提示中段 → 3 轮里 1 轮照做；只说一句要求+放末尾 → 2/3；
    # 所以最后改成**给可照抄的收尾格式**，把不可靠的动作（写「本节结束」）
    # 绑到可靠的动作（出小测，3/3）上一起做。格式的唯一出处是 quiz_service。
    if (state.get("session") or {}).get("study", {}).get("course"):
        base_prompt += "\n\n--- " + quiz_service.closing_block()
    # 工具报错时的行为约束（实测：批改失败后模型会"假装重试"、
    # 编造一个不存在的工具名（get_paper_answers），或者让学生改格式重发）
    if any(isinstance(m, str) and '"error"' in m for m in tool_results):
        base_prompt += (
            "\n\n--- 注意：这一步有工具返回了 error ---\n"
            "· **不要**说「我将再次尝试」然后什么都不做；也**不要**编造工具名或调用方式"
            "（例如凭空写 `get_paper_answers {…}`）—— 学生会看到内部协议。\n"
            "· **不要**把工具的问题说成学生的格式问题（除非确实是他没给答案）。\n"
            "· 该做的：用上面【当前试卷】/【当前学习位置】里的信息，能自己补就补一次；"
            "补不了就**如实告诉学生卡在哪**，并问他一个具体的问题（如「是这一份卷子吗」）。")

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
        # 兜底：把泄漏成正文的工具调用片段、编造的图片链接、学习收尾标记都处理掉
        # （正解是允许一次重规划，见文件头注释）。清理逻辑与 ReAct 侧共用
        # `clean_answer_text` —— 分散写就会出现"只修好一条路"。
        if isinstance(response, AIMessage) and response.content:
            cleaned = clean_answer_text(str(response.content), state)
            # 开场那轮：四种模式必须都在（模型实测会漏掉「自拟练习」）
            if progress.has_any() and _is_study_kickoff(user_content):
                cleaned, added = progress.ensure_modes(cleaned)
                if added:
                    logger.info("开场建议漏了 %d 条 → 已用代码补上", len(added))
            if cleaned != response.content:
                response = AIMessage(content=cleaned)
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
        # 【1.7.3】ReAct 这条路的回答也要清理 —— 原来这里**什么都不清**，
        # 于是 `explain_question {"question_index": 1}` 这种工具调用原文
        # 会直接进学生气泡（评测用例 react_mode_streams 抓到过）。
        # 与 synthesizer 共用 `clean_answer_text`，避免再分叉。
        if isinstance(response, AIMessage) and response.content:
            cleaned = clean_answer_text(str(response.content), state)
            if cleaned != response.content:
                response = AIMessage(content=cleaned)
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
