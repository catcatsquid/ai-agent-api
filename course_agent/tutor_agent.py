"""课程学习助手 —— LLM 主导的对话式学习 Agent

设计原则（相对旧版的关键改动）
────────────────────────────────────────────────────────────
旧版把"分组出题、不许念题干、必须调工具批改"这些**结构约束写死在 system prompt 里**，
靠"⚠️ 铁则""绝对不能"去求模型遵守。结果一是不自然（模型动不动就复述规则），
二是不可靠（模型换种说法就失效），三是 prompt 越写越硬，交互越死。

新版把这些约束下沉到**工具返回值**里：
  - 一组的题量由 get_paper_questions 决定（选择/填空/判断 5 题、综合题 1 题），
    模型拿到几条就只能展示几条，不需要它自己数。
  - 出题接口只返回题号/题型/知识点/图片，题干和答案根本不在上下文里，
    模型想泄题也无从泄露。
  - 每个工具结果的 `_guide` 字段直接告诉模型"这一步该怎么呈现、下一步能做什么"，
    所以 system prompt 可以写得很短、很自然，只描述人设和可用能力。
"""

from __future__ import annotations

import json
import re
import time
import uuid

import requests

from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL, ZHIPU_MODEL
from course_agent import qbank, quiz_service

# ============================================================
# 1. System Prompt —— 只写人设和能力，不写流程铁律
# ============================================================

SYSTEM_PROMPT = """你是「知识寻回BOT」，一位耐心、专业的大学课程学习助手，主要负责操作系统和计算机网络两门课。

你像一位真人助教一样跟学生聊天，能做的事：
- 讲课答疑：检索课程知识库，把知识点讲清楚
- 刷题训练：从历年真题/模拟题里出题，学生做完帮着批改和讲解
- 复习梳理：总结知识点、串讲易混概念、回顾做错的题
- 其他学习相关的请求：按学生的意思来

几条工作习惯：
- 出题和批改都有专门的工具，工具返回什么就照着呈现什么，不要自己另编题目或自己判对错。
- 工具返回里如果带 `_guide` 字段，那是当前这一步的呈现建议，照做即可。
- 学生说「不会 / 没思路 / 讲讲这道题」时，他是真的要你教，不是要你鼓励：
  先用 explain_question 把题面和参考要点取来，再把考点、为什么这么做、
  容易错在哪一步步讲清楚。只回一个答案、或者让他「再想想」，都等于没帮上忙。
- 学生可能随时岔开话题（问知识点、抱怨、让你换卷子、让你额外出题），
  你就顺着他的意思走，不要硬把他拽回原来的流程。
- 回答用 Markdown，语气自然，别把内部规则念出来。

【系统讲课 / 复习】学生说「我要学/我要复习某门课（或某章）」时，是带你系统过一遍，
不是甩一堆知识点。先 list_course_sections 拿资料地图，再用 read_section 取课件原文讲
（**优先课件**，别用记忆里的通用教材替换）。
**具体节奏照 read_section 返回里的契约走**（它会按 mode 给出：复习＝快速过＋多测、
学习＝细讲＋少测；讲完说「本节结束」、出小测、逐题批改并重讲错的知识点、
新符号先给全称、确认懂了再进下一节、不串课）。这里的"快慢多少"由你按内容判断，
契约给的是原则不是死规矩。
讲完一节用 record_study_position 记下位置（带上 mode），
这样学生说「继续讲 / 下一节」时你知道从哪接着来。"""


# ============================================================
# 2. 工具定义
#
# 说明：工具描述里写的是"这个工具返回什么、拿它做什么"，
# 而不是"你必须如何如何"——约束靠返回值本身兜住，不靠语气。
# ============================================================

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "检索课程知识库（课件、课本笔记），返回相关知识点片段。用于讲解、答疑、复习。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "检索关键词或问题"},
                    "course": {"type": "string", "enum": ["操作系统", "计算机网络"],
                               "description": "限定课程，避免跨课程串味"},
                    "source": {"type": "string",
                               "description": "限定到某一份课件（讲某一章时用，名字来自 list_course_sections）"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_papers",
            "description": "列出题库里可用的试卷，按课程分组，并给出每份卷子的题型构成和分组数。学生说想刷题/做卷子时调用它。",
            "parameters": {
                "type": "object",
                "properties": {
                    "course": {"type": "string", "enum": ["操作系统", "计算机网络"],
                               "description": "只列某门课的卷子；不传则两门课都列"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_paper_questions",
            # 描述只讲"是什么 + 硬边界"，呈现细节放在返回值的 reply_text / _guide 里。
            # （schema 里的枚举式说明会被模型当成规章，反而收窄行为 —— 见成因分析文档）
            "description": (
                "从题库取一组题给学生做，一次一组。\n"
                "只能出题库里已有的卷子；学生要「临时出几道题/针对某知识点出题」时不要用这个工具。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_source": {"type": "string", "description": "试卷名（用 list_papers 返回的 source 原文）"},
                    "group_index": {"type": "integer",
                                    "description": "组序号，从 1 开始；不传则自动取下一组"},
                    "qtype": {"type": "string",
                              "description": "可选，只出某个题型：选择题/填空题/判断题/综合题/整页综合"},
                },
                "required": ["paper_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grade_answers",
            # 只保留"什么时候调 + 硬边界"这两件必须放在 schema 里的事。
            # 举例要写明"举例而非穷尽"，否则会被当成规章：
            # 原来那版列了 4 行格式清单，输入 AAAAA 时模型判定"不在清单里"直接拒绝。
            # 但也**不能完全不给例子** —— 一个例子都没有时，模型会自己发明格式要求
            # （实测出现过"请按 题号+知识点+图片 提交答案"）。所以用"举例而非穷尽"的写法。
            "description": (
                "批改学生答案，返回逐题结果（选择/判断由代码精确判定，填空/综合依参考要点评分）。\n"
                "学生发来的内容只要像作答就原文照传，例如「1.A 2.B」「ABCDA」「AAAAA」"
                "「全都选A」「第一题C」「对错对错」—— **这是举例，不是穷尽**，"
                "凡是作答写法都交给工具去解析。\n"
                "不要自己判断格式合不合规，也不要先让学生重发。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_source": {"type": "string", "description": "试卷名"},
                    "user_answer_text": {"type": "string",
                                         "description": "学生答案原文，原样传入，工具自己会解析格式"},
                    "qids": {"type": "array", "items": {"type": "string"},
                             "description": "这一组题的 qid 列表（get_paper_questions 返回过），强烈建议传入，能准确定位到具体题目"},
                },
                "required": ["paper_source", "user_answer_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "prepare_custom_quiz",
            "description": (
                "取「临时出题」所需的课程材料。学生不想做题库里的卷子、想针对某知识点多练几道、"
                "想改题型或难度时用它。\n"
                "注意：它**不返回题目**，返回的是素材 —— 拿到素材后由你自己把题写出来。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "topic": {"type": "string", "description": "知识点或章节，如「死锁」「TCP拥塞控制」"},
                    "course": {"type": "string", "enum": ["操作系统", "计算机网络"],
                               "description": "课程；能从话题推断出来就填"},
                    "qtype": {"type": "string",
                              "description": "想要什么题型：选择题/填空题/判断题/简答题/计算题"},
                    "count": {"type": "integer", "description": "出几道，默认 5"},
                    "difficulty": {"type": "string", "description": "难度：基础/进阶/综合"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "explain_question",
            "description": (
                "取出题目原文、参考答案和评分要点，用于给学生讲题"
                "（学生说不会 / 没思路 / 让你讲讲时用它）。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "paper_source": {"type": "string", "description": "试卷名；不传则用当前正在做的卷子"},
                    "no": {"type": "array", "items": {"type": "integer"},
                           "description": "要讲的题号，如 [1] 或 [1,2]；不传则讲当前这一组"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_practice_stats",
            "description": "查询某份试卷被练过几次。",
            "parameters": {
                "type": "object",
                "properties": {"paper_source": {"type": "string"}},
                "required": ["paper_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_practice",
            "description": "记一次练习：把某份试卷的练习次数 +1。一份卷子完整做完后调用。",
            "parameters": {
                "type": "object",
                "properties": {"paper_source": {"type": "string"}},
                "required": ["paper_source"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_course_sections",
            "description": "列出某门课**实际有哪些课件/课本/笔记**（带章节名与内容量）。"
                           "学生说「我要复习/我要学 某门课或某章」时先调它拿到真实资料地图，"
                           "再决定怎么分节 —— 不要凭记忆编章节。",
            "parameters": {
                "type": "object",
                "properties": {
                    "course": {"type": "string", "enum": ["操作系统", "计算机网络"],
                               "description": "课程名；留空则两门课都给"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_section",
            "description": "按**课件原始顺序**读某一节（可分页）。系统讲一节时优先用它 —— "
                           "比 search_knowledge_base 更全、顺序不乱。不需要记文件名："
                           "学生说「某一章」就用 chapter，说「第几节/接着讲」就用 section。",
            "parameters": {
                "type": "object",
                "properties": {
                    "course": {"type": "string", "enum": ["操作系统", "计算机网络"],
                               "description": "课程名（必填）"},
                    "chapter": {"type": "string",
                                "description": "章名，如 死锁 / OS调度 / Transport Layer"},
                    "section": {"type": "integer", "description": "第几份课件，从 1 开始"},
                    "source": {"type": "string", "description": "课件名（一般不用）"},
                    "offset": {"type": "integer", "description": "从第几块开始，默认 0"},
                    "limit": {"type": "integer", "description": "读几块，默认 15"},
                    "mode": {"type": "string", "enum": ["学习", "复习"],
                             "description": "复习=快速过+多测；学习=细讲+少测"},
                },
                "required": ["course"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "record_study_position",
            "description": "记下「学到哪一节了」的书签（**不是状态机**，只是记忆）。"
                           "开始讲一节时记一次；本节讲完小测通过或学生卡住时用 status 更新。",
            "parameters": {
                "type": "object",
                "properties": {
                    "course": {"type": "string", "enum": ["操作系统", "计算机网络"]},
                    "section": {"type": "string", "description": "这一节的标题（你自己起的）"},
                    "mode": {"type": "string", "enum": ["学习", "复习"]},
                    "chapter": {"type": "string"},
                    "source": {"type": "string", "description": "本节依据哪份课件"},
                    "status": {"type": "string",
                               "description": "讲解中 / 已通过 / 待巩固"},
                },
                "required": ["course", "section"],
            },
        },
    },
]


# ============================================================
# 3. 工具实现
#
# 统一签名 handler(args, session)，session 用来记住"当前在做哪份卷子、出到第几组"，
# 这样学生说"下一组""继续"时不需要模型自己记进度。
# ============================================================

def _tool_search_knowledge_base(args, session):
    from rag_tool import get_knowledge_base

    query = args.get("query", "")
    if not query.strip():
        return {"error": "query 不能为空"}

    course = (args.get("course") or "").strip()
    source = (args.get("source") or "").strip()
    kb = get_knowledge_base()
    top_k = int(args.get("top_k") or 5)
    chapter = args.get("chapter")

    if source:
        # 限定到某一份课件 —— 「讲某一章」时最准（课件名来自 list_course_sections）
        real = quiz_service.match_source(
            source, [d["source"] for d in kb.list_documents()])
        if real is None:
            return {"error": f"没找到叫「{source}」的资料",
                    "_guide": "先用 list_course_sections 看这门课有哪些课件。"}
        hits = kb.search(query, top_k=top_k, sources=[real])
        note = f"（只在《{real}》里检索）"
    elif course:
        hits, confirmed = quiz_service.search_with_course(kb, query, course,
                                                             top_k=top_k)
        note = "" if confirmed else f"（限定课程「{course}」没有真正相关的片段，下面是全库检索结果）"
    else:
        hits = kb.search(query, top_k=top_k, chapter=chapter)
        note = ""

    return {
        "query": query,
        "course": course or "全部",
        "scope": source or course or "全库",
        "note": note,
        "count": len(hits),
        "passages": [
            {"chapter": r.get("chapter", ""), "source": r.get("source", ""),
             "text": r.get("text", "")[:700]}
            for r in hits
        ],
        "_guide": (
            "基于这些片段用自己的话讲解；引用到具体章节时可以点名，不要整段照抄。"
            "**以这些片段为准**，不要用你记忆里的通用教材内容替换它；"
            "片段里没覆盖的点，就直说「课程资料里这部分没找到」，"
            "再用通用知识补充并标明那是补充。\n"
            "任何缩写/符号第一次出现都要给全称和含义（如 LRU＝Least Recently Used）。\n"
            "如果学生要的是「针对某个知识点另外出几道题」，就用这些内容自己出题"
            "（题号 + 题干 + 选项/空格，可以另附答案折叠或等学生做完再给）。"
        ),
    }


def _tool_list_papers(args, session):
    requested = (args.get("course") or "").strip() or None
    stated = session.get("last_user_message", "")

    # 只有学生自己说了要哪门课，才真的按课程过滤。
    # 模型经常在"我想刷题"这种没指定课程的情况下自己填 course，
    # 结果两门课的题库只列出一半 —— 而"按课程分类、两门课都能看到"
    # 正是刷题入口的硬要求，交给模型随手决定不可靠。
    ignored_filter = None
    course = None
    if requested:
        if requested in stated:
            course = requested
        else:
            ignored_filter = requested

    papers = qbank.list_papers(course)
    if not papers:
        return {"papers": [], "courses": [],
                "error": "题库为空或该课程下没有试卷",
                "_guide": "如实告诉学生题库里还没有卷子。"}

    courses: dict[str, list] = {}
    for p in papers:
        outline = qbank.paper_outline(p["source"])
        sections = "、".join(
            f"{s['type']}{s['count']}题（{s['group_count']}组）" for s in outline["sections"]
        )
        courses.setdefault(p["course"], []).append({
            "source": p["source"],
            "total": p["total"],
            "sections": sections,
            "auto_gradable": p["auto_gradable"],
        })

    # 直接把清单排好版给模型抄。
    # 只给嵌套结构的话，模型会漏段 —— 实测"我想刷题"时工具返回了 8 份卷子
    # （两门课都在），模型却只把操作系统那 5 份写进了回复里。
    # 字段名用 reply_text（"这就是要发出去的话"）比 _render_example 更容易被照搬。
    lines = []
    for c, ps in sorted(courses.items()):
        lines.append(f"**{c}**")
        for i, p in enumerate(ps, 1):
            lines.append(f"{i}. {p['source']}（共{p['total']}题：{p['sections']}）")
        lines.append("")
    listing = "\n".join(lines).strip()
    reply_text = (
        f"这里是题库里的试卷，按课程分好了：\n\n{listing}\n\n"
        "想做哪一份直接告诉我卷名就行；如果这些都不想做，"
        "我也可以现场给你另出题（可以说课程、章节、题型、难度）。"
    )

    return {
        "courses": [
            {"course": c, "papers": ps} for c, ps in sorted(courses.items())
        ],
        "total_papers": len(papers),
        "applied_course_filter": course,
        "ignored_course_filter": ignored_filter,
        "reply_text": reply_text,
        "_render_example": listing,
        "_guide": (
            "reply_text 就是这一步要发给学生的话，把它作为回复主体发出去。"
            "可以加一两句自己的语气，但**不要删掉任何一份卷子** —— "
            "学生没说只要某门课时，两门课的卷子都必须出现在回复里。"
            "这一步只列清单，不要开始出题。"
        ),
    }


# 学生"要新题"的意图：明确要另外出题，而不是继续做手头这套卷子。
# 这不是在写死对话流程，而是给"出题范围"划一条界 ——
# get_paper_questions 只能出题库里已有的卷子，学生要新题时用它必然答非所问
# （实测模型会把计网的选择题标成"死锁的定义""产生死锁的四个必要条件"发出去）。
_CUSTOM_QUIZ_INTENT = re.compile(
    r"(另外|额外|重新|再来|再出|给我出|帮我出|现场出|自己出|生成|编几道|针对).{0,12}?题"
)
_CONTINUE_INTENT = re.compile(r"下一组|下一题|继续|接着|往下|做完|换一套|换卷|换题")

# 学生说"没看到题 / 题目呢 / 再发一遍" —— 说明上一次出题没能真正送达。
# 这种情况下必须把**当前这一组**重发，不能顺着游标往下走：
# get_paper_questions 是会推进游标的，模型如果把题取出来却没展示给学生，
# 那一组就被静默跳过了（实测发生过：学生说"继续刷题"，游标前进但一条题都没给）。
_MISSED_QUESTIONS = re.compile(
    r"没看到|没有题|题目呢|题呢|没给|没发|没显示|没出来|漏了|再发一遍|重新发|再看一遍|重复一遍"
)


def _wants_custom_quiz(text: str) -> bool:
    """学生是不是在要"新出的题"（而不是继续做题库里的卷子）"""
    text = str(text or "")
    return bool(_CUSTOM_QUIZ_INTENT.search(text)) and not _CONTINUE_INTENT.search(text)


def _missed_questions(text: str) -> bool:
    """学生是不是在说"题目没给我" """
    return bool(_MISSED_QUESTIONS.search(str(text or "")))


def _tool_get_paper_questions(args, session):
    # 范围守卫：学生要的是"临时出的新题"，而本工具只能翻题库。
    # 直接拒绝并指路，比让模型把题库题目硬说成"死锁专题"要好。
    if _wants_custom_quiz(session.get("last_user_message", "")):
        return {
            "error": "学生这次要的是临时出的新题，不是题库里的卷子。",
            "use_instead": "prepare_custom_quiz",
            "_guide": (
                "不要从题库出题了。改用 prepare_custom_quiz 取到相关课程材料，"
                "然后自己写题（题号 + 题干 + 选项/空格）。"
                "如果材料没覆盖到，就直接凭你的知识写，不要假装是题库里的原题。"
            ),
        }

    raw_name = args.get("paper_source", "")
    course_hint = session.get("current_course")
    paper, candidates = qbank.resolve_paper(raw_name, course_hint)

    if paper is None:
        if candidates:
            return {
                "error": f"没能确定是哪份卷子: {raw_name}",
                "available_papers": candidates,
                "_guide": "把这几份候选卷子报给学生，让他确认是哪一份，不要自己猜。",
            }
        return {"error": "题库为空", "available_papers": []}

    source = paper["source"]
    qtype = (args.get("qtype") or "").strip() or None

    group_index = args.get("group_index")
    served_current = False
    if group_index in (None, "", 0):
        last = session.get("last_group") or {}
        same_paper = last.get("source") == source and (qtype is None or last.get("type") == qtype)
        if same_paper and _missed_questions(session.get("last_user_message", "")):
            # 学生说没看到题 → 重发当前这一组，不要推进游标（否则整组被静默跳过）
            group_index = int(last.get("index", 1)) or 1
            served_current = True
        elif same_paper:
            # 没指定就顺着上次的进度往下走；换卷子或换题型则从头开始
            group_index = int(last.get("index", 0)) + 1
        else:
            group_index = 1
    group_index = max(1, int(group_index))

    # 出题/排版逻辑在 quiz_service（与 tools.py 共用一份）。
    # 这里保留"这一侧特有"的两件事：会话游标的推进策略，以及重发守卫。
    payload = quiz_service.group_payload(paper, qtype=qtype, group_index=group_index)
    group = payload

    if group.get("error"):
        return {**group, "_guide": "如实告诉学生这个题型不存在或卷子没题，并给出可选的题型。"}
    if group.get("finished"):
        session["last_group"] = {"source": source, "type": None, "index": group["total_groups"]}
        return {
            **group,
            "_guide": ("这份卷子出完了。可以问问学生要不要换一份卷子、"
                       "或者针对错题再练几道、或者我现场出几道新题。"),
        }

    session["current_paper"] = source
    session["current_course"] = paper.get("course", "")
    session["last_group"] = {"source": source, "type": group["type"], "index": group_index}
    # 记下这一组，批改时即使模型没传 qids 也能定位
    session["current_qids"] = list(group["qids"])

    return {
        "paper": payload["paper"],
        "type": payload["type"],
        "group_index": payload["group_index"],
        "group_count": payload["group_count"],
        "progress": payload["progress"],
        "questions": payload["questions"],
        "qids": payload["qids"],
        "has_more": payload["has_more"],
        "next_group_index": payload["next_group_index"],
        "reply_text": payload["reply_text"],
        "_render_hint": "每题一行，题号 + 知识点 + 图片，格式如下：",
        "_render_example": payload["render_example"],
        "_guide": (
            "reply_text 就是这一步要发给学生的话，把它作为回复主体发出去 —— "
            "**题目必须整组出现在回复里**，不能只说一句「这是下一组」却忘了贴题。\n"
            "注意 reply_text 末尾特意说明了「答案怎么写都行」：这里的题号+知识点+图片是"
            "**展示格式**，不是作答格式。学生回你 `1.D 2.B`、`DBDAC`、`全都选A` 都算作答，"
            "你都要原样转给 grade_answers，绝不能反过来要求他按展示格式提交。\n"
            "题干文字和答案都不在你手里，也不用去猜；学生做完了会发答案给你。"
            "不要自己给这些题另起标题或改动知识点名称 —— reply_text 里的就是题库原文。\n"
            "重要：学生接下来发的任何**像作答**的内容，一律原样传给 grade_answers 再解析 —— "
            "包括「AAAAA」「全选B」「我选C」「1.A 2.B」「第一题对」「1.D2.B」这类写法。"
            "你没有资格判断格式合不合规，解析是工具的事；不要先回一句『你没提交答案』"
            "『格式不对』把学生挡回去。\n"
            "如果学生说不会 / 没思路 / 让你讲讲某道题，调用 explain_question 取到题面和"
            "参考答案后再讲，别凭图片瞎猜。\n"
            "如果这一次你没能把题目真的发给学生（正文里没有题目），"
            "下一次必须用 group_index 指定 group_index="
            f"{group_index} 把同一组补上 —— 出题会推进进度，别以为取过就等于学生看到了。"
        ),
    }


def _tool_grade_answers(args, session):
    study = session.get("study") or {}
    raw_name = args.get("paper_source", "") or session.get("current_paper") or ""

    # 硬守卫：正在"讲课/复习"时，学生交的答案属于**本节自拟小测**，不是题库真题。
    # 实测模型会习惯性地走这个工具，然后反问学生"你在做哪份卷子"，
    # 把学习流程硬拽回刷题流程 —— 光在 system prompt 里写"不要用"它不跟，
    # 所以在**工具入口**直接拦下来，并把正确做法写在返回里。
    if study.get("course") and not raw_name:
        return {
            "error": "当前在讲课/复习，学生交的是本节小测的答案 —— 不走题库批改",
            "study": study,
            "_guide": (
                "**你自己出的小测由你自己批**：直接逐题给 ✓/✗ 和一句讲解，"
                "错的要说清**错在哪、为什么错、正确答案是什么**，并把相关知识点**重讲一遍**。\n"
                "**不要**再问学生「是哪份卷子」，也不要调题库工具 —— "
                "现在是在讲 "
                f"{study['course']}"
                + (f" · {study['chapter']}" if study.get("chapter") else "")
                + f" 的《{study.get('section', '')}》这一节。"
            ),
        }

    # 结构性纠偏：模型会把 qids 列表填进 paper_source（实测），
    # 这里认出 qids、并用它反查卷子，而不是拿 "2025OS_Q1,…" 当卷名去解析。
    raw_name, fixed_qids = quiz_service.normalize_grade_args(
        raw_name, args.get("qids"), session.get("current_paper") or "")
    if fixed_qids != (args.get("qids") or []):
        args = {**args, "qids": fixed_qids}
    paper, candidates = qbank.resolve_paper(raw_name, session.get("current_course"))
    if paper is None:
        if candidates:
            return {"error": f"没确定是哪份卷子: {raw_name}", "available_papers": candidates,
                    "_guide": "先确认学生在做哪份卷子。"}
        return {"error": "题库为空"}

    source = paper["source"]
    user_text = str(args.get("user_answer_text") or "").strip()
    if not user_text:
        return {"error": "学生答案为空", "_guide": "请学生把答案发过来。"}

    all_qs = qbank.paper_questions(source)
    by_qid = {q["qid"]: q for q in all_qs}
    # 题号以"题型内序号"为准（题库自带的 display_num 在不同题型间会重复）
    numbers = qbank.section_numbers(all_qs)

    # 批改逻辑在 quiz_service（与 tools.py 的 Plan/ReAct 工具共用一份实现）。
    # 这里只负责"这一侧特有"的东西：会话里的 current_qids 兜底、以及 _guide 文案。
    # 注意 fallback_qids 只在模型没传 qids 时才用 —— 学生可能是回头补做旧的一组。
    graded = quiz_service.grade_submission(
        paper, user_text,
        qids=args.get("qids"),
        fallback_qids=session.get("current_qids"),
    )
    notes = graded["notes"]

    if graded["total"] == 0:
        return {
            "total": 0,
            "parsed_answers": graded["parsed_answers"],
            "notes": notes,
            "_guide": ("没能从学生这段话里认出答案。可以请他用「题号+答案」的形式再说一次，"
                       "比如「1.A 2.B」，或者直接说不会。"),
        }

    return {
        "paper": source,
        "total": graded["total"],
        "correct": graded["correct"],
        "wrong": graded["wrong"],
        "needs_manual": graded["needs_manual"],
        "accuracy": graded["accuracy"],
        "results": graded["results"],
        "parse_notes": notes,
        "_guide": (
            "先报结果（对几题/错几题），再逐题讲。讲错题时**必须给出为什么**："
            "结合每条结果里的 reference_points 把依据/推导说清楚，"
            "只说「正确答案是 B」等于没讲。"
            "correct=null 表示题库参考答案缺失或不可靠，如实说明需要人工核对，不要硬判。"
            "如果某道题你手上的依据不够，用 explain_question 把题面和参考要点取来再讲。"
            "讲完问一句要不要继续下一组。"
            "学生要是说不会，就直接讲，不要只鼓励他再试试。"
        ),
    }


def _tool_prepare_custom_quiz(args, session):
    """为「临时出题」准备素材（不返回题目，题目由模型自己写）

    把"额外出题"做成一个正式能力而不是靠 prompt 求模型别用错工具：
      - 模型有了明确的落点，就不会去翻题库把计网题说成死锁题；
      - 出题素材来自课程知识库，出的题不会跑偏到别的课。

    实现已统一到 quiz_service（与 tools.py 的 Plan/ReAct 工具共用一份）。
    两边只有字段名不同：助教侧用 `_guide`（呈现契约），共享层用 `guide`。
    """
    payload = quiz_service.prepare_custom_quiz(
        topic=args.get("topic") or args.get("chapter") or "",
        course=args.get("course") or session.get("current_course") or "",
        qtype=args.get("qtype") or "选择题",
        count=args.get("count") or 5,
        difficulty=args.get("difficulty") or "",
    )
    result = dict(payload)
    result["_guide"] = result.pop("guide", "")
    return result


def _tool_explain_question(args, session):
    """讲题：把题面 + 参考答案 + 评分要点交给模型

    出题接口刻意不返回题面和答案（防泄题），代价是**模型手里没有题目，根本讲不了题** ——
    这正是"学生说不会，BOT 只能说些空话让他再试试"的根因。
    这个工具是"真要讲题"时的正当入口：只有确实要讲解时才取题面与答案。
    """
    raw_name = args.get("paper_source") or session.get("current_paper") or ""
    paper, candidates = qbank.resolve_paper(raw_name, session.get("current_course"))
    if paper is None:
        if candidates:
            return {"error": f"没确定是哪份卷子: {raw_name}", "available_papers": candidates,
                    "_guide": "先确认学生在做哪份卷子。"}
        return {"error": "题库为空"}

    # 取题面/答案的逻辑在 quiz_service（与 tools.py 共用一份）。
    # 区别在定位策略：这里知道"当前这一组"，所以先收敛到那一组；
    # 实在定位不到就返回错误让模型去问学生，而不是把整卷都倒出来。
    payload = quiz_service.explain_payload(
        paper,
        no=args.get("no"),
        qids=args.get("qids"),
        restrict_to=session.get("current_qids"),
        fallback_qids=session.get("current_qids"),
        whole_paper_if_unspecified=False,
    )
    if payload.get("error") == "题号在不同题型里重复，需要指明是哪一道":
        return {
            "error": payload["error"],
            "ambiguous": payload["ambiguous"],
            "_guide": ("这个题号同时对应多道题（比如选择题第 1 题和综合题第 1 题）。"
                       "把组内序号或 qids 说清楚再讲，不要猜。"),
        }
    if payload.get("error"):
        return {"error": payload["error"],
                "_guide": "先确认学生在做哪份卷子、哪一道题，或者直接问他卡在哪一步。"}

    return {
        "paper": payload["paper"],
        "questions": payload["questions"],
        "_guide": (
            "下面是题目原文、参考答案和参考要点。请把它**讲明白**，按这个顺序：\n"
            "① 这题在考什么（一句话点出考点）；\n"
            "② 正确答案为什么对 —— 把推导/依据一步步摆出来，别跳步；\n"
            "③ 其他选项或常见错法错在哪；\n"
            "④ 一句话记忆点或同类题的通用套路。\n"
            "学生说「不会」就是要你讲，不要只丢一个答案，也不要空泛地鼓励他再试试；\n"
            "如果只有一道题，可以先把思路和第一步给他，留最后一步让他自己走。\n"
            "参考答案存疑时如实说明是依据参考要点讲的，别硬编。"
        ),
    }


def _tool_get_practice_stats(args, session):
    source = args.get("paper_source", "") or session.get("current_paper", "")
    return {"paper": source, "practice_count": quiz_service.practice_count(source)}


def _tool_record_practice(args, session):
    source = args.get("paper_source", "") or session.get("current_paper", "")
    if not source:
        return {"error": "缺少 paper_source"}
    # 读-改-写在 quiz_service 里持锁（原来两边各写一份且无锁，
    # 实测 12 个并发调用只能记到 3 次）
    return {"paper": source, "practice_count": quiz_service.record_practice(source)}


def _tool_list_course_sections(args, session):
    """课程的素材地图（与 LangGraph 那条路共用 quiz_service.section_index）"""
    course = (args.get("course") or session.get("current_course") or "").strip()
    data = quiz_service.section_index(course)
    items = data.items() if not course else [(course, data)]
    lines = []
    for c, v in items:
        lines.append(f"## {c}")
        if v.get("note"):
            lines.append(v["note"])
        lines.append("### 课件（讲课时**优先用这些**，按课程顺序排好）")
        for i, s in enumerate(v.get("ppt", []), 1):
            lines.append(f"{i}. {s['source']}  ｜章节：{s['chapter']}｜约 {s['chunks']} 块")
        if v.get("ppt"):
            lines.append(f"→ 学生没说从哪章开始时，从第 1 份课件开始"
                         f"（《{v['ppt'][0]['source']}》）。")
        if v.get("others"):
            lines.append("### 其它资料（**只在课件没覆盖时**才用；课本章节顺序"
                         "不代表本课讲授顺序，不要拿它当「第一节」）")
            for s in v["others"]:
                lines.append(f"- {s['source']}  ｜{s['doc_type']}｜约 {s['chunks']} 块")
        if v.get("not_ingested"):
            lines.append("### 在磁盘上但**还没入库**（讲不了，需要先导入）")
            for n in v["not_ingested"]:
                lines.append(f"- {n}")
        lines.append("")
    # 注意：这里**故意不给 reply_text**。
    # 原来给了一份"排好版的目录"当成品文本，而模型（被我教过"工具给了成品文本就
    # 直接转发"）真的把它整段发给了学生 —— 学生说"我要复习"，收到的却是一张目录。
    # 地图是**给你看的内部资料**，不是回复内容。
    return {
        "material_map": "\n".join(lines),
        "sections": data,
        "_guide": (
            "上面是这门课的资料地图（**内部资料，不要把它作为回复内容发给学生**）。\n"
            "**现在立刻调 read_section(course=…, section=1) 取第 1 份课件的原文并开讲** —— "
            "最多用一句话告诉学生「这门课我按这几块带你过」，然后就开始讲第一节。\n"
            "⚠️ 学生**按章名**提问时（「复习传输层」「学死锁那一章」），"
            "必须用 `read_section(course=…, chapter=\"章名\")` —— "
            "**不要自己数编号**去猜 section。实测数错过：说传输层却读了网络层那章。\n"
            "**分节粒度由你判断**：地图只告诉你有哪些资料、各有多厚，"
            "不规定一节包含几个知识点；一节控制在学生一节课能消化的量。\n"
            "讲某一节时用 read_section 把课件原文取出来讲 —— "
            "**不要凭记忆讲这门课的内容**。\n"
            "一次只讲一节，讲完说「本节结束」，**不要**提前把后面的倒出来。\n"
            "定了这一节讲什么之后，用 record_study_position 记下位置。"
        ),
    }


def _tool_read_section(args, session):
    """按课件原始顺序读一节（与 LangGraph 那条路共用 quiz_service.study_read）"""
    mode = (args.get("mode") or session.get("study_mode") or "学习").strip()
    res = quiz_service.study_read(
        course=(args.get("course") or session.get("current_course") or "").strip(),
        chapter=args.get("chapter") or "",
        section=args.get("section") or 0,
        source=args.get("source") or "",
        offset=args.get("offset") or 0,
        limit=args.get("limit") or 15,
        mode=mode,
    )
    if res.get("error"):
        return {"error": res["error"],
                "_guide": "换个写法：read_section(course=\"操作系统\", chapter=\"死锁\") "
                          "或 read_section(course=\"操作系统\", section=1)。"}
    # 手写循环有 session 可以直接写，不必像 Plan 图那样从工具结果里"抽"
    study = dict(session.get("study") or {})
    study.update(res.get("study_state") or {})
    session["study"] = study
    session["study_mode"] = mode
    return {"text": res["text"], "study": study,
            "_guide": "按上面的原文讲解；讲完说「本节结束」，然后按模式出小测。"}


def _tool_record_study_position(args, session):
    """学习书签（只是记忆，不是状态机）"""
    course = (args.get("course") or "").strip()
    known = ("操作系统", "计算机网络")
    chapter = (args.get("chapter") or "").strip()
    if course and course not in known:
        # 实测模型会把章名填进 course —— 自我纠正，别污染书签
        for c in known:
            if quiz_service.find_chapter(c, course):
                chapter = chapter or course
                course = c
                break
    pos = {
        "mode": (args.get("mode") or session.get("study_mode") or "学习").strip(),
        "course": course,
        "chapter": chapter,
        "section": str(args.get("section") or "").strip(),
        "source": (args.get("source") or "").strip(),
        "status": (args.get("status") or "讲解中").strip(),
    }
    study = dict(session.get("study") or {})
    study.update({k: v for k, v in pos.items() if str(v or "").strip()})
    session["study"] = study
    if course:
        session["current_course"] = course
    session["study_mode"] = pos["mode"]
    guide = quiz_service.study_guide(pos["mode"])
    return {
        "recorded": pos,
        "_guide": (
            f"已记下：{pos['course']} · {pos['chapter'] or '（未标章）'} · {pos['section']}"
            f"（{pos['status']}）。\n" + guide
        ),
    }


TOOL_HANDLERS = {
    "search_knowledge_base": _tool_search_knowledge_base,
    "list_course_sections": _tool_list_course_sections,
    "read_section": _tool_read_section,
    "record_study_position": _tool_record_study_position,
    "list_papers": _tool_list_papers,
    "get_paper_questions": _tool_get_paper_questions,
    "prepare_custom_quiz": _tool_prepare_custom_quiz,
    "explain_question": _tool_explain_question,
    "grade_answers": _tool_grade_answers,
    "get_practice_stats": _tool_get_practice_stats,
    "record_practice": _tool_record_practice,
}


# ============================================================
# 4. 阅卷（填空题 / 综合题）
# ============================================================




# ============================================================
# 5. 智谱 API 调用（支持流式 + function calling）
# ============================================================

_HEADERS = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}


def _post_stream(payload: dict, timeout: int = 180):
    """带 429 退避的流式 POST"""
    last = None
    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=_HEADERS, json=payload,
                                 stream=True, timeout=timeout)
            if resp.status_code == 429:
                raise requests.exceptions.HTTPError("429 rate limited")
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            last = e
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            raise
        except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
            last = e
            if attempt < 2:
                time.sleep(2)
                continue
            raise
    raise last


def _tools_for(session: dict) -> list:
    """按会话状态决定**这一轮给模型看哪些工具**

    这是最强的一种纠偏：**不给它，它就没法调错**。

    起因（实测）：在讲课/复习中，学生交了自拟小测的答案，模型**习惯性**去调
    `grade_answers`，然后反问"你在做哪份卷子" —— 把学习流程硬拽回刷题流程。
    我先在 system prompt 里写了"不要用"，它不跟；又在工具入口加了硬守卫
    （返回"这是小测，你自己批"），**它还是不跟，照样问学生卷名**。
    所以在"讲课中且没有当前试卷"这种状态下，干脆不提供这些题库工具。
    """
    study = session.get("study") or {}
    quiz_like = {"grade_answers", "explain_question", "get_paper_questions",
                 "get_practice_stats", "record_practice"}
    if study.get("course") and not session.get("current_paper"):
        return [t for t in TOOLS if t["function"]["name"] not in quiz_like]
    return TOOLS


def _stream_round(messages, tools=None, temperature=0.7, max_tokens=2000):
    """跑一轮 LLM，边收边产出事件

    yield: ("content", 文本片段) | ("tool_calls", [...])
    流式解析 tool_calls 的 delta 分片，拼成完整调用。
    """
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": True,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = "auto"

    resp = _post_stream(payload)
    calls: dict[int, dict] = {}
    try:
        for raw in resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except json.JSONDecodeError:
                continue
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta") or {}

            piece = delta.get("content")
            if piece:
                yield ("content", piece)

            for tc in (delta.get("tool_calls") or []):
                idx = tc.get("index", 0)
                slot = calls.setdefault(idx, {"id": "", "type": "function",
                                              "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["function"]["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["function"]["arguments"] += fn["arguments"]
    finally:
        resp.close()

    if calls:
        ordered = [calls[i] for i in sorted(calls)]
        for n, c in enumerate(ordered):
            c["id"] = c["id"] or f"call_{n}"
        yield ("tool_calls", ordered)





# ============================================================
# 6. 会话管理
# ============================================================

_SESSIONS: dict[str, dict] = {}
_SESSION_TTL = 3 * 3600          # 3 小时不活跃就回收，避免内存无限增长
_MAX_HISTORY_MESSAGES = 40       # 超过就按"从某条 user 消息起"截断（保证 tool 配对完整）
_COMPACT_KEEP_LAST_TOOL = 2      # 只保留最近 2 条工具结果全文，更早的压缩掉


def _system_prompt_for(session: dict) -> str:
    """基础 system prompt + **动态的当前状态**

    为什么动态注入而不写进基础 prompt：
      · 基础 prompt 有长度约束（有测试卡着 1200 字符），堆细节会让模型变没主见；
      · 学习书签是**跨轮会话状态**，只有这里能按状态给出。

    尤其是"本节小测由你自己批"这一条：它必须**在学生交答案那一轮**就可见 ——
    而那轮学生不会去调 read_section，所以那个工具返回里的契约根本到不了模型手里。
    （实测：学生交了自拟小测的答案，助教反问「你想做哪份模拟卷」，
      把学习流程硬拽回了刷题流程。）
    """
    extra = []
    study = session.get("study") or {}
    if study.get("course"):
        line = (f"【当前学习位置】{study.get('mode', '学习')} · {study['course']}"
                + (f" · {study['chapter']}" if study.get("chapter") else "")
                + f" · 上一节《{study.get('section', '')}》"
                  f"({study.get('status', '讲解中')})")
        if study.get("last_source"):
            line += f" · 刚读到《{study['last_source']}》的 {study.get('last_read', '')}"
            line += (f"，**继续讲要用 offset={study['next_offset']}**"
                     if study.get("next_offset") is not None else "（这份已读完）")
        extra.append(
            line + "\n"
            "· **本节的测验题是你自己出的，所以由你自己批**：逐题给 ✓/✗ 与讲解，"
            "说清**错在哪、为什么错、正确答案是什么**，错的知识点要重讲一遍。\n"
            "· **不要**调 grade_answers / get_paper_questions —— 那是题库真题的批改通道，"
            "学生答的是你刚出的小测，题库里没有它。\n"
            "· 学生说「继续讲」＝接着读**同一份**课件；说「下一节」＝进下一份。\n"
            "· 学生**按章名**提问（「复习传输层」「学死锁那一章」）→ 用 "
            "read_section(course=…, chapter=\"章名\")，**不要自己数编号**去猜 section"
            "（实测数错过：说传输层却读了网络层那章）。\n"
            "· 学生说「没听懂 / 换个讲法」→ **用 read_section 把那段原文再取一次**"
            "（同一个 source），换个角度讲（类比、画流程、举个具体例子），"
            "**不要**让学生报题号、也不要复读上一轮的原话。\n"
            "· 切换课程/章节时明确说清现在在讲哪门课、哪一章、进到哪了，**不串课**。"
        )
    if session.get("current_paper"):
        extra.append(f"【当前试卷】{session['current_paper']}"
                     + (f"（第 {session['last_group']['index']} 组）"
                        if session.get("last_group") else ""))
    return SYSTEM_PROMPT + ("\n\n" + "\n".join(extra) if extra else "")


def _new_session(sid: str) -> dict:
    return {
        "id": sid,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}],
        "current_paper": None,
        "current_course": None,
        "current_qids": [],
        "last_group": None,
        # 工具需要知道学生"这句话"说了什么，才能判断他是在继续做题
        # 还是在要新题（get_paper_questions 的范围守卫要用）
        "last_user_message": "",
        "touched": time.time(),
    }


def _sweep_sessions():
    now = time.time()
    for sid in [k for k, v in _SESSIONS.items() if now - v.get("touched", 0) > _SESSION_TTL]:
        _SESSIONS.pop(sid, None)


def _get_session(session_id: str | None) -> dict:
    _sweep_sessions()
    if session_id and session_id in _SESSIONS:
        s = _SESSIONS[session_id]
        s["touched"] = time.time()
        return s
    sid = session_id or str(uuid.uuid4())
    s = _new_session(sid)
    _SESSIONS[sid] = s
    return s


def _compact_tools(session: dict):
    """把较早的工具结果压成占位符

    一次 get_paper_questions 的结果可能上万字符（整卷题目），
    不清掉的话跑三四个回合就会顶爆上下文窗口。
    注意不能删除 tool 消息本身——assistant 的 tool_calls 必须一一对应，
    所以只替换 content。
    """
    tool_idxs = [i for i, m in enumerate(session["messages"]) if m.get("role") == "tool"]
    if len(tool_idxs) <= _COMPACT_KEEP_LAST_TOOL:
        return
    for i in tool_idxs[:-_COMPACT_KEEP_LAST_TOOL]:
        m = session["messages"][i]
        if len(m.get("content", "")) > 200:
            m["content"] = json.dumps({"note": "（这一步的工具结果已省略，避免上下文过长）"},
                                      ensure_ascii=False)


def _trim_history(session: dict):
    """保留 system + 最近若干条消息

    只在 user 消息处切开：一次对话轮次必然是 user → assistant/tool → ... → user，
    从 user 处切分不会留下"孤立的 tool 消息"（那会被 API 判为 400）。
    """
    msgs = session["messages"]
    if len(msgs) <= _MAX_HISTORY_MESSAGES:
        return
    cut = len(msgs) - _MAX_HISTORY_MESSAGES
    while cut < len(msgs) and msgs[cut].get("role") != "user":
        cut += 1
    if cut < len(msgs):
        session["messages"] = msgs[:1] + msgs[cut:]


# ============================================================
# 7. 主循环
# ============================================================

_MAX_TOOL_ROUNDS = 6


def process_message(session_id: str | None, user_input: str):
    """处理一条用户消息，yield SSE 事件

    事件类型：
      start          会话开始
      phase          进度提示（思考中/调用工具/回复中）
      content        正文片段（流式）
      discard        丢弃本轮已推给前端的正文（说明模型其实要调工具，刚才是前言）
      tool           工具执行摘要
      done           结束，带最终回答
      error          出错
    """
    session = _get_session(session_id)
    sid = session["id"]

    # 每轮刷新 system prompt 里的动态状态（学习位置 / 当前试卷）
    # —— 基础 prompt 保持简短，状态按会话实时注入。
    if session["messages"] and session["messages"][0].get("role") == "system":
        session["messages"][0]["content"] = _system_prompt_for(session)

    yield {"type": "start", "session_id": sid}

    _compact_tools(session)
    _trim_history(session)
    session["messages"].append({"role": "user", "content": user_input})
    session["last_user_message"] = user_input

    final_text = ""
    called_tools: list[str] = []
    nudged = False

    for round_idx in range(_MAX_TOOL_ROUNDS):
        yield {"type": "phase", "phase": "thinking", "message": "🧠 思考中..."}

        content_parts: list[str] = []
        tool_calls = []
        streamed_any = False

        try:
            for kind, payload in _stream_round(session["messages"],
                                               tools=_tools_for(session)):
                if kind == "content":
                    content_parts.append(payload)
                    streamed_any = True
                    yield {"type": "content", "content": payload}
                elif kind == "tool_calls":
                    tool_calls = payload
        except Exception as e:
            yield {"type": "error", "msg": f"模型调用失败: {str(e)[:200]}"}
            return

        content = "".join(content_parts).strip()

        # 没有工具调用 → 通常这就是最终回答
        if not tool_calls:
            # 但有一种情况必须纠偏：学生说"题目没给我"，模型却只在嘴上答应
            # （实测：「很抱歉…我将重新发送这组题目」，然后一条题都没发，也没调工具）。
            # 只重试一次，且只在"确实有当前这一组、这一轮又没调过出题工具"时触发。
            if (not nudged
                    and _missed_questions(user_input)
                    and session.get("last_group")
                    and "get_paper_questions" not in called_tools):
                nudged = True
                if streamed_any:
                    yield {"type": "discard"}
                last = session["last_group"]
                session["messages"].append({
                    "role": "system",
                    "content": (
                        "注意：学生说没有收到题目。不要只是口头道歉或承诺，"
                        "现在立刻调用 get_paper_questions，"
                        f"paper_source=\"{last.get('source', '')}\"，"
                        f"group_index={last.get('index', 1)}"
                        "（同一组，不要往下走），并把返回的 reply_text 完整发给学生。"
                    ),
                })
                continue
            final_text = content
            if not streamed_any and content:
                yield {"type": "content", "content": content}
            break

        # 有工具调用 → 刚才流出去的内容只是"前言"，让前端丢掉
        if streamed_any:
            yield {"type": "discard"}

        session["messages"].append({
            "role": "assistant",
            "content": content,
            "tool_calls": tool_calls,
        })

        for tc in tool_calls:
            fn = tc.get("function", {}) or {}
            name = fn.get("name", "")
            tc_id = tc.get("id", "")
            raw_args = fn.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                if not isinstance(args, dict):
                    args = {}
            except json.JSONDecodeError:
                args = {}

            yield {"type": "phase", "phase": "tool", "message": f"🔧 {_tool_label(name, args)}"}
            called_tools.append(name)

            handler = TOOL_HANDLERS.get(name)
            if handler is None:
                result = {"error": f"未知工具: {name}"}
            else:
                try:
                    result = handler(args, session)
                except Exception as e:
                    result = {"error": f"{type(e).__name__}: {str(e)[:200]}"}

            yield {"type": "tool", "tool": name, "result": _tool_summary(name, result)}

            session["messages"].append({
                "role": "tool",
                "tool_call_id": tc_id,
                "content": json.dumps(result, ensure_ascii=False),
            })
    else:
        # 工具轮数用尽：不要给一句干巴巴的兜底，再要一次纯文本回答
        try:
            parts = []
            for kind, payload in _stream_round(session["messages"], tools=None):
                if kind == "content":
                    parts.append(payload)
                    yield {"type": "content", "content": payload}
            final_text = "".join(parts).strip()
        except Exception as e:
            final_text = "抱歉，这次处理超时了，你可以再说一次想要什么吗？"
            yield {"type": "content", "content": final_text}
            yield {"type": "error", "msg": str(e)[:200]}

    if not final_text:
        final_text = "我在的，你想学点什么或者做哪套题？"
        yield {"type": "content", "content": final_text}

    session["messages"].append({"role": "assistant", "content": final_text})
    _compact_tools(session)
    _trim_history(session)
    session["touched"] = time.time()

    yield {"type": "done", "answer": final_text, "session_id": sid}


def _tool_label(name: str, args: dict) -> str:
    if name == "search_knowledge_base":
        return f"检索知识库：{args.get('query', '')[:24]}"
    if name == "list_papers":
        return f"查看题库{('（' + args['course'] + '）') if args.get('course') else ''}"
    if name == "get_paper_questions":
        return f"出题：{args.get('paper_source', '')}"
    if name == "prepare_custom_quiz":
        return f"准备自拟题素材：{args.get('topic', '') or '按上下文'}"
    if name == "explain_question":
        nos = args.get("no")
        return f"取题面讲解：{nos if nos else '当前这组'}"
    if name == "grade_answers":
        return "批改答案"
    if name == "get_practice_stats":
        return "查询练习记录"
    if name == "record_practice":
        return "记录练习"
    return name


def _tool_summary(name: str, result) -> str:
    """给前端"思维过程"面板用的一句话摘要（不暴露答案）"""
    if not isinstance(result, dict):
        return "完成"
    if result.get("error"):
        return f"出错：{str(result['error'])[:60]}"
    if name == "list_papers":
        return f"{result.get('total_papers', 0)} 份试卷"
    if name == "get_paper_questions":
        return f"{result.get('paper', '')} {result.get('progress', '')}"
    if name == "prepare_custom_quiz":
        n = len(result.get("material") or [])
        return f"出题素材 {n} 段（{result.get('topic', '')}）"
    if name == "explain_question":
        n = len(result.get("questions") or [])
        return f"取出 {n} 道题的题面与参考要点"
    if name == "grade_answers":
        return f"批改 {result.get('total', 0)} 题，正确 {result.get('accuracy', '-')}"
    if name == "search_knowledge_base":
        return f"命中 {result.get('count', 0)} 个片段"
    return "完成"
