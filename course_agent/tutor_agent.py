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
- 回答用 Markdown，语气自然，别把内部规则念出来。"""


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
    kb = get_knowledge_base()
    top_k = int(args.get("top_k") or 5)
    chapter = args.get("chapter")

    if course:
        hits, confirmed = quiz_service.search_with_course(kb, query, course,
                                                             top_k=top_k)
        note = "" if confirmed else f"（限定课程「{course}」没有真正相关的片段，下面是全库检索结果）"
    else:
        hits = kb.search(query, top_k=top_k, chapter=chapter)
        note = ""

    return {
        "query": query,
        "course": course or "全部",
        "note": note,
        "count": len(hits),
        "passages": [
            {"chapter": r.get("chapter", ""), "source": r.get("source", ""),
             "text": r.get("text", "")[:700]}
            for r in hits
        ],
        "_guide": (
            "基于这些片段用自己的话讲解；引用到具体章节时可以点名，不要整段照抄。"
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
    raw_name = args.get("paper_source", "") or session.get("current_paper") or ""
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


TOOL_HANDLERS = {
    "search_knowledge_base": _tool_search_knowledge_base,
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
            for kind, payload in _stream_round(session["messages"], tools=TOOLS):
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
