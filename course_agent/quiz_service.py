"""刷题相关的共享业务逻辑 —— 两个 Agent 共用同一份实现

## 为什么要有这个模块

`tools.py`（LangChain 工具，给 Plan/ReAct 图用）和 `tutor_agent.py`
（手写循环用的工具层）曾经**各写一份**刷题逻辑。两份已经真实分叉：

  · tools.py 的阅卷既没带 `basis_warning`（参考答案存疑时学生会看到一句
    "正确答案就是这样"），又没把题号传给 `qbank.build_llm_basis`
    （于是题号可能算错）；
  · tutor 那边两份都有。

这类"修一份忘另一份"的重复在历史上已经咬过一次：`tools.py` 的
`get_paper_questions` 长期把 `group_index` 硬编码成 1，而文档写着"取下一组"。

## 边界

这里**不碰会话状态**（没有 session 参数）：
游标、当前卷子、当前 qids 这些"每个运行面各不相同"的东西留在调用方。
本模块只接受显式参数，返回中立的 dict；
两边的"外壳差异"（一边返回 JSON 字符串 + 显式参数，一边返回 dict + `_guide` + 会话游标）
由各自包装函数负责。
"""

import json
import re
import threading
import time

import requests

from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL, ZHIPU_MODEL, RELEVANCE_THRESHOLD
from course_agent import qbank
from course_agent.logger import logger


# ============================================================
# 1. 底层 LLM JSON 调用（阅卷用）
# ============================================================

_HEADERS = {"Authorization": f"Bearer {ZHIPU_API_KEY}", "Content-Type": "application/json"}

GRADE_SYSTEM = """你是大学课程的阅卷老师。

下面会给你若干道题，每题包含题干、参考答案、评分要点和小问答案。
请依据参考内容判断学生作答，逐题给出结论。

输出严格的 JSON（不要加解释文字）：
{
  "items": [
    {
      "no": 题号(整数),
      "correct": true/false,
      "score": "得分描述，如 8/10",
      "comment": "50字以内的点评：对在哪、错在哪",
      "missing": ["学生漏掉的要点"]
    }
  ]
}

判定原则：
- 只要核心要点答到就算对，表述不同不扣分。
- 参考答案缺失（answer 为空）时，以 reference_points / sub_questions 为评分依据。
"""


def chat_json(system: str, user: str, max_tokens: int = 1500) -> dict:
    """非流式调用并解析 JSON（带 429 退避重试）"""
    payload = {
        "model": ZHIPU_MODEL,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0.1,
        "max_tokens": max_tokens,
    }
    last = None
    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=_HEADERS, json=payload, timeout=120)
            resp.raise_for_status()
            text = resp.json()["choices"][0]["message"]["content"].strip()
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                raise ValueError("阅卷返回内容里没有 JSON")
            return json.loads(m.group())
        except Exception as e:
            last = e
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            if attempt < 2:
                time.sleep(1.5)
                continue
            raise last
    raise last


# ============================================================
# 2. 题库解析辅助
# ============================================================

def resolve_paper(name: str, course_hint: str = None):
    """解析卷名，返回 (paper, candidates)"""
    return qbank.resolve_paper(name or "", course_hint)


def paper_index(source: str):
    """取一份卷子的题、题型内序号、qid 索引

    题号一律用 `section_numbers()`（题型内序号）：题库自带的 display_num
    在不同题型之间会重复，直接拿来当题号会串题型。
    """
    all_qs = qbank.paper_questions(source)
    return all_qs, qbank.section_numbers(all_qs), {q["qid"]: q for q in all_qs}


def _select_scope(all_qs, by_qid, numbers, qids=None, fallback_qids=None, text=""):
    """决定这次批改/讲解覆盖哪几道题，并解析学生作答

    优先级：显式 qids > 会话里的 current_qids（fallback）> 整卷按题型分段解析。
    """
    wanted = [str(x) for x in (qids or []) if str(x) in by_qid]
    if wanted:
        questions = [by_qid[x] for x in wanted]
        questions.sort(key=qbank._sort_key)
        return questions, qbank.parse_submission(text, questions, numbers)

    cur = [str(x) for x in (fallback_qids or []) if str(x) in by_qid]
    if cur:
        questions = [by_qid[x] for x in cur]
        questions.sort(key=qbank._sort_key)
        return questions, qbank.parse_submission(text, questions, numbers)

    return all_qs, qbank.parse_against_paper(text, all_qs)


def grade_with_llm(items: list) -> list:
    """填空题/综合题由阅卷模型评分

    items: [(question, 学生作答, 预生成的 result dict), ...]

    出错时结果标成 correct=None（需人工核对），**绝不能静默当成"答错"** ——
    原实现 except 后直接 return False，任何一次网络抖动都会让学生白丢分。
    """
    payload_items = []
    for q, ans, r in items:
        # 题号必须从 result 里取：那是按"题型内序号"算好的，
        # 直接调 build_llm_basis(q) 会退化成 q_num/display_num，题号可能对不上。
        basis = qbank.build_llm_basis(q, no=r.get("no"))
        payload_items.append({
            "no": basis["no"],
            "type": basis["type"],
            "题干": basis["text"][:1200],
            "参考答案": basis["answer"],
            "参考答案不可靠": basis["answer_unreliable"],
            "评分要点": basis["reference_points"][:12],
            "小问答案": basis["sub_questions"][:12],
            "学生作答": ans[:1500],
        })

    results_by_no = {}
    try:
        data = chat_json(GRADE_SYSTEM,
                         "题目列表：\n" + json.dumps(payload_items, ensure_ascii=False, indent=1),
                         max_tokens=1800)
        for it in (data.get("items") or []):
            if isinstance(it, dict) and it.get("no") is not None:
                results_by_no[int(it["no"])] = it
    except Exception as e:
        logger.error("阅卷模型调用失败: %s: %s", type(e).__name__, str(e)[:200])
        results_by_no = {}

    out = []
    for q, ans, r in items:
        judged = results_by_no.get(r.get("no"))
        if judged is None:
            r.update(correct=None, needs_manual=True,
                     explanation="这题需要人工核对（自动阅卷没给出结果）。")
        else:
            r["correct"] = bool(judged.get("correct"))
            r["score"] = judged.get("score", "")
            r["explanation"] = judged.get("comment", "")
            r["missing"] = judged.get("missing", [])
            # 参考答案存疑时必须说出来，否则学生会以为"标准答案就是这样"
            if r.get("basis_warning"):
                r["explanation"] = f"{r['explanation']}（{r['basis_warning']}）"
        r.pop("needs_llm", None)
        out.append(r)
    return out


# ============================================================
# 3. 批改
# ============================================================

def grade_submission(paper: dict, user_answer_text: str,
                     qids=None, fallback_qids=None) -> dict:
    """批改一份作答，返回中立的评分结果

    Returns:
        {total, correct, wrong, needs_manual, accuracy, results, notes, parsed_answers}
        results 为空表示"没认出任何答案"（调用方负责引导用户换个写法）。
    """
    all_qs, numbers, by_qid = paper_index(paper["source"])
    questions, (answer_map, notes) = _select_scope(
        all_qs, by_qid, numbers, qids=qids, fallback_qids=fallback_qids,
        text=user_answer_text,
    )

    results, need_llm = [], []
    for q in questions:
        ans = answer_map.get(q["qid"])
        if ans is None or ans == "":
            continue
        r = qbank.grade_answer(q, ans, no=numbers.get(q["qid"]))
        if r.get("needs_llm"):
            need_llm.append((q, ans, r))
        else:
            results.append(r)

    if need_llm:
        results.extend(grade_with_llm(need_llm))

    correct = sum(1 for r in results if r.get("correct") is True)
    manual = sum(1 for r in results if r.get("correct") is None)
    results.sort(key=lambda r: r.get("no") or 0)

    return {
        "total": len(results),
        "correct": correct,
        "wrong": len(results) - correct - manual,
        "needs_manual": manual,
        "accuracy": f"{correct}/{len(results)}" if results else "0/0",
        "results": results,
        "notes": notes,
        "parsed_answers": answer_map,
    }


# ============================================================
# 4. 讲题
# ============================================================

EXPLAIN_GUIDE = (
    "按这个顺序讲：① 这题在考什么；② 正确答案为什么对（推导/依据一步步来，别跳步）；"
    "③ 其他选项或常见错法错在哪；④ 一句话记忆点。"
    "学生说不会就是要你讲，不要只丢一个答案，也不要空泛地让他再试试。"
    "参考答案存疑时如实说明是依参考要点讲的，别硬编。"
)


def explain_payload(paper: dict, no=None, qids=None, restrict_to=None,
                    fallback_qids=None, whole_paper_if_unspecified: bool = True) -> dict:
    """取题面 + 参考答案 + 参考要点，供讲题用

    定位顺序：显式 qids > 题号（可限制在 restrict_to 里找）> fallback_qids > 整卷。

    两个运行面在这里的默认行为不同，所以做成参数：
      · Plan/ReAct 工具没有会话状态，"没说讲哪道"就只能给整卷
        （whole_paper_if_unspecified=True）；
      · 手写助教知道"当前这一组"，所以先收敛到那一组，实在不知道就返回
        "不知道要讲哪道题"让模型去问（whole_paper_if_unspecified=False）。
    """
    all_qs, numbers, by_qid = paper_index(paper["source"])

    wanted = [str(x) for x in (qids or []) if str(x) in by_qid]
    ambiguous = {}
    if not wanted:
        wanted, ambiguous = qbank.resolve_question_numbers(
            all_qs, numbers, no, restrict_to=restrict_to)
    if ambiguous:
        return {
            "error": "题号在不同题型里重复，需要指明是哪一道",
            "ambiguous": {str(n): qs for n, qs in ambiguous.items()},
            "提示": "这个题号同时对应多道题（如选择题第 1 题和综合题第 1 题），请用 qids 指定。",
        }

    if not wanted and fallback_qids:
        wanted = [str(x) for x in fallback_qids if str(x) in by_qid]

    if not wanted and not whole_paper_if_unspecified:
        return {"error": "不知道要讲哪道题"}

    scope = [by_qid[x] for x in wanted] if wanted else all_qs
    scope.sort(key=qbank._sort_key)

    details = []
    for q in scope:
        b = qbank.build_llm_basis(q, no=numbers.get(q["qid"]))
        details.append({
            "no": b["no"], "type": b["type"], "topic": q.get("topic", ""),
            "题目原文": b["text"], "参考答案": b["answer"],
            "参考要点": b["reference_points"], "小问答案": b["sub_questions"],
            "参考答案存疑": b["answer_unreliable"],
            "image_url": qbank.image_url(q),
        })

    return {"paper": paper["source"], "questions": details, "讲解要求": EXPLAIN_GUIDE}


# ============================================================
# 5. 出题 / 列卷
# ============================================================

def group_payload(paper: dict, qtype: str = None, group_index: int = None,
                  cursor: dict = None) -> dict:
    """取一组题

    Args:
        cursor: 可选的会话游标 {"source","type","index"}；用于"没指定组号时接着上次"。
                会话状态由调用方持有，这里只做无状态的推导。
    Returns:
        {error} 或 {finished} 或 {paper,type,group_index,group_count,progress,
                              questions,qids,has_more,next_group_index,reply_text}
    """
    source = paper["source"]
    qtype = (qtype or "").strip() or None

    if group_index in (None, "", 0):
        same = bool(cursor) and cursor.get("source") == source and \
            (qtype is None or cursor.get("type") == qtype)
        group_index = (int(cursor.get("index", 0)) + 1) if same else 1
    try:
        group_index = max(1, int(group_index))
    except (TypeError, ValueError):
        group_index = 1

    group = qbank.get_group(source, qtype=qtype, group_index=group_index)
    if group.get("error") or group.get("finished"):
        return group

    shown = "\n".join(f"{q['no']}. {q['topic']}  ![题目]({q['image_url']})"
                      for q in group["questions"])
    reply_text = (
        f"这是《{source}》的{group['type']}，{group['progress']}：\n\n"
        f"{shown}\n\n"
        "请慢慢做。答案怎么写都行 —— 比如 `1.D 2.B 3.D`、`DBDAC`、`全都选A`，"
        "直接把选项发给我就行，**不用照上面的格式**，我来批改。"
    )
    return {
        "paper": source,
        "type": group["type"],
        "group_index": group_index,
        "group_count": group["group_count"],
        "progress": group["progress"],
        "questions": group["questions"],
        "qids": group["qids"],
        "has_more": group["has_more"],
        "next_group_index": group_index + 1 if group["has_more"] else None,
        "reply_text": reply_text,
        "render_example": shown,
    }


def papers_payload(course: str = None) -> list:
    """按课程列出可用试卷"""
    return qbank.list_papers(course or None)


def papers_text(course: str = None) -> str:
    """把试卷列表渲染成一段可直接发给学生的文本"""
    papers = papers_payload(course)
    if not papers:
        return "没有找到符合条件的试卷。"
    lines = []
    for i, p in enumerate(papers, 1):
        outline = qbank.paper_outline(p["source"])
        type_str = "、".join(f"{s['type']}{s['count']}题（{s['group_count']}组）"
                            for s in outline["sections"])
        lines.append(f"{i}. {p['source']}（{p['course']}，共{p['total']}题：{type_str}）")
    return "\n".join(lines)


# ============================================================
# 6. 课程范围的检索（两个 Agent 共用的判断）
# ============================================================

_COURSE_SOURCE_CACHE: dict[str, list[str]] = {}


def course_sources(course: str) -> list[str]:
    """把课程名映射成知识库里的文件名列表

    知识库的 chunk metadata 只有 source（文件名）、doc_type、chapter，没有课程字段，
    所以按 data/<课程>/ 目录下的文件名来反查。
    """
    if course in _COURSE_SOURCE_CACHE:
        return _COURSE_SOURCE_CACHE[course]
    root = qbank.PROJECT_ROOT / "data" / course
    names: list[str] = []
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in (".pdf", ".docx", ".md", ".txt", ".pptx"):
                names.append(p.name)
    _COURSE_SOURCE_CACHE[course] = names
    return names


def search_with_course(kb, query: str, course: str, top_k: int = 6):
    """先按课程检索，判断这个话题是否真的落在这门课里

    注意不能拿"结果非空"当命中：`kb.search()` 对非空知识库**永远**会返回 top_k 条，
    所以只要库里有东西，按课程过滤就"总能命中" —— 于是学生在做计网时问「死锁」，
    拿回来的会是网络层/应用层的材料，模型再把它们当成死锁的素材。
    必须用**绝对相似度**来判断。

    Returns:
        (hits, course_confirmed)
    """
    if not course:
        return kb.search(query, top_k=top_k), False

    sources = course_sources(course)
    hits = kb.search(query, top_k=top_k, sources=sources) if sources else []
    confirmed = bool(hits) and hits[0].get("score", 0) >= RELEVANCE_THRESHOLD
    if confirmed:
        return hits, True
    # 话题不在当前课程里 → 退回全库，并让调用方知道课程标签要作废
    return kb.search(query, top_k=top_k), False


# ============================================================
# 7. 临时出题（不返回题目，返回出题素材）
# ============================================================

def prepare_custom_quiz(topic: str = "", course: str = "", qtype: str = "",
                        count=5, difficulty: str = "") -> dict:
    """为"临时出题"准备素材 —— 出题本身由模型完成

    为什么要有它："另外出几道题"是一个真实需求，但它**不是**从题库取题。
    如果没有这个明确的落点，模型就会去翻题库、把计网的选择题标成"死锁的定义"
    （实测发生过）。
    """
    topic = (topic or "").strip()
    course = (course or "").strip()
    qtype = (qtype or "").strip() or "选择题"
    try:
        count = int(count or 5)
    except (TypeError, ValueError):
        count = 5
    count = min(max(count, 1), 20)
    difficulty = (difficulty or "").strip()

    passages = []
    course_confirmed = False
    if topic:
        try:
            from rag_tool import get_knowledge_base

            kb = get_knowledge_base()
            hits, course_confirmed = search_with_course(kb, topic, course, top_k=6)
            passages = [
                {"chapter": h.get("chapter", ""), "source": h.get("source", ""),
                 "text": h.get("text", "")[:800]}
                for h in hits
            ]
        except Exception as e:
            logger.warning("出题素材检索失败: %s", e)

    effective_course = course if (course and course_confirmed) else ""
    if course and not course_confirmed:
        course_note = (f"注意：这个话题在「{course}」的资料里没找到，"
                       f"题目请按「{topic}」本身所属的课程来出，不要硬说成是{course}的内容。")
    else:
        course_note = ""

    difficulty_note = f"难度按「{difficulty}」把握。" if difficulty else ""
    if passages:
        material_hint = (f"上面 {len(passages)} 段材料就是出题依据，"
                         f"请确保每道题都能在材料里找到对应知识点。")
    else:
        material_hint = ("这次没检索到材料（话题可能太窄或知识库没覆盖）。"
                         "那就凭你自己的课程知识出题，并在题前说清楚这是自拟题，"
                         "不要声称来自题库。")

    return {
        "topic": topic or "（未指定，按学生最近提到的内容把握）",
        "course": effective_course or "（未指定）",
        "qtype": qtype,
        "count": count,
        "difficulty": difficulty or "默认",
        "material": passages,
        "guide": (
            f"现在请自己写 {count} 道关于「{topic or '该主题'}」的{qtype}，"
            f"题号从 1 开始。{difficulty_note}{material_hint}{course_note}\n"
            "格式：题号 + 题干（选择题要列出选项）。"
            "答案先不要写在题目里，等学生做完再由你批改。"
            "不要调用题库工具，也不要把题目说成是某份试卷里的真题。"
        ),
    }


# ============================================================
# 8. 练习统计（读-改-写必须加锁）
# ============================================================

_STATS_FILE = qbank.PROJECT_ROOT / "data" / "question_bank" / "practice_stats.json"
_STATS_LOCK = threading.Lock()


def _read_stats() -> dict:
    if not _STATS_FILE.exists():
        return {}
    try:
        return json.loads(_STATS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def practice_count(paper_source: str) -> int:
    with _STATS_LOCK:
        return int(_read_stats().get(paper_source, 0))


def record_practice(paper_source: str) -> int:
    """练习次数 +1

    必须持锁：原来是"读 JSON → 改 → 写回"，没有任何互斥，
    实测 12 个并发调用只能记到 3 次（读-改-写互相覆盖）。
    """
    with _STATS_LOCK:
        stats = _read_stats()
        stats[paper_source] = int(stats.get(paper_source, 0)) + 1
        _STATS_FILE.parent.mkdir(parents=True, exist_ok=True)
        _STATS_FILE.write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                               encoding="utf-8")
        return stats[paper_source]
