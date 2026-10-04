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


# qid 形如 2025OS_Q1 / 2019网模拟_F3 / 2025OS_S1（卷前缀_题型字母+序号）
_QID_RE = re.compile(r"([A-Za-z0-9\u4e00-\u9fff]{1,24}_[QSPF]\d{1,3})")


def qids_in_text(text) -> list:
    """从一段文字里抽出所有 qid

    为什么需要它：实测 planner 会把 qids 列表**填进 paper_source**
    （我给它的提示只说「用上面的 qids 调 grade_answers」，没说填哪个参数），
    于是 grade_answers 拿 "2025OS_Q1,2025OS_Q2,…" 当卷名去解析，直接失败。
    """
    seen, out = set(), []
    for m in _QID_RE.finditer(str(text or "")):
        q = m.group(1)
        if q not in seen:
            seen.add(q)
            out.append(q)
    return out


def paper_of_qids(qids) -> dict:
    """用 qid 反查它属于哪份卷子（qid 前缀就是卷子标识）

    这样即使模型把 qids 填错了参数，也能**确定性地**把卷子找回来，
    而不是让它回头去问学生「你要批改哪一份卷子」。
    """
    want = set(qids or [])
    if not want:
        return None
    for q in (qbank.load_questions() or []):
        if q.get("qid") in want:
            return {"source": q.get("source", ""), "course": q.get("course", "")}
    return None


def normalize_grade_args(paper_source: str, qids=None, session_paper: str = ""):
    """把模型填错的批改参数**就地纠正**，返回 (paper_source, qids)

    这是「结构性纠偏」，不是提示词纠偏 —— 实测提示词写「不要填错参数」基本无效，
    而在工具入口把 qids 认出来、把卷子找回来，是确定有效的。
    """
    qids = [q for q in (qids or []) if q]
    found = qids_in_text(paper_source)
    if found:
        # paper_source 里装的是 qid 列表 → 它其实想传的是 qids
        for q in found:
            if q not in qids:
                qids.append(q)
        paper_source = ""
    if not paper_source and not session_paper and qids:
        hit = paper_of_qids(qids)
        if hit:
            paper_source = hit["source"]
    return paper_source or session_paper, qids


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


def section_index(course: str = "") -> dict:
    """把课程的资料构成整理成"章节地图"，供学习/复习模式分节

    ## 为什么需要它

    学生说"我要复习操作系统"或"我要学死锁那一章"时，模型需要知道**这门课实际
    有哪些资料**，否则只能凭记忆编出一套章节（"第一天：操作系统概述…"），
    与本课课件对不上 —— 这正是用户实测反馈的问题。

    ## 节是从哪来的

    **PPT 的文件名本身就是天然的分节结构**：
        02-1-进程与线程之进程.pdf / 03-OS调度.pdf / 05-死锁.pdf / Chapter4-Network Layer.pdf
    文件名里带着编号与章节名，`chunk_count` 又近似等于内容量 ——
    这两样就是给模型判断"怎么分节、一节讲多少"的依据。

    ⚠️ 刻意**不**替模型决定粒度：不写"每 N 个知识点一节"这类硬规则，
    也不给出"第几节"的固定切分。返回的是**素材地图**，
    怎么切由模型按内容判断（用户明确要求不要写死）。

    Args:
        course: 课程名（操作系统 / 计算机网络）；留空则两门课都返回
    """
    from rag_tool import get_knowledge_base

    courses = [course] if course else ["操作系统", "计算机网络"]
    kb_docs = {d["source"]: d for d in get_knowledge_base().list_documents()}

    out = {}
    for c in courses:
        names = set(course_sources(c))
        if not names:
            out[c] = {"ppt": [], "others": [], "not_ingested": [],
                      "note": f"没找到「{c}」的资料目录（data/{c}/）"}
            continue
        ppts, others = [], []
        for name in sorted(names):
            d = kb_docs.get(name)
            if not d:
                continue                      # 在磁盘上但还没入库
            item = {
                "source": name,
                "chapter": d.get("chapter") or _chapter_from_name(name),
                "chunks": d.get("chunk_count", 0),
                "doc_type": d.get("doc_type", ""),
            }
            (ppts if item["doc_type"] == "ppt" else others).append(item)
        # 按文件名里的编号排序（02-1 < 02-2 < 03 < Chapter4…），与课程讲授顺序一致
        ppts.sort(key=lambda x: (_leading_number(x["source"]), x["source"]))
        others.sort(key=lambda x: -x["chunks"])
        out[c] = {
            "ppt": ppts,
            "others": others,
            "not_ingested": sorted(n for n in names if n not in kb_docs),
        }
    return out if not course else out.get(course, {})


def _leading_number(name: str) -> int:
    """取文件名开头的编号（02-1-xxx → 2，Chapter4-xxx → 4，没有 → 999）"""
    m = re.match(r"^(\d+)", name)
    if m:
        return int(m.group(1))
    m = re.match(r"^Chapter(\d+)", name, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return 999


def _chapter_from_name(name: str) -> str:
    """从文件名兜底推章节名（规则与入库时的 _extract_chapter 保持一致）"""
    stem = name.rsplit(".", 1)[0]
    m = re.match(r"^\d+[-_]?\d*[-_](.+)", stem)
    if m:
        return re.sub(r"之.+$", "", m.group(1))
    m = re.match(r"^Chapter\d+[-_](.+)", stem, re.IGNORECASE)
    if m:
        return m.group(1)
    return stem


def section_chunks(source: str, offset: int = 0, limit: int = 15) -> dict:
    """按**课件原始顺序**取出某一节的内容

    为什么不是"取 top-k 相似片段"：讲一章时，模型需要的往往不是"最相似的几句"，
    而是**这章按顺序讲了哪些东西** —— PPT 的 chunk_index 顺序就是它的讲授顺序。
    只给相似片段会让讲解跳跃、丢东西。

    同时这也天然支持"分节"：一份课件可能很长（06-内存管理 314 块），
    用 offset/limit 分页读，读多少算一节的量由模型判断。

    Returns:
        {"source", "chapter", "total", "offset", "limit", "returned",
         "has_more", "next_offset", "chunks": [{"index", "text"}]}
    """
    from rag_tool import get_knowledge_base

    kb = get_knowledge_base()
    try:
        got = kb._get_collection().get(where={"source": source},
                                      include=["documents", "metadatas"])
    except Exception as e:
        return {"source": source, "error": f"读取失败: {str(e)[:100]}"}

    ids = got.get("ids") or []
    if not ids:
        return {"source": source, "error": "这份资料不在知识库里"}

    rows = []
    for text, meta in zip(got.get("documents") or [], got.get("metadatas") or []):
        rows.append({"index": int(meta.get("chunk_index", 0)),
                     "text": text, "chapter": meta.get("chapter", "")})
    rows.sort(key=lambda r: r["index"])

    offset = max(0, int(offset or 0))
    limit = max(1, min(int(limit or 15), 40))     # 上限防止一次灌爆上下文
    page = rows[offset:offset + limit]
    return {
        "source": source,
        "chapter": rows[0]["chapter"] if rows else "",
        "total": len(rows),
        "offset": offset,
        "limit": limit,
        "returned": len(page),
        "has_more": offset + len(page) < len(rows),
        "next_offset": (offset + len(page)) if offset + len(page) < len(rows) else None,
        "chunks": [{"index": r["index"], "text": r["text"]} for r in page],
    }


# ============================================================
# 7. 学习 / 复习（两个运行面共用同一份实现）
# ============================================================
#
# 为什么放在这里而不是 tools.py：两个运行面（LangGraph 图 / 手写助教）都要用，
# 放进任一面的外壳里就必然分叉 —— 刷题那套已经踩过一次这个坑了。

def as_int(v, default: int) -> int:
    """容错取整数（模型常传字符串、负数、甚至「2节」这种）"""
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"-?\d+", str(v or ""))
    return int(m.group(0)) if m else default


def course_of(source: str) -> str:
    """反查某份资料属于哪门课（知识库 metadata 里没有课程字段，只能按目录反查）"""
    for c in ("操作系统", "计算机网络"):
        try:
            if source in set(course_sources(c)):
                return c
        except Exception:
            continue
    return ""


def find_chapter(course: str, chapter: str):
    """按章名找课件（学生说「死锁那一章」时用）

    匹配优先级：章节名完全相等 → 互相包含 → 出现在文件名里。
    """
    want = (chapter or "").strip()
    if not want:
        return None
    ppts = (section_index(course) or {}).get("ppt") or []
    norm = lambda s: re.sub(r"[\s\-_·、,，。()（）]", "", str(s or "")).lower()
    w = norm(want)
    # 中文 ↔ 英文对照：计网的课件名是英文（Transport Layer），
    # 而学生说的是中文（传输层）—— 实测说"复习传输层"时对不上，
    # 结果模型自己挑课件，挑成了网络层，讲的内容跟学生要的不是一回事。
    alias = {"传输层": "transportlayer", "网络层": "networklayer",
             "应用层": "applicationlayer", "数据链路层": "datalinklayer",
             "物理层": "physicallayer", "无线": "wireless", "课程简介": "introduction"}
    w_alias = alias.get(want, "")
    for p in ppts:
        if norm(p["chapter"]) == w:
            return p
    if w_alias:
        for p in ppts:
            if w_alias in norm(p["source"]) or w_alias in norm(p["chapter"]):
                return p
    for p in ppts:
        c = norm(p["chapter"])
        if w and (w in c or c in w):
            return p
    for p in ppts:
        if w and w in norm(p["source"]):
            return p
    return None


def match_source(given: str, names: list):
    """把模型给的文件名对到真实 source（允许省略后缀、写章节名）"""
    g = (given or "").strip()
    if not g:
        return None
    if g in names:
        return g
    gl = g.lower()
    for n in names:
        nl = n.lower()
        if nl == gl or nl.startswith(gl) or gl in nl or nl in gl:
            return n
    stem = g.rsplit(".", 1)[0]
    return match_source(stem, names) if stem != g else None


# 学习与复习的**节奏差异**。
#
# 用户的原话：「复习」更偏向快速过 + 多测，「学习」更偏向细讲 + 少测。
# 但同时也说过不要把粒度写死 —— 所以这里给的是**倾向与区间**，
# 具体每节讲多少、出几道题仍然由模型按内容判断。
STUDY_MODES = {
    "复习": {
        "pace": "**快速过**：按考点脉络走，略过细枝末节与推导，抓「考什么、怎么考」；"
                "同一份课件可以一带而过地过完，不必逐块展开",
        "quiz": "小测**多一些**（约 5~8 道），覆盖本节各个考点，多问「哪个对/哪个错」这类快问快答",
        "quiz_n": "5~8",
    },
    "学习": {
        "pace": "**细讲**：把机制讲透 —— 为什么这样设计、内部怎么运作、举个例子走一遍；"
                "宁可慢一点、少讲一点，也不要囫囵吞枣",
        "quiz": "小测**少一些**（约 2~3 道），但要问到「为什么」，确认真的理解了而不是记住了",
        "quiz_n": "2~3",
    },
}


def study_guide(mode: str = "学习") -> str:
    """学习/复习的呈现契约（两个运行面共用）

    这里是**细节的唯一出处**：两个 system prompt 只留指针（都要求提示词保持简短，
    细节堆进 prompt 会让模型变没主见 —— 助教那条甚至有测试卡着 1200 字符）。
    """
    m = STUDY_MODES.get((mode or "学习").strip(), STUDY_MODES["学习"])
    label = (mode or "学习").strip() if (mode or "").strip() in STUDY_MODES else "学习"
    return (
        f"【怎么用这一节｜模式：{label}】上面是这门课件的原文，按它自己的顺序排好了：\n"
        "1. **以它为准讲解**，不要用你记忆里的通用教材内容替换它；\n"
        f"2. 节奏按「{label}」来 —— {m['pace']}；\n"
        "3. 一次只讲**一节**（节的划分由你按内容判断），讲完明确说「本节结束」，\n"
        "   然后出覆盖本节核心考点的小测题，**不要**提前讲后面的内容；\n"
        f"4. {m['quiz']}。题型混着来（选择/填空/简答/计算），"
        "题量按内容定、不必凑固定数字；\n"
        "5. 学生作答后**逐题批改**：每题给 ✓/✗ 和一句讲解，错的要说清"
        "**错在哪、为什么错、正确答案是什么**，并把相关知识点**重讲一遍**；\n"
        "6. 任何缩写/符号**第一次出现**都给全称与含义（如 TLB＝Translation "
        "Look-aside Buffer，快表），别默认学生认识；\n"
        "7. 确认学生真懂了再进下一节（本节卡住就停下来解决，不要硬推）；\n"
        "8. 作业类请求：先教涉及的知识点 → 让学生自己写 → 批改 → 最后给完整答案；\n"
        "9. **不串课**：切换课程/章节时明确说清现在在讲哪门课、哪一章、进到哪了，"
        "两门课的内容不要混在一起；\n"
        "10. 讲完这一节用 `record_study_position` 记下位置（带上 mode），"
        "学生说「继续讲」时你才知道从哪接着读；\n"
        "11. 原文若是 PPT 碎片、语句不通顺，可以**组织成通顺的讲解**，"
        "但不要改变它的技术含义，也不要编造它没说的内容。\n"
        "⚠️ **本节的测验题是你自己出的**，所以要**由你自己批改** —— "
        "逐题在对话里给 ✓/✗ 与讲解就行。\n"
        "   **不要**调 grade_answers / get_paper_questions 去题库找卷子来批："
        "那是真题的批改通道，学生答的是你刚出的小测，题库里没有它。\n"
        "   （真实踩过：学生交小测答案，助教反问「你想做哪份模拟卷」—— "
        "把学习流程硬拽回了刷题流程。）"
    )


def study_read(course: str = "", chapter: str = "", section=0, source: str = "",
               offset=0, limit=15, mode: str = "学习") -> dict:
    """按课件原始顺序读一节 —— 两个运行面共用的实现

    返回:
        {"text": 给模型看的完整文本（含机器可读的 [study-state] 行）,
         "study_state": 供上层写进会话状态的书签/位置,
         "error": 可选}
    """
    from rag_tool import get_knowledge_base
    kb_docs = {d["source"]: d for d in get_knowledge_base().list_documents()}
    offset, limit, section = as_int(offset, 0), as_int(limit, 15), as_int(section, 0)
    offset = max(0, offset)
    note = ""
    real = None

    # 0) 按章名找 —— 学生最自然的说法就是「某一章」
    if chapter and course:
        hit = find_chapter(course, chapter)
        if hit is None:
            names = "、".join(f"《{p['chapter']}》"
                              for p in (section_index(course) or {}).get("ppt") or [])
            return {"error": f"「{course}」里没有叫「{chapter}」的章。这门课的章有：{names}",
                    "text": ""}
        real = hit["source"]
        note = f"【按章定位】「{chapter}」→《{real}》（章节：{hit['chapter']}）\n\n"

    # 1) 给了文件名
    if real is None and source:
        matched = match_source(source, list(kb_docs))
        if matched is not None:
            real = matched
            dtype = (kb_docs.get(real) or {}).get("doc_type", "")
            if dtype in ("textbook", "notes") and course:
                ppts = (section_index(course) or {}).get("ppt") or []
                if ppts:
                    pick = ppts[max(0, min(section - 1, len(ppts) - 1))] if section else ppts[0]
                    note = (f"【已自动换材料】你指定的是"
                            f"{'课本' if dtype == 'textbook' else '补充笔记'}《{real}》，"
                            f"讲课程内容应该用课件，已改为《{pick['source']}》。"
                            f"（课本/笔记只在课件没覆盖某个点时作补充。）\n\n")
                    real = pick["source"]
        elif course:
            ppts = (section_index(course) or {}).get("ppt") or []
            if ppts:
                pick = ppts[max(0, min(section - 1, len(ppts) - 1))] if section else ppts[0]
                note = (f"【已自动选材料】没找到叫「{source}」的资料，"
                        f"已改用「{course}」的课件《{pick['source']}》。"
                        f"下次可以直接 read_section(course=\"{course}\", section=编号)。\n\n")
                real = pick["source"]
        if real is None:
            names = "、".join(f"《{n}》" for n in list(kb_docs)[:8])
            return {"error": f"没找到叫「{source}」的资料。请改用 "
                             f"read_section(course=\"操作系统\", section=1) 这种写法；"
                             f"现有资料如：{names}", "text": ""}

    # 2) 只给了课程
    if real is None and course:
        ppts = (section_index(course) or {}).get("ppt") or []
        if not ppts:
            return {"error": f"「{course}」没有可用课件，先调 list_course_sections 确认",
                    "text": ""}
        # 计网的课件是 ChapterN- 命名的，而模型说"传输层"时会想成 Chapter5、
        # 把 5 传进 section —— 按**位置**数第 5 份是 Chapter4（课件从 Chapter0 起，差一位），
        # 于是它读了网络层、却凭记忆讲 TCP/UDP（实测）。
        # 所以：这类课程里 section **优先按章号解释**。
        chap_nums = {}
        for p in ppts:
            m = re.match(r"^Chapter(\d+)", p["source"], re.IGNORECASE)
            if m:
                chap_nums.setdefault(int(m.group(1)), p)
        if chap_nums and len(chap_nums) >= len(ppts) - 1 and section in chap_nums:
            pick = chap_nums[section]
            idx = ppts.index(pick) + 1
            real = pick["source"]
            note = (f"【按章号定位】{course} · Chapter{section} → 《{real}》"
                    f"（章节：{pick['chapter']}）\n\n")
        else:
            idx = section if 1 <= section <= len(ppts) else 1
            pick = ppts[idx - 1]
            real = pick["source"]
            note = (f"【本课课件 {idx}/{len(ppts)}】{course} · 《{real}》"
                    f"（章节：{pick['chapter']}）"
                    + (f"\n下一份是第 {idx + 1} 份《{ppts[idx]['source']}》"
                       if idx < len(ppts) else "\n这是最后一份课件。") + "\n\n")

    if real is None:
        return {"error": "要指定材料：read_section(course=\"操作系统\", chapter=\"死锁\") "
                         "或 read_section(course=\"操作系统\", section=1)", "text": ""}

    data = section_chunks(real, offset=offset, limit=limit)
    if data.get("error"):
        return {"error": data["error"], "text": ""}

    head = (f"《{real}》｜章节：{data['chapter'] or '未标注'}｜"
            f"共 {data['total']} 块｜本次第 {data['offset'] + 1}~"
            f"{data['offset'] + data['returned']} 块"
            + (f"（后面还有 {data['total'] - data['offset'] - data['returned']} 块，"
               f"下次用 offset={data['next_offset']} 继续）"
               if data["has_more"] else "（已到本章末尾）"))
    body = "\n\n".join(f"[{c['index']}] {c['text']}" for c in data["chunks"])
    # 机器可读的一行：上层靠它把「读到哪了」记进会话状态。
    # 明写而不是让上层解析人话 —— 解析措辞太脆（我第一版就这么干，结果 study 里
    # 没有 course，planner 的整块提示被静默跳过，模型照样不知道读到哪）。
    state_line = (f"[study-state] course={course or course_of(real)} "
                  f"chapter={data['chapter'] or ''} source={real} "
                  f"read={data['offset'] + 1}-{data['offset'] + data['returned']} "
                  f"total={data['total']} mode={mode or '学习'} "
                  f"next_offset={data['next_offset'] if data['next_offset'] is not None else 'none'}")
    text = f"{note}{head}\n\n{body}\n\n{state_line}\n\n----\n{study_guide(mode)}"
    return {
        "text": text,
        "study_state": {
            "course": course or course_of(real),
            "chapter": data["chapter"] or "",
            "last_source": real,
            "last_read": f"第 {data['offset'] + 1}-{data['offset'] + data['returned']} 块",
            "next_offset": data["next_offset"],
            "mode": mode or "学习",
        },
    }


# ============================================================
# 8. 临时出题（不返回题目，返回出题素材）
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
