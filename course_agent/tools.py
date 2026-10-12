"""工具定义

统一工具库，供 Plan-and-Execute 和 ReAct 两种 Agent 使用。
Week 4 工具调用能力 + 刷题扩展能力。

刷题相关的**逻辑**一律走 course_agent/quiz_service.py（与 tutor_agent.py 共用的
同一份实现），这里只做 LangChain 的薄适配；检索相关走 rag_tool.py。
不要再往这个文件里加业务逻辑 —— 历史上这里和 tutor_agent.py 各写了一份，
已经真实分叉过（这一侧的阅卷缺 basis_warning、题号也没按题型内序号算）。

工具列表:
  - search_knowledge: 检索 Chroma 知识库（计网+OS课程笔记）
  - web_search: 联网搜索（智谱 web_search），知识库不够时补充
  - get_course_info: 获取知识库统计信息
  - list_papers: 列出习题库中的试卷
  - get_paper_questions: 获取指定试卷某组的题目
  - grade_answers: 批量批改学生答案
  - explain_question: 取题面与参考答案用于讲题
  - get_practice_stats: 获取试卷练习统计
  - record_practice: 记录一次试卷练习
"""

import json
import re

import requests
from langchain_core.tools import tool

from rag_tool import search_knowledge as _search_knowledge
from rag_tool import get_knowledge_base
from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL
from course_agent import quiz_service
from course_agent.errors import retry_on_rate_limit, RateLimitError


# ============================================================
# 知识库相关工具
# ============================================================

@tool
def search_knowledge(query: str, course: str = "", source: str = "") -> str:
    """搜索课程知识库（计算机网络+操作系统）。输入问题或关键词，返回相关的知识点内容。

    知识库包含: TCP/UDP、拥塞控制、进程调度、死锁、同步互斥等课程内容。

    **讲解课程内容时务必带上 course（或者在知道具体是哪份课件时带上 source）** ——
    这样检索会收敛到那门课/那份课件的资料上，不会串到另一门课，
    也才谈得上"优先参考 PPT"。

    Args:
        query: 搜索关键词或问题，如 "TCP三次握手" 或 "死锁的四个必要条件"
        course: 可选，限定课程：操作系统 / 计算机网络
        source: 可选，限定到某个具体文件（课件名，可用 list_course_sections 拿到），
                如 "05-死锁.pdf"。给了它就只在这份资料里找，最适合"讲某一章"
    """
    if source:
        raw, scope_note = _search_in_source(query, source, top_k=5)
    elif course:
        raw, scope_note = _search_in_course(query, course, top_k=4)
    else:
        raw, scope_note = _search_knowledge(query, top_k=3), ""
    # 呈现契约必须写进返回值 —— 和 list_papers / get_paper_questions 一个道理：
    # 只给一段资料，模型会拿自己记忆里的通用教材内容替换它，
    # 于是学生问"这门课怎么讲"，得到的是模型背的教科书。
    # 实测：说"想复习操作系统"，它讲的全是 MS-DOS / OS/360 这类通用常识，
    # 而课程课件讲的是进程/调度/死锁/内存 —— 检索到的资料完全没被用上。
    return (
        f"{scope_note}{raw}\n\n"
        "----\n"
        "【怎么用上面的片段】它们就是这门课的资料原文，是你讲解的依据：\n"
        "1. **以这些片段为准**来讲，不要用你记忆里的通用教材内容替换它；\n"
        "2. 讲完可以点一下来源（如「来自 03-OS调度」），让学生知道这是本课资料；\n"
        "3. 如果片段里**没有**覆盖学生问的点，就直说「课程资料里这部分没找到」，\n"
        "   再用你的通用知识补充，并**明确标注**那是补充、不是本课内容；\n"
        "4. 不要编造片段里没有的「课件原文」或页码；\n"
        "5. **任何缩写或符号第一次出现**，都要给出全称和含义（如 LRU＝Least Recently Used，"
        "最近最少使用），别默认学生认识。"
    )


def _search_in_course(query: str, course: str, top_k: int = 4):
    """在指定课程范围内检索，并如实告诉调用方"这门课到底有没有这个内容"

    复用 quiz_service.search_with_course：它用**绝对相似度**判断话题是否真落在
    这门课里（拿"结果非空"当命中是不行的 —— 非空库永远返回 top_k 条）。
    """
    from rag_tool import get_knowledge_base
    try:
        hits, confirmed = quiz_service.search_with_course(
            get_knowledge_base(), query, course, top_k=top_k)
    except Exception as e:                      # 检索失败不能把整轮打崩
        return f"（检索出错：{str(e)[:80]}）", ""
    text = _format_hits(hits)
    if confirmed:
        return text, f"【检索范围】已在「{course}」的资料内检索。\n\n"
    return (text, f"【检索范围】「{course}」的资料里**没找到**这个话题，"
                  f"下面是全库结果 —— 讲的时候要说清楚这不是该课程的内容。\n\n")


def _search_in_source(query: str, source: str, top_k: int = 5):
    """在某个具体文件里检索（"讲某一章"就用它）"""
    from rag_tool import get_knowledge_base
    kb = get_knowledge_base()
    real = _match_source_name(source, [d["source"] for d in kb.list_documents()])
    if real is None:
        return (f"（没找到叫「{source}」的资料，先用 list_course_sections 看清课程结构）",
                "")
    try:
        hits = kb.search(query, top_k=top_k, sources=[real])
    except Exception as e:
        return f"（检索出错：{str(e)[:80]}）", ""
    if not hits:
        return (f"（「{real}」里没检索到与「{query}」相关的内容，"
                f"可以换个关键词，或先 list_course_sections 看看这一章都有什么）", "")
    return _format_hits(hits), f"【检索范围】只在《{real}》里检索。\n\n"


def _match_source_name(given: str, names: list):
    """把用户/模型给的文件名对到真实 source（允许省略后缀、写章节名）"""
    g = (given or "").strip()
    if not g:
        return None
    for n in names:
        if n == g:
            return n
    gl = g.lower()
    for n in names:
        nl = n.lower()
        if nl == gl or nl.startswith(gl) or gl in nl or nl in gl:
            return n
    stem = g.rsplit(".", 1)[0]
    if stem != g:
        return _match_source_name(stem, names)
    return None


def _format_hits(hits: list) -> str:
    """把检索结果排成模型好读的样子（带来源与相关度）

    ⚠️ 两个字段语义不同，必须分清楚（我在这里踩过一次）：
      · `kb.search()`（纯向量）返回 **score = 绝对余弦**，没有 sim；
      · `kb._hybrid_search()`（融合）返回 sim = 绝对余弦、score = **归一化融合分**。
    归一化分只能排序、不能当"够不够相关"的判据，所以显示时优先用 sim。
    """
    if not hits:
        return "（没有检索到内容）"
    parts = []
    for h in hits:
        rel = h.get("sim") if h.get("sim") is not None else h.get("score", 0)
        parts.append(f"[{h.get('source', '?')} · 相关度={rel:.3f}]\n{h.get('text', '')}")
    return "\n\n".join(parts)


def _as_int(v, default: int) -> int:
    """容错取整数（模型常传字符串、负数、甚至"2节"这种）"""
    if isinstance(v, bool):
        return default
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"-?\d+", str(v or ""))
    return int(m.group(0)) if m else default


def _course_ppt_list(course: str) -> list:
    idx = quiz_service.section_index(course)
    return idx.get("ppt") or []


def _course_of(source: str) -> str:
    """反查某份资料属于哪门课（知识库 metadata 里没有课程字段，只能按目录反查）"""
    for c in ("操作系统", "计算机网络"):
        try:
            if source in set(quiz_service.course_sources(c)):
                return c
        except Exception:
            continue
    return ""


def _find_chapter(course: str, chapter: str):
    """按章名找课件（学生说"死锁那一章"时用）

    匹配优先级：章节名完全相等 → 互相包含 → 出现在文件名里。
    """
    want = (chapter or "").strip()
    if not want:
        return None
    ppts = _course_ppt_list(course)
    norm = lambda s: re.sub(r"[\s\-_·、,，。()（）]", "", str(s or "")).lower()
    w = norm(want)
    for p in ppts:
        if norm(p["chapter"]) == w:
            return p
    for p in ppts:
        c = norm(p["chapter"])
        if w and (w in c or c in w):
            return p
    for p in ppts:
        if w and w in norm(p["source"]):
            return p
    return None


@tool
def read_section(course: str = "", chapter: str = "",
                 section: "int | str" = 0, source: str = "",
                 offset: "int | str" = 0, limit: "int | str" = 15,
                 mode: str = "学习") -> str:
    """按**课件原始顺序**读取某一节的内容 —— 讲课程内容时用它（优先于 search_knowledge）

    ## 怎么用（**不需要你记文件名**）

    学生说"**某一章**"（如"我要学死锁那一章"）→ 用章名：
        read_section(course="操作系统", chapter="死锁")
    学生说"**第几节 / 接着讲**"→ 用节号（编号来自 list_course_sections）：
        read_section(course="操作系统", section=1)
    想知道这门课有哪些章、几份课件，先调 list_course_sections。

    一份课件很长时（如"06-内存管理与虚拟内存"有 314 块）用 offset 续读。
    **一块/一节读多少由你判断**，本工具不规定。

    Args:
        course: 课程名（**必填**）：操作系统 / 计算机网络
        chapter: 章名（学生按章提问时用它，如 "死锁"、"OS调度"、"Transport Layer"）
        section: 第几份课件，从 1 开始
        source: 也可以直接给课件名（一般不用）
        offset: 从第几块开始，默认 0
        limit: 本次读几块，默认 15（上限 40）
    """
    # 实现已下沉到 quiz_service.study_read —— 两个运行面（LangGraph 图 / 手写助教）
    # 共用同一份，避免像刷题那样各写一份、然后真实分叉。
    res = quiz_service.study_read(course=course, chapter=chapter, section=section,
                                  source=source, offset=offset, limit=limit, mode=mode)
    if res.get("error"):
        return f"（{res['error']}）"
    return res["text"]


@tool
def list_course_sections(course: str = "") -> str:
    """列出某门课**实际有哪些资料**（课件/课本/笔记），用于分节讲解与复习

    学生说"我要复习操作系统""我要学死锁那一章""带我一章一章过"时，
    先调它拿到这门课的**资料地图**，再决定怎么分节 —— 不要凭记忆编章节。

    返回值里每份课件都带 `chapter`（章节名）与 `chunks`（内容量，近似课时长度），
    分节粒度由你按内容判断（本工具刻意不给固定切分）。

    Args:
        course: 课程名：操作系统 / 计算机网络；留空则两门课都给
    """
    data = quiz_service.section_index(course)
    lines = []
    for c, v in (data.items() if isinstance(data, dict) and course == "" else
                 [(course, data)]):
        lines.append(f"## {c}")
        if v.get("note"):
            lines.append(v["note"])
        lines.append("### 课件（讲课时**优先用这些**，按课程顺序排好）")
        for i, s in enumerate(v.get("ppt", []), 1):
            lines.append(f"{i}. {s['source']}  ｜章节：{s['chapter']}｜约 {s['chunks']} 块")
        if v.get("ppt"):
            lines.append(f"→ 学生没说从哪章开始时，**从第 1 份课件开始**"
                         f"（《{v['ppt'][0]['source']}》），不要跳到课本的章节去讲。")
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
    body = "\n".join(lines)
    return (
        f"{body}\n"
        "----\n"
        "【怎么用这份地图】\n"
        "⚠️ **不要把这份清单直接发给学生** —— 他要的是「带我复习」，不是一份目录。\n"
        "拿到地图后**接着就开讲第一节**：用 read_section(course=…, section=1) "
        "取课件原文，然后按契约讲。最多用一句话说明「这门课有哪几块内容」，就开讲。\n"
        "⚠️ 学生**按章名**提问时（「复习传输层」「学死锁那一章」），"
        "必须用 `read_section(course=…, chapter=\"章名\")` —— "
        "**不要自己数编号**去猜 section。实测数错过：说传输层却读了网络层那章。\n"
        "1. **分节由你判断**：这份地图只告诉你「有哪些资料、各有多厚」，"
        "不规定一节包含几个知识点。按内容的内在结构切，"
        "一节控制在「学生一节课（约半小时到一小时）能消化」的量；\n"
        "2. 讲某一节时，用 `search_knowledge(query, source=\"那份课件名\")` "
        "把资料取出来再讲 —— **不要凭记忆讲这门课的内容**；\n"
        "3. 一次只讲一节，讲完就说「本节结束」，不要提前把后面的内容也倒出来；\n"
        "4. 定了这一节要讲什么之后，用 `record_study_position` 记一下当前位置，"
        "这样学生说「下一节」时你知道该从哪继续（也避免两门课串味）。\n"
        # 只调了这份地图、没调 read_section 时，会话里就没有学习书签，
        # 而"以课件原文为准 / 讲完说本节结束 / 不要编题编图"这些契约都挂在书签上
        # （实测那一轮模型凭记忆讲课，还编了题和 example.com 图片）。
        # 所以这里显式把"正在学哪门课"报给上层。
        + (quiz_service.study_course_line(course) if course else "")
    )


@retry_on_rate_limit(max_retries=3, base_wait=2.0)
def _call_zhipu_web_search(query: str) -> dict:
    """调用智谱 API 做联网搜索（带重试）"""
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "glm-4-flash",
        "messages": [{"role": "user", "content": query}],
        "tools": [{"type": "web_search", "web_search": {"enable": True}}],
    }
    resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=60)
    if resp.status_code == 429:
        # 必须抛 RateLimitError：retry_on_rate_limit 只认这个类型（或 requests 的 HTTPError），
        # 抛 APIError 的话重试装饰器直接放行，等于没有重试
        raise RateLimitError("Rate limited (429)")
    return resp.json()


@tool
def web_search(query: str) -> str:
    """联网搜索获取互联网上的最新信息。当知识库中没有找到相关内容，或问题涉及课程之外的知识时使用。

    Args:
        query: 搜索关键词，如 "2024年最新TCP拥塞控制算法" 或 "Linux内核调度器CFS原理"
    """
    try:
        data = _call_zhipu_web_search(query)
    except Exception as e:
        return f"联网搜索失败: {type(e).__name__}: {str(e)[:100]}"

    if "choices" in data:
        content = data["choices"][0]["message"].get("content", "")
        refs = data["choices"][0]["message"].get("web_search", [])
        if refs:
            sources = "\n".join(
                f"[{i+1}] {r.get('title', '')} - {r.get('link', '')}"
                for i, r in enumerate(refs[:5])
            )
            return f"{content}\n\n参考链接:\n{sources}"
        return content
    return f"联网搜索失败: {data.get('error', {}).get('message', '未知错误')}"


@tool
def get_course_info() -> str:
    """查看知识库信息：包含哪些课程文档、共多少条知识点。"""
    kb = get_knowledge_base()
    count = kb.count()
    if count == 0:
        return "知识库为空。请先运行 python build_knowledge.py 建库。"
    return f"知识库当前有 {count} 条文档分块。"


# ============================================================
# 刷题相关工具
#
# 这几件工具的**逻辑**都在 course_agent/quiz_service.py —— 与 tutor_agent.py
# 的手写工具层共用同一份实现。这里只是 LangChain 的薄适配：
# 把显式参数转成 service 调用，再把结果序列化成 JSON 字符串。
#
# 原来这里和 tutor_agent.py 各写了一份，而且已经真实分叉：
# 这一侧的阅卷既没带 basis_warning，也没把题号传给 build_llm_basis。
# 不要再往这里加业务逻辑 —— 改就改 quiz_service。
# ============================================================

@tool
def list_papers(course: str = "") -> str:
    """列出习题库中所有可用的试卷，用于刷题前让学生选择。

    Args:
        course: 可选，按课程筛选：操作系统 或 计算机网络
    """
    listing = quiz_service.papers_text(course or None)
    # 必须带上"这一步只做什么"的指引。只给一份清单的话，模型会认为
    # "用户想刷题"还没被满足，于是**自己编题目和图片链接**（实测编出过
    # https://example.com/... 这种占位地址）。手写助教那侧靠 _guide 解决，
    # 这里用同样的机制：把呈现契约写进工具返回值。
    return (
        f"{listing}\n\n"
        "----\n"
        "【本步只做一件事】把上面这份清单发给学生，并问他选哪一份。\n"
        "**这一步不要开始出题**：题目只能来自 list_papers 里列出的试卷，"
        "必须等学生选定后调用 get_paper_questions 才有题。\n"
        "**绝对不要自己编写题目、选项或图片链接** —— 编出来的题和图片都是假的。\n"
        "如果学生要刷的是**某个知识点**的题（而不是整份卷子），"
        "题库没有按知识点组卷 —— 那种情况用 prepare_custom_quiz 自己出题，"
        "不要拿某份卷子的第一组硬凑给他。"
    )


@tool
def get_paper_questions(paper_source: str, group_index: int = 1) -> str:
    """获取指定试卷的一组题目，用于开始/继续刷题。

    一次只返回一组：选择题/填空题/判断题一组 5 题，综合题一组 1 题。
    默认给第 1 组；学生做完要下一组时，把上一组结果里的 next_group_index 传进来
    （这里没有会话状态，所以必须由调用方带上组号，否则只会反复给第 1 组）。
    每题包含题号、题型、知识点和图片地址；只展示这些，题干和答案不对外给出。
    学生说不会 / 让你讲讲时，用 explain_question 取题面和答案要点再讲。

    Args:
        paper_source: 试卷来源名称（从 list_papers 结果中获取）
        group_index: 第几组，从 1 开始；继续做题时传上一组返回的 next_group_index
    """
    paper, candidates = quiz_service.resolve_paper(paper_source)
    if paper is None:
        if candidates:
            return "没能确定是哪份卷子，请从这些里选一个：\n" + "\n".join(candidates)
        return "习题库为空。"

    payload = quiz_service.group_payload(paper, group_index=group_index)
    if payload.get("error") or payload.get("finished"):
        return json.dumps(payload, ensure_ascii=False)

    # 把 ready-to-send 的成品文本一并给出：模型不用自己拼排版，
    # 也就没有"拼不出题就编题"的空间。
    return json.dumps({
        "paper": payload["paper"],
        # type 要带上：executor 靠它把"当前在做哪种题型"写进跨轮游标，
        # 漏了的话重启后恢复出来的游标里题型是 null（实测到过）。
        "type": payload["type"],
        "progress": payload["progress"],
        "group_index": payload["group_index"],
        "group_count": payload["group_count"],
        "has_more": payload["has_more"],
        "next_group_index": payload["next_group_index"],
        "questions": payload["questions"],
        "qids": payload["qids"],
        "reply_text": payload["reply_text"],
        "render_hint": ("直接把 reply_text 作为回复主体发出去；题目必须整组出现，"
                        "图片必须用 reply_text 里的 /qb_images/ 地址。"
                        "不要改动题号或知识点，也不要自己补题目文字或选项。"),
    }, ensure_ascii=False)


@tool
def grade_answers(paper_source: str, user_answer_text: str, qids: list[str] = None) -> str:
    """批量批改学生答案。学生提交答案后，调用此工具获取批改结果。

    支持多种答案格式：1.A 2.B、ABCDA、1.对 2.错等，工具会自动解析。
    选择题/判断题由代码精确判定；填空题/综合题依据答案与参考要点由阅卷模型评分。

    Args:
        paper_source: 试卷来源名称
        user_answer_text: 学生提交的答案原文，原样传入即可，工具会自动解析
        qids: 可选，要批改的题目 qid 列表（get_paper_questions 返回过）。
              传入可精确定位到刚出的那一组；不传则按题型分段解析整卷。
    """
    paper_source, qids = quiz_service.normalize_grade_args(paper_source, qids)
    paper, candidates = quiz_service.resolve_paper(paper_source)
    if paper is None:
        if candidates:
            return json.dumps({"error": f"没能确定试卷: {paper_source}",
                               "candidates": candidates}, ensure_ascii=False)
        return json.dumps({"error": "习题库为空"}, ensure_ascii=False)

    graded = quiz_service.grade_submission(paper, user_answer_text, qids=qids)
    if graded["total"] == 0:
        return json.dumps({
            "total": 0,
            "notes": graded["notes"],
            "message": "没能从这段话里认出答案，请让学生用「题号+答案」的形式再说一次",
        }, ensure_ascii=False)

    return json.dumps({
        "total": graded["total"],
        "correct": graded["correct"],
        "wrong": graded["wrong"],
        "needs_manual": graded["needs_manual"],
        "accuracy": graded["accuracy"],
        "results": graded["results"],
    }, ensure_ascii=False)


@tool
def explain_question(paper_source: str, no: list[int] = None, qids: list[str] = None) -> str:
    """取出题目的原文、参考答案和评分要点，用来给学生讲题。

    学生说不会 / 没思路 / 让你讲讲 / 问为什么，或者你要讲解某道错题时调用它。
    出题接口刻意不给题面和答案（防泄题），所以没有这个工具你就讲不了题。

    Args:
        paper_source: 试卷来源名称
        no: 要讲的题号（题型内序号），如 [1] 或 [1,2]；不传则讲整卷全部题
        qids: 也可以用 get_paper_questions 返回的 qid 指定
    """
    paper, candidates = quiz_service.resolve_paper(paper_source)
    if paper is None:
        if candidates:
            return json.dumps({"error": f"没能确定试卷: {paper_source}",
                               "candidates": candidates}, ensure_ascii=False)
        return json.dumps({"error": "习题库为空"}, ensure_ascii=False)

    return json.dumps(quiz_service.explain_payload(paper, no=no, qids=qids),
                      ensure_ascii=False)


@tool
def prepare_custom_quiz(topic: str, course: str = "", qtype: str = "选择题",
                        count: int = 5, difficulty: str = "") -> str:
    """为「临时出题」取素材（学生不想做题库里的卷子、想针对某个知识点多练几道时用它）。

    注意：这个工具**不返回题目**，返回的是出题素材 —— 拿到素材后由你自己写题。
    写出来的题要在题前说清楚是自拟题，不要声称来自题库。

    Args:
        topic: 知识点或章节，如「死锁」「TCP拥塞控制」
        course: 可选，课程：操作系统 / 计算机网络
        qtype: 想要什么题型，默认选择题
        count: 出几道，默认 5
        difficulty: 可选，难度：基础/进阶/综合
    """
    payload = quiz_service.prepare_custom_quiz(
        topic=topic, course=course, qtype=qtype, count=count, difficulty=difficulty)
    return json.dumps(payload, ensure_ascii=False)


@tool
def record_study_position(course: str, section: "str | int" = "", mode: str = "学习",
                          chapter: str = "", source: str = "", status: str = "讲解中") -> str:
    """记下当前学习/复习到哪了，供下一轮「下一节」接着讲

    **它只是一个书签，不是状态机** —— 讲什么、怎么分节、何时进下一节，
    全部由你判断；这个工具只负责让下一轮的你知道"现在在哪"。

    什么时候调：
      · 定好要讲哪一节、开始讲时 → status="讲解中"
      · 本节讲完并小测通过时 → status="已通过"（下次要讲下一节）
      · 学生卡住了没听懂 → status="待巩固"（下次先回来解决这个）

    Args:
        course: 课程名：操作系统 / 计算机网络
        section: 这一节的标题（**你自己起的**，如"进程与线程：进程的引入"）
        mode: 学习 / 复习
        chapter: 可选，对应的章（一般填课件对应的章节名）
        source: 可选，这一节主要依据哪份课件（如 "02-1-进程与线程之进程.pdf"）
        status: 讲解中 / 已通过 / 待巩固
    """
    # 纠偏：实测模型会把**章名**填进 course（course="死锁"）。
    # 课程名是有限集合，不是就直接当章名处理，避免污染书签里的课程字段。
    known = ("操作系统", "计算机网络")
    course = (course or "").strip()
    if course and course not in known:
        for c in known:
            if _find_chapter(c, course):
                chapter = chapter or course
                course = c
                break
    pos = {
        "mode": (mode or "学习").strip(),
        "course": (course or "").strip(),
        "chapter": (chapter or "").strip(),
        "section": str(section or "").strip(),
        "source": (source or "").strip(),
        "status": (status or "讲解中").strip(),
    }
    return json.dumps({
        "recorded": pos,
        "guide": (
            f"已记下：{pos['course']} · {pos['chapter'] or '（未标章）'} · {pos['section']}"
            f"（{pos['status']}）。\n"
            "接下来按你自己定的节奏走：\n"
            "- 本节还在讲 → 只讲这一节，讲完明确说「本节结束」，不要提前讲后面的内容；\n"
            "- 本节已讲完 → 出 3~5 道**覆盖本节核心考点**的小测题"
            "（选择/填空/简答/计算混合，题量与题型由你按内容定，不必凑固定数目）；\n"
            "- 学生作答后 → 逐题批改：每题给 ✓/✗ + 一句讲解；"
            "**错的要说清「错在哪、为什么错、正确答案」**，并把相关知识点重讲一遍；\n"
            "- 确认学生真懂了再进下一节（再用本工具记一次新位置）；"
            "学生卡住就停下来解决，不要硬推。"
        ),
    }, ensure_ascii=False)


@tool
def get_practice_stats(paper_source: str) -> str:
    """获取试卷练习统计（练习次数）。

    Args:
        paper_source: 试卷来源名称
    """
    return json.dumps({"practice_count": quiz_service.practice_count(paper_source)},
                      ensure_ascii=False)


@tool
def record_practice(paper_source: str) -> str:
    """记录完成了一次试卷练习，练习次数+1。

    Args:
        paper_source: 试卷来源名称
    """
    return json.dumps({"practice_count": quiz_service.record_practice(paper_source)},
                      ensure_ascii=False)


# ============================================================
# 导出
# ============================================================

tools = [
    search_knowledge,
    list_course_sections,
    read_section,
    record_study_position,
    web_search,
    get_course_info,
    list_papers,
    get_paper_questions,
    explain_question,
    grade_answers,
    prepare_custom_quiz,
    get_practice_stats,
    record_practice,
]

tools_by_name = {t.name: t for t in tools}
