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
def search_knowledge(query: str) -> str:
    """搜索课程知识库（计算机网络+操作系统）。输入问题或关键词，返回相关的知识点内容。

    知识库包含: TCP/UDP、拥塞控制、进程调度、死锁、同步互斥等课程内容。

    Args:
        query: 搜索关键词或问题，如 "TCP三次握手" 或 "死锁的四个必要条件"
    """
    return _search_knowledge(query, top_k=3)


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
        "**绝对不要自己编写题目、选项或图片链接** —— 编出来的题和图片都是假的。"
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
