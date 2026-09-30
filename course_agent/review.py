"""复习模式 - 知识点速览 + 刷题训练（错题跟踪）

知识点速览: 用户选章节 → KB检索 → LLM生成重点速查
刷题训练: 从习题库加载试卷 → 选择题5题一组 → 综合题1题一组 → 批改讲解 → 错题标记
"""

import json
import re
import uuid
import random
from pathlib import Path
from collections import defaultdict

from course_agent.llm import stream_chat, chat
from course_agent.chapter_utils import (
    get_chapters, match_chapter_with_fallback, format_chapter_list,
)
from rag_tool import get_knowledge_base
from course_agent.logger import logger

_sessions = {}
_QB_PATH = Path("data") / "question_bank" / "question_bank.json"
_STATS_PATH = Path("data") / "question_bank" / "practice_stats.json"

KNOWLEDGE_REVIEW_PROMPT = """请为《{chapter}》生成重点知识速查卡。

知识库内容:
{context}

要求:
1. 列出本章所有核心知识点
2. 每个知识点用1-2句话概括
3. 标注常见考点
4. 适合快速浏览复习

格式:
## 《{chapter}》重点速查

### 1. 知识点名称
**要点**: ...
**考点**: ...

### 2. ...
"""

QUIZ_GRADE_PROMPT = """请批改以下{q_type}:

题目:
{question_text}

参考答案:
{correct_answer}

学生答案:
{user_answer}

请批改并讲解:
1. 判断对错（✓/✗）
2. 如果错误，指出错在哪里
3. 给出正确解题思路

格式:
**判定**: ✓/✗
**解析**: ...
"""


def _load_question_bank():
    """加载习题库"""
    if not _QB_PATH.exists():
        return []
    try:
        with open(_QB_PATH, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception as e:
        logger.error("加载习题库失败: %s", e)
        return []


def _load_stats():
    """加载练习统计 {source: count}"""
    if not _STATS_PATH.exists():
        return {}
    try:
        with open(_STATS_PATH, "r", encoding="utf-8-sig") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_stats(stats):
    """保存练习统计"""
    try:
        _STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_STATS_PATH, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error("保存练习统计失败: %s", e)


def _get_paper_list():
    """获取所有试卷列表（按课程分组）"""
    all_questions = _load_question_bank()
    if not all_questions:
        return {}

    papers = defaultdict(list)
    for q in all_questions:
        papers[q.get("source", "unknown")].append(q)

    stats = _load_stats()
    result = {}
    for source, qs in papers.items():
        course = ""
        if qs:
            course = qs[0].get("course", "")
        choice_count = sum(1 for q in qs if q.get("type") == "选择题")
        complex_count = sum(1 for q in qs if q.get("type") == "综合题")
        result[source] = {
            "course": course,
            "total": len(qs),
            "choice": choice_count,
            "complex": complex_count,
            "practice_count": stats.get(source, 0),
        }
    return result


def _get_image_url(q):
    """获取题目的图片 URL"""
    images = q.get("images") or []
    if images:
        path = images[0].get("path", "")
        if path:
            # JSON 中存的路径前缀是 static/qb_images/，实际挂载在 /qb_images/
            path = path.replace("static/qb_images/", "", 1).replace("static\\qb_images\\", "", 1)
            if not path.startswith("/"):
                path = "/" + path
            return f"/qb_images{path}"
    image = q.get("image", "")
    if image:
        return f"/qb_images/{image}"
    return None


# ============================================================
# 知识点速览
# ============================================================

def process_knowledge_review(session_id, user_input):
    """处理知识点速览消息"""
    chapters = get_chapters()
    chapter, fail_msg = match_chapter_with_fallback(user_input, chapters)

    if not chapter:
        if chapters:
            chapter_list = format_chapter_list(chapters)
            msg = f"{fail_msg}\n\n📖 可选章节:\n{chapter_list}\n\n请直接说出章节名称，如：复习进程与线程"
        else:
            msg = "知识库为空，请先导入课程文档"
        yield {"type": "error", "msg": msg}
        return

    try:
        kb = get_knowledge_base()
        collection = kb._get_collection()
        results = collection.get(where={"chapter": chapter})
        chunks = [
            {"text": doc, "source": results["metadatas"][i].get("source", "")}
            for i, doc in enumerate(results["documents"])
        ]
    except Exception:
        chunks = []

    if not chunks:
        try:
            chunks = get_knowledge_base().search(chapter, top_k=15)
        except Exception:
            chunks = []

    context = "\n\n".join(c["text"] for c in chunks[:20])

    sid = session_id or str(uuid.uuid4())
    yield {"type": "start", "session_id": sid}
    yield {"type": "phase", "phase": "knowledge_review", "chapter": chapter,
           "message": f"📋 《{chapter}》重点速查"}

    messages = [
        {"role": "system", "content": "你是课程教师，擅长总结重点。"},
        {"role": "user", "content": KNOWLEDGE_REVIEW_PROMPT.format(chapter=chapter, context=context)},
    ]

    full_text = ""
    for chunk in stream_chat(messages, temperature=0.5):
        full_text += chunk
        yield {"type": "content", "content": chunk}

    hint = f"\n\n---\n《{chapter}》重点速查完毕。可以复习其他章节，或切换到「刷题训练」模式。"
    yield {"type": "content", "content": hint}
    full_text += hint

    yield {"type": "done", "answer": full_text, "session_id": sid}


# ============================================================
# 刷题训练
# ============================================================

def _is_answer_like(text):
    """判断是否是答案输入"""
    text = text.strip()
    if "不会" in text or "不知道" in text:
        return True
    if re.match(r"^[A-Da-d\s.、,]+$", text) and len(text) <= 30:
        return True
    if re.search(r"\d+\s*[.．、]?\s*[A-Da-d]", text):
        return True
    if re.search(r"答案\s*[是为]?\s*[A-Da-d]", text):
        return True
    return False


def _is_command(text):
    """判断是否是指令（换卷子、退出等）"""
    text = text.strip()
    commands = ["换", "重选", "返回", "退出", "结束", "其他", "别的", "重新", "不做了"]
    return any(c in text for c in commands)


def process_quiz(session_id, user_input):
    """处理刷题消息"""

    if not session_id or session_id not in _sessions:
        sid = str(uuid.uuid4())
        _sessions[sid] = {"phase": "selecting_paper"}
        yield {"type": "start", "session_id": sid}
        yield from _list_papers(sid)
        return

    session = _sessions[session_id]

    if session["phase"] == "selecting_paper":
        yield from _select_paper(session_id, user_input)
    elif session["phase"] == "answering":
        if _is_command(user_input):
            yield from _switch_paper(session_id)
        elif _is_answer_like(user_input):
            yield from _grade_batch(session_id, user_input)
        else:
            yield from _answer_quiz_question(session_id, user_input)
    elif session["phase"] == "graded":
        if _is_command(user_input):
            yield from _switch_paper(session_id)
        else:
            yield from _present_batch(session_id)
    elif session["phase"] == "done":
        if _is_command(user_input):
            yield from _switch_paper(session_id)
        else:
            yield {"type": "start", "session_id": session_id}
            msg = "所有题目已完成！可以输入「换」重新选卷，或切换模式。"
            yield {"type": "content", "content": msg}
            yield {"type": "done", "answer": msg, "session_id": session_id}


def _switch_paper(session_id):
    """回到试卷选择"""
    _sessions[session_id] = {"phase": "selecting_paper"}
    yield {"type": "start", "session_id": session_id}
    yield from _list_papers(session_id)


def _answer_quiz_question(session_id, user_input):
    """在刷题中回答用户提问"""
    yield {"type": "start", "session_id": session_id}
    yield {"type": "phase", "phase": "answering", "message": "💬 回答问题"}

    messages = [
        {"role": "system", "content": "你是课程教师，正在给学生刷题。学生有问题要问，请简洁回答。"},
        {"role": "user", "content": user_input},
    ]

    full_text = ""
    for chunk in stream_chat(messages, temperature=0.5):
        full_text += chunk
        yield {"type": "content", "content": chunk}

    reminder = "\n\n---\n回答完上方问题，请继续作答当前题目。"
    yield {"type": "content", "content": reminder}
    full_text += reminder

    yield {"type": "done", "answer": full_text, "session_id": session_id}


def _list_papers(session_id):
    """列出所有试卷供用户选择"""
    papers = _get_paper_list()
    if not papers:
        yield {"type": "error", "msg": "习题库为空或无法加载"}
        return

    yield {"type": "phase", "phase": "selecting_paper", "message": "📝 请选择试卷"}

    lines = ["## 选择试卷\n"]
    for i, (source, info) in enumerate(papers.items(), 1):
        course = info["course"] or "未知"
        count_label = f"{info['choice']}选择+{info['complex']}综合"
        practice = info["practice_count"]
        practice_label = f"（已练习{practice}次）" if practice > 0 else ""
        lines.append(f"**{i}.** {source} [{course}] {count_label}{practice_label}\n\n")

    lines.append("\n输入编号或试卷名称开始刷题。")
    text = "".join(lines)
    yield {"type": "content", "content": text}
    yield {"type": "done", "answer": text, "session_id": session_id}

    _sessions[session_id]["paper_list"] = list(papers.keys())


def _select_paper(session_id, user_input):
    """用户选择试卷后初始化刷题"""
    session = _sessions[session_id]
    paper_list = session.get("paper_list", [])

    selected = None

    # 尝试数字编号匹配
    num = re.search(r"\d+", user_input.strip())
    if num:
        idx = int(num.group()) - 1
        if 0 <= idx < len(paper_list):
            selected = paper_list[idx]

    # 尝试名称匹配
    if not selected:
        for src in paper_list:
            if src in user_input or user_input.strip() in src:
                selected = src
                break

    if not selected:
        yield {"type": "error", "msg": f"未匹配到试卷，请输入编号(1-{len(paper_list)})或试卷名称"}
        return

    # 初始化刷题
    all_questions = _load_question_bank()
    questions = [q for q in all_questions if q.get("source") == selected]

    choice_qs = [q for q in questions if q.get("type") == "选择题"]
    complex_qs = [q for q in questions if q.get("type") == "综合题"]
    other_qs = [q for q in questions if q.get("type") not in ("选择题", "综合题")]
    complex_qs.extend(other_qs)

    session["source"] = selected
    session["choice_questions"] = choice_qs
    session["complex_questions"] = complex_qs
    session["current_type"] = "选择题"
    session["choice_idx"] = 0
    session["complex_idx"] = 0
    session["phase"] = "presenting"
    session["wrong_questions"] = {}
    session["batch_num"] = 0
    session["current_batch"] = []

    logger.info("刷题会话创建: source=%s, choice=%d, complex=%d",
                selected, len(choice_qs), len(complex_qs))
    yield from _present_batch(session_id)


def _get_next_batch(session):
    """获取下一批题目"""
    if session["current_type"] == "选择题":
        idx = session["choice_idx"]
        remaining = session["choice_questions"][idx:]
        if remaining:
            return remaining[:min(5, len(remaining))]
        session["current_type"] = "综合题"
        session["complex_idx"] = 0

    if session["current_type"] == "综合题":
        idx = session["complex_idx"]
        remaining = session["complex_questions"][idx:]
        if remaining:
            return [remaining[0]]

    return []


def _present_batch(session_id):
    """展示当前批次题目"""
    session = _sessions[session_id]
    batch = _get_next_batch(session)

    yield {"type": "start", "session_id": session_id}

    if not batch:
        session["phase"] = "done"
        summary = _generate_summary(session)
        yield {"type": "phase", "phase": "done", "message": "🎉 所有题目完成！"}
        yield {"type": "content", "content": summary}
        yield {"type": "done", "answer": summary, "session_id": session_id}
        return

    q_type = batch[0].get("type", "题")
    session["batch_num"] += 1
    batch_num = session["batch_num"]

    # 错题优先插入
    wrong_qids = [qid for qid, count in session["wrong_questions"].items() if count > 0]
    if wrong_qids and session["current_type"] == "选择题" and random.random() < 0.6:
        wrong_pool = [q for q in session["choice_questions"] if q.get("qid") in wrong_qids]
        if wrong_pool and wrong_pool[0] not in batch:
            batch.insert(0, wrong_pool[0])

    session["current_batch"] = batch
    session["phase"] = "answering"

    yield {"type": "phase", "phase": "presenting", "message": f"📝 {q_type}（第{batch_num}组）"}

    header = f"## {q_type}（第{batch_num}组）\n\n来源: {session['source']}\n\n"
    yield {"type": "content", "content": header}
    full_text = header

    for i, q in enumerate(batch):
        img_url = _get_image_url(q)
        qid = q.get("qid", f"Q{i+1}")
        topic = q.get("topic", "")

        q_header = f"**{qid}**"
        if topic:
            q_header += f" [{topic}]"
        q_header += "\n\n"
        yield {"type": "content", "content": q_header}
        full_text += q_header

        if img_url:
            img_md = f"![{qid}]({img_url})\n\n"
            yield {"type": "content", "content": img_md}
            full_text += img_md

    if q_type == "选择题":
        hint = "---\n请依次输入答案（如: A B C D E），不会的题目请输入「不会」\n"
    else:
        hint = "---\n请输入你的答案，不会的题目请输入「不会」\n"
    yield {"type": "content", "content": hint}
    full_text += hint

    yield {"type": "done", "answer": full_text, "session_id": session_id}


def _grade_batch(session_id, user_input):
    """批改当前批次"""
    session = _sessions[session_id]
    batch = session.get("current_batch", [])

    yield {"type": "start", "session_id": session_id}
    yield {"type": "phase", "phase": "grading", "message": "✍️ 批改中..."}

    if batch and batch[0].get("type") == "选择题":
        answers = _parse_choice_answers(user_input, len(batch))
    else:
        answers = [user_input]

    full_text = "## 批改结果\n\n"
    yield {"type": "content", "content": full_text}

    correct_count = 0
    for i, q in enumerate(batch):
        user_ans = answers[i] if i < len(answers) else "未作答"
        correct_ans = q.get("answer", "")
        qid = q.get("qid", f"Q{i+1}")

        if "不会" in str(user_ans):
            is_correct = False
            verdict = "✗ 不会"
        elif q.get("type") == "选择题":
            is_correct = user_ans.upper().strip() == correct_ans.upper().strip()
            verdict = "✓" if is_correct else "✗"
        else:
            is_correct = _llm_grade_complex(q, user_ans)
            verdict = "✓" if is_correct else "✗"

        if not is_correct:
            session["wrong_questions"][qid] = session["wrong_questions"].get(qid, 0) + 1
        else:
            correct_count += 1

        result_text = f"**{qid}**: {verdict}\n"
        if not is_correct:
            ref_points = q.get("reference_points", [])
            ref_text = "\n".join(ref_points) if ref_points else correct_ans
            try:
                explain = chat([
                    {"role": "system", "content": "你是批改作业的教师。"},
                    {"role": "user", "content": QUIZ_GRADE_PROMPT.format(
                        q_type=q.get("type", "题"),
                        question_text=q.get("text", "")[:500],
                        correct_answer=ref_text,
                        user_answer=str(user_ans),
                    )},
                ], temperature=0.3)
            except Exception as e:
                explain = f"解析生成失败: {str(e)[:100]}"
            result_text += f"正确答案: {correct_ans}\n解析: {explain}\n\n"
        else:
            result_text += "\n"

        yield {"type": "content", "content": result_text}
        full_text += result_text

    total = len(batch)
    summary = f"\n---\n**本组正确率: {correct_count}/{total}**\n"

    wrong_qids = [qid for qid, count in session["wrong_questions"].items() if count > 0]
    if wrong_qids:
        summary += f"**错题集** ({len(wrong_qids)}题): {', '.join(wrong_qids)}\n"
        summary += "错题将在后续优先出现\n"

    summary += "\n输入「继续」做下一组题目\n"
    yield {"type": "content", "content": summary}
    full_text += summary

    if session["current_type"] == "选择题":
        session["choice_idx"] += len(batch)
    else:
        session["complex_idx"] += len(batch)

    session["phase"] = "graded"
    yield {"type": "done", "answer": full_text, "session_id": session_id}


def _parse_choice_answers(text, count):
    """解析选择题答案"""
    text = text.strip()
    tokens = re.split(r"[\s,，、;；]+", text)
    answers = []
    for t in tokens:
        t = t.strip().upper()
        m = re.match(r"(\d+)\s*[.．、]?\s*([A-D])", t)
        if m:
            answers.append(m.group(2))
        elif re.match(r"^[A-D]$", t):
            answers.append(t)
        elif "不会" in t or "不知道" in t:
            answers.append("不会")

    if not answers and len(text) <= count:
        for ch in text.upper():
            if ch in "ABCD":
                answers.append(ch)
        if "不会" in text:
            answers = ["不会"]

    return answers[:count] if answers else ["未作答"] * count


def _llm_grade_complex(question, user_answer):
    """用 LLM 判断综合题答案"""
    correct = question.get("answer", "")
    ref_points = question.get("reference_points", [])
    if not correct and ref_points:
        correct = "\n".join(ref_points)
    if not correct:
        return True

    try:
        result = chat([
            {"role": "system", "content": "你是批改作业的教师。判断学生答案是否正确。只回答'对'或'错'。"},
            {"role": "user", "content": f"题目: {question.get('text', '')[:300]}\n参考答案: {correct[:500]}\n学生答案: {str(user_answer)[:500]}\n判断: "},
        ], temperature=0.1)
        return "对" in result
    except Exception:
        return True


def _generate_summary(session):
    """生成刷题总结"""
    total_choice = len(session["choice_questions"])
    total_complex = len(session["complex_questions"])
    choice_done = min(session["choice_idx"], total_choice)
    complex_done = min(session["complex_idx"], total_complex)
    wrong_count = len([1 for c in session["wrong_questions"].values() if c > 0])

    # 记录练习次数
    stats = _load_stats()
    source = session["source"]
    stats[source] = stats.get(source, 0) + 1
    _save_stats(stats)
    practice_count = stats[source]

    summary = f"""## 刷题总结

**试卷**: {source}

**完成情况**:
- 选择题: {choice_done}/{total_choice}
- 综合题: {complex_done}/{total_complex}

**错题数**: {wrong_count}

**本套试卷累计练习**: {practice_count}次

"""
    if session["wrong_questions"]:
        summary += "**错题列表**:\n"
        for qid, count in session["wrong_questions"].items():
            if count > 0:
                summary += f"- {qid} (错{count}次)\n"

    summary += "\n可以输入「开始刷题」重新开始，或切换到其他模式。"
    return summary
