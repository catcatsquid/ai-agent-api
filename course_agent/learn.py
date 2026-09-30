"""学习模式 - 逐节教学 + 小测 + 批改 + 章节总复习

状态流转:
  新会话: 解析课程章节 → KB检索 → LLM生成大纲 → 教第1节 + 出小测题
  quiz_pending: 用户提交答案 → 批改讲解
  graded: 用户说"继续" → 教下一节(+小测) 或 章节总复习
  done: 本章学习完成
"""

import re
import uuid

from course_agent.llm import stream_chat, chat
from course_agent.chapter_utils import (
    get_chapters, match_chapter_with_fallback, format_chapter_list,
    get_chapter_content,
)
from rag_tool import get_knowledge_base
from course_agent.logger import logger

_sessions = {}

TEACHING_PROMPT = """你是一位经验丰富的大学教师，正在给学生上课。
请根据知识库内容，用通俗易懂的方式讲解本节知识点。

要求:
1. 从基础概念开始，逐步深入
2. 用生活类比帮助理解抽象概念
3. 关键术语要解释清楚
4. 适当使用例子
5. 控制篇幅，重点突出
6. 用 Markdown 格式输出
7. 只讲解知识点，不要介绍讲师、联系方式、课程大纲等元信息
8. 不要照读知识库原文，要用自己的话讲解
"""

OUTLINE_PROMPT = """请根据以下知识库内容，为《{chapter}》这一章生成分节教学大纲。

要求:
1. 分成 3-5 个小节，按教学顺序排列
2. 每个小节标题简洁明了
3. 覆盖本章核心知识点
4. 忽略讲师介绍、联系方式、课程大纲等非知识内容

知识库内容:
{context}

只输出小节标题，每行一个，不要编号，不要额外说明。
"""

QUIZ_PROMPT = """请基于以下教学内容，出3道小测题（2道选择题 + 1道简答题）。

教学内容:
{teaching_content}

要求:
1. 题目覆盖本节核心知识点
2. 选择题提供4个选项(A/B/C/D)
3. 难度适中
4. 绝对不要在题目中泄露答案
5. 不要在题目后附带答案或解析

格式:
## 本节小测

**第1题（选择题）**
[题目内容]
A. 选项A
B. 选项B
C. 选项C
D. 选项D

**第2题（选择题）**
[题目内容]
A. ...
B. ...
C. ...
D. ...

**第3题（简答题）**
[题目内容]

---
请作答后发送答案，选择题格式如 1.A 2.B，简答题直接写出你的回答。
"""

GRADING_PROMPT = """请批改以下小测题:

题目:
{quiz_content}

学生答案:
{user_answers}

逐题批改，格式:
## 批改结果

**第1题**: ✓/✗
正确答案: X
解析: ...

**第2题**: ✓/✗
...

**第3题**: ✓/✗
...

## 总结
正确率: X/3
{建议}
"""

REVIEW_PROMPT = """请为《{chapter}》这一章做总复习。

本章小节:
{sections}

知识库内容:
{context}

要求:
1. 梳理核心知识框架
2. 每个知识点一句话总结
3. 标注常见考点和易错点

格式:
## 《{chapter}》章节总复习

### 知识框架
...

### 核心知识点
...

### 常见考点与易错点
...
"""


def process_message(session_id, user_input):
    """处理学习模式消息，yield SSE dict 事件"""

    if not session_id or session_id not in _sessions:
        session_id, fail_msg = _start_session(user_input)
        if not session_id:
            chapters = get_chapters()
            if chapters:
                chapter_list = format_chapter_list(chapters)
                msg = f"{fail_msg}\n\n📖 可选章节:\n{chapter_list}\n\n请直接说出章节名称，如：学进程与线程"
            else:
                msg = "知识库为空，请先导入课程文档"
            yield {"type": "error", "msg": msg}
            return
        yield from _teach_and_quiz(session_id)
        return

    session = _sessions[session_id]

    if session["phase"] == "quiz_pending":
        if _is_quiz_answer(user_input):
            yield from _grade_quiz(session_id, user_input)
        else:
            yield from _answer_question(session_id, user_input)
    elif session["phase"] == "graded":
        if "继续" in user_input or "下一" in user_input:
            if session["current_idx"] < len(session["sections"]) - 1:
                session["current_idx"] += 1
                yield from _teach_and_quiz(session_id)
            else:
                yield from _chapter_review(session_id)
        elif "复习" in user_input:
            yield from _chapter_review(session_id)
        else:
            yield from _answer_question(session_id, user_input)
    elif session["phase"] in ("review", "done"):
        yield {"type": "start", "session_id": session_id}
        msg = "本章学习已完成！可以开始学习其他章节，或切换到复习模式刷题。"
        yield {"type": "content", "content": msg}
        yield {"type": "done", "answer": msg, "session_id": session_id}


def _start_session(user_input):
    """开始新学习会话，返回 (session_id, fail_msg)"""
    chapters = get_chapters()
    if not chapters:
        logger.error("知识库为空")
        return None, "知识库为空，请先导入课程文档"

    chapter, fail_msg = match_chapter_with_fallback(user_input, chapters)
    if not chapter:
        return None, fail_msg

    chunks = get_chapter_content(chapter)
    if not chunks:
        try:
            results = get_knowledge_base().search(chapter, top_k=15)
            chunks = results
        except Exception:
            pass
    if not chunks:
        return None, f"章节《{chapter}》在知识库中没有内容"

    context = "\n\n".join(c["text"] for c in chunks[:20])

    try:
        outline = chat([
            {"role": "system", "content": "你是课程规划专家。"},
            {"role": "user", "content": OUTLINE_PROMPT.format(chapter=chapter, context=context)},
        ], temperature=0.3)
    except Exception as e:
        logger.error("生成大纲失败: %s", e)
        return None, "大纲生成失败，请重试"

    sections = [s.strip() for s in outline.strip().split("\n") if s.strip()]
    sections = [re.sub(r"^[\d.、\-\*]+", "", s).strip() for s in sections]
    sections = [s for s in sections if len(s) > 1 and not s.startswith("#")]

    if not sections:
        sections = [chapter]

    session_id = str(uuid.uuid4())
    _sessions[session_id] = {
        "chapter": chapter,
        "sections": sections,
        "current_idx": 0,
        "phase": "teaching",
        "context": context,
        "teaching_content": "",
        "quiz_questions": "",
    }

    logger.info("学习会话创建: chapter=%s, sections=%s", chapter, sections)
    return session_id, ""


def _teach_and_quiz(session_id):
    """教学当前节 + 生成小测题"""
    session = _sessions[session_id]
    idx = session["current_idx"]
    section = session["sections"][idx]
    chapter = session["chapter"]
    context = session["context"]

    yield {"type": "start", "session_id": session_id}
    yield {
        "type": "phase", "phase": "teaching",
        "section": section, "section_idx": idx + 1,
        "total_sections": len(session["sections"]),
        "chapter": chapter,
        "message": f"📖 第{idx+1}/{len(session['sections'])}节: {section}",
    }

    messages = [
        {"role": "system", "content": TEACHING_PROMPT},
        {"role": "user", "content": f"请讲解《{chapter}》的第{idx+1}节：{section}\n\n知识库参考:\n{context}"},
    ]

    teaching_text = ""
    for chunk in stream_chat(messages, temperature=0.7):
        teaching_text += chunk
        yield {"type": "content", "content": chunk}

    yield {"type": "content", "content": "\n\n---\n\n"}
    teaching_text += "\n\n---\n\n"

    yield {"type": "phase", "phase": "quiz", "message": "📝 本节小测"}

    quiz_messages = [
        {"role": "system", "content": "你是课程教师。"},
        {"role": "user", "content": QUIZ_PROMPT.format(teaching_content=teaching_text)},
    ]

    quiz_text = ""
    for chunk in stream_chat(quiz_messages, temperature=0.5):
        quiz_text += chunk
        yield {"type": "content", "content": chunk}

    session["teaching_content"] = teaching_text
    session["quiz_questions"] = quiz_text
    session["phase"] = "quiz_pending"

    yield {"type": "done", "answer": teaching_text + quiz_text, "session_id": session_id}


def _grade_quiz(session_id, user_input):
    """批改小测题"""
    session = _sessions[session_id]
    quiz_content = session.get("quiz_questions", "")

    yield {"type": "start", "session_id": session_id}
    yield {"type": "phase", "phase": "grading", "message": "✍️ 批改中..."}

    messages = [
        {"role": "system", "content": "你是批改作业的教师。"},
        {"role": "user", "content": GRADING_PROMPT.format(quiz_content=quiz_content, user_answers=user_input)},
    ]

    full_text = ""
    for chunk in stream_chat(messages, temperature=0.3):
        full_text += chunk
        yield {"type": "content", "content": chunk}

    if session["current_idx"] >= len(session["sections"]) - 1:
        hint = "\n\n---\n输入「继续」进行本章总复习"
    else:
        hint = "\n\n---\n输入「继续」学习下一节"
    yield {"type": "content", "content": hint}
    full_text += hint

    session["phase"] = "graded"
    yield {"type": "done", "answer": full_text, "session_id": session_id}


def _answer_question(session_id, user_input):
    """回答用户在教学中提出的问题"""
    session = _sessions[session_id]
    chapter = session["chapter"]
    context = session["context"]
    teaching = session.get("teaching_content", "")

    yield {"type": "start", "session_id": session_id}

    messages = [
        {"role": "system", "content": f"你是大学教师，正在教《{chapter}》。回答学生的问题。"},
        {"role": "user", "content": f"已教内容:\n{teaching}\n\n知识库参考:\n{context}\n\n学生提问: {user_input}"},
    ]

    full_text = ""
    for chunk in stream_chat(messages, temperature=0.7):
        full_text += chunk
        yield {"type": "content", "content": chunk}

    if session["phase"] == "quiz_pending":
        reminder = "\n\n---\n请完成上方的小测题，发送答案后我会批改。"
        yield {"type": "content", "content": reminder}
        full_text += reminder

    yield {"type": "done", "answer": full_text, "session_id": session_id}


def _chapter_review(session_id):
    """章节总复习"""
    session = _sessions[session_id]
    chapter = session["chapter"]
    sections = session["sections"]
    context = session["context"]

    yield {"type": "start", "session_id": session_id}
    yield {"type": "phase", "phase": "review", "message": "📋 本章总复习"}

    sections_text = "\n".join(f"  {i+1}. {s}" for i, s in enumerate(sections))
    messages = [
        {"role": "system", "content": "你是课程教师。"},
        {"role": "user", "content": REVIEW_PROMPT.format(chapter=chapter, sections=sections_text, context=context)},
    ]

    full_text = ""
    for chunk in stream_chat(messages, temperature=0.5):
        full_text += chunk
        yield {"type": "content", "content": chunk}

    session["phase"] = "done"

    yield {"type": "done", "answer": full_text, "session_id": session_id}


def _is_quiz_answer(text):
    """判断用户输入是否是测验答案"""
    text = text.strip()
    if "不会" in text or "不知道" in text:
        return True
    if re.match(r"^[A-Da-d\s.、,]+$", text) and len(text) <= 20:
        return True
    if re.search(r"\d+\s*[.．、]?\s*[A-Da-d]", text):
        return True
    if re.search(r"答案\s*[是为]?\s*[A-Da-d]", text):
        return True
    return False
