"""章节匹配共享工具 — learn.py 和 review.py 共用"""

import re

from course_agent.llm import chat
from rag_tool import get_knowledge_base
from course_agent.logger import logger


def _chinese_to_int(s):
    """中文数字或阿拉伯数字转int"""
    if s.isdigit():
        return int(s)
    mapping = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5,
              "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    return mapping.get(s, 0)


CHAPTER_INDEX_MAP = {
    1: ["课程简介", "Introduction", "进程与线程"],
    2: ["进程与线程", "OS调度"],
    3: ["OS调度", "并发与同步"],
    4: ["并发与同步", "死锁"],
    5: ["死锁", "内存管理与虚拟内存"],
    6: ["内存管理与虚拟内存"],
}

_SYNONYMS = {
    "进程": "进程与线程", "线程": "进程与线程",
    "调度": "OS调度", "CPU调度": "OS调度",
    "并发": "并发与同步", "同步": "并发与同步", "互斥": "并发与同步",
    "死锁": "死锁",
    "内存": "内存管理与虚拟内存", "虚拟内存": "内存管理与虚拟内存",
    "分页": "内存管理与虚拟内存", "分段": "内存管理与虚拟内存",
}


def get_chapters():
    """获取知识库中所有章节"""
    try:
        kb = get_knowledge_base()
        stats = kb.get_stats()
        return list(stats.get("by_chapter", {}).keys())
    except Exception as e:
        logger.error("获取章节列表失败: %s", e)
        return []


def match_chapter(user_input, chapters):
    """从用户输入中匹配知识库章节（直接→关键词→同义词→序号）"""
    for ch in chapters:
        if ch in user_input:
            return ch
    for ch in chapters:
        keywords = re.split(r"[\s与和及]", ch)
        matched = sum(1 for kw in keywords if len(kw) > 1 and kw in user_input)
        if matched > 0 and matched >= len(keywords) * 0.5:
            return ch
    for keyword, chapter in _SYNONYMS.items():
        if keyword in user_input and chapter in chapters:
            return chapter
    m = re.search(r"第([一二三四五六七八九十\d]+)章", user_input)
    if m:
        idx = _chinese_to_int(m.group(1))
        for ch in CHAPTER_INDEX_MAP.get(idx, []):
            if ch in chapters:
                return ch
    return None


def llm_match_chapter(user_input, chapters):
    """用 LLM 匹配章节（作为最后兜底）"""
    prompt = (
        f"用户想学习: {user_input}\n"
        f"可选章节:\n{chr(10).join(chapters)}\n"
        "只输出匹配的章节名称，无匹配则输出'无'。"
    )
    try:
        result = chat([
            {"role": "system", "content": "你是章节匹配器。"},
            {"role": "user", "content": prompt},
        ], temperature=0.1)
        result = result.strip().strip("《》\"'")
        if result in chapters:
            return result
    except Exception as e:
        logger.warning("LLM 章节匹配失败: %s", e)
    return None


def match_chapter_with_fallback(user_input, chapters):
    """匹配章节，含 LLM 兜底，返回 (chapter, fail_msg)"""
    if not chapters:
        return None, "知识库为空，请先导入课程文档"
    chapter = match_chapter(user_input, chapters)
    if not chapter:
        chapter = llm_match_chapter(user_input, chapters)
    if not chapter:
        return None, "无法识别章节，请从以下章节中选择"
    return chapter, ""


def format_chapter_list(chapters):
    """格式化章节列表为可读字符串"""
    return "\n".join(f"  - {ch}" for ch in chapters)


def get_chapter_content(chapter):
    """检索某章节的全部知识库 chunk"""
    try:
        kb = get_knowledge_base()
        collection = kb._get_collection()
        results = collection.get(where={"chapter": chapter})
        chunks = []
        for i, doc in enumerate(results["documents"]):
            meta = results["metadatas"][i]
            chunks.append({
                "text": doc,
                "source": meta.get("source", ""),
                "doc_type": meta.get("doc_type", ""),
            })
        return chunks
    except Exception as e:
        logger.error("检索章节内容失败: %s", e)
        return []
