"""工具定义

工具列表:
  - search_knowledge: 检索 Chroma 知识库（计网+OS课程笔记）
  - web_search: 联网搜索（智谱 web_search），知识库不够时补充
  - get_course_info: 获取知识库统计信息
"""

import os
import requests

from langchain_core.tools import tool

from rag_tool import search_knowledge as _search_knowledge
from rag_tool import get_knowledge_base
from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL


@tool
def search_knowledge(query: str) -> str:
    """搜索课程知识库（计算机网络+操作系统）。输入问题或关键词，返回相关的知识点内容。

    知识库包含: TCP/UDP、拥塞控制、进程调度、死锁、同步互斥等课程内容。

    Args:
        query: 搜索关键词或问题，如 "TCP三次握手" 或 "死锁的四个必要条件"
    """
    return _search_knowledge(query, top_k=3)


@tool
def web_search(query: str) -> str:
    """联网搜索获取互联网上的最新信息。当知识库中没有找到相关内容，或问题涉及课程之外的知识时使用。

    Args:
        query: 搜索关键词，如 "2024年最新TCP拥塞控制算法" 或 "Linux内核调度器CFS原理"
    """
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
    data = resp.json()
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


# 工具列表 - 传给 LLM 做函数调用
tools = [search_knowledge, web_search, get_course_info]

# 工具名 -> 函数的映射（工具节点用）
tools_by_name = {t.name: t for t in tools}
