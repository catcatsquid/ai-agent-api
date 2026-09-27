from langchain_core.tools import tool
from rag_tool import search_knowledge as _rag_search
from rag_tool import get_knowledge_base


@tool
def search_knowledge(query: str) -> str:
    """搜索课程知识库（计算机网络+操作系统）。输入问题或关键词，返回相关的知识点内容。

    知识库包含:
    - 计算机网络：TCP三次握手/四次挥手、TCP与UDP区别、拥塞控制等
    - 操作系统：进程与线程、调度算法、死锁、同步互斥机制等

    Args:
        query: 搜索关键词或问题，如 "TCP三次握手" 或 "死锁的四个必要条件"
    """
    return _rag_search(query, top_k=3)


@tool
def get_course_info() -> str:
    """查看知识库信息：包含哪些课程文档、共多少条知识点。"""
    kb = get_knowledge_base()
    count = kb.count()
    if count == 0:
        return "知识库为空。请先运行 python build_knowledge.py 建库。"
    return f"知识库当前有 {count} 条文档分块，涵盖计算机网络和操作系统两门课程。"


# 工具列表 - 传给 LLM 做函数调用
tools = [search_knowledge, get_course_info]

# 工具名 → 函数 的映射
tools_by_name = {t.name: t for t in tools}
