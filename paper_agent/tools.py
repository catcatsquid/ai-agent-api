"""工具定义

Day 2 占位：一个简单的知识库搜索工具，验证骨架能跑通。
Day 3-4 会替换为真实工具集：
  - search_knowledge: 检索知识库（接入第2周的 RAG 检索）
  - get_course_list: 获取课程列表
  - get_course_outline: 获取课程大纲

对比第3周 agent_api.py 里手写的工具：
  1. 手动拼 TOOLS 字典（name/description/parameters JSON Schema）
  2. 手动写 TOOL_MAP 映射
  3. 手动 json.loads(tool_call["function"]["arguments"]) 解析参数

LangChain 的 @tool 装饰器一步到位：
  - 自动从函数签名生成 JSON Schema
  - 自动从 docstring 提取 description
  - 直接调用 tool.invoke(args) 即可
"""

from langchain_core.tools import tool


@tool
def search_knowledge(query: str) -> str:
    """搜索课程知识库。输入问题或关键词，返回相关的知识点和解答。

    Args:
        query: 搜索关键词或问题，如 "什么是RAG" 或 "ReAct循环流程"
    """
    # 占位实现 - Day 3 替换为第2周的 RAG 检索
    return (
        f"知识库检索 '{query}' 返回以下内容（占位响应）:\n"
        f"1. {query} 的基本概念与定义\n"
        f"2. {query} 的核心流程与关键步骤\n"
        f"3. {query} 的常见应用场景\n"
        f"（Day 3-4 将接入第2周 RAG 检索 + 真实课程文档）"
    )


# 工具列表 - 传给 LLM 做函数调用
tools = [search_knowledge]

# 工具名 -> 函数的映射（工具节点用）
tools_by_name = {t.name: t for t in tools}
