"""快速验证 LangGraph 项目骨架

从项目根目录运行: python test_graph.py
"""

from paper_agent.graph import graph
from langchain_core.messages import HumanMessage


def main():
    print("=" * 60)
    print("课程知识库问答 Agent 骨架验证")
    print("=" * 60)

    question = "什么是ReAct？帮我查一下"
    print(f"\n用户提问: {question}\n")
    print("-" * 60)

    result = graph.invoke({"messages": [HumanMessage(content=question)]})

    print("\n对话消息流:")
    print("-" * 60)
    for i, msg in enumerate(result["messages"]):
        role = msg.type.upper()
        content = msg.content[:200] if msg.content else ""
        tool_calls = getattr(msg, "tool_calls", None)
        tool_info = ""
        if tool_calls:
            tool_info = f"  [调用工具: {[tc['name'] for tc in tool_calls]}]"
        print(f"  [{i}] {role}{tool_info}")
        if content:
            print(f"      {content}")
        print()

    print("=" * 60)
    print("骨架验证完成! 图能正常执行 ReAct 循环。")
    print("=" * 60)


if __name__ == "__main__":
    main()
