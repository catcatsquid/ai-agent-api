"""ReAct 工具环的轮数上限（F8）

背景：should_continue 原来只看"最后一条消息有没有 tool_calls"、tool_node 也不计数，
也就是说循环次数完全由模型决定，唯一的上限是 LangGraph 默认的 recursion_limit=10007。
实测模型每轮都要求调工具时，抛错前真实发生 LLM 调用 5004 次、工具调用 5003 次。
"""
import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import END

from course_agent import nodes


def _ai_with_tool_call(call_id="c1"):
    return AIMessage(content="", tool_calls=[
        {"id": call_id, "name": "search_knowledge", "args": {"query": "死锁"}}
    ])


class TestRoundCap:
    def test_cap_value_is_business_level(self):
        """上限必须是个位数级别，不是框架默认的上万"""
        assert nodes.MAX_TOOL_ROUNDS <= 10

    def test_should_continue_routes_to_tools_when_under_cap(self):
        state = {"messages": [_ai_with_tool_call()], "tool_rounds": 0}
        assert nodes.should_continue(state) == "tools"

    def test_should_continue_ends_at_cap(self):
        """到达上限时即使模型还在要求调工具也必须收敛"""
        state = {"messages": [_ai_with_tool_call()], "tool_rounds": nodes.MAX_TOOL_ROUNDS}
        assert nodes.should_continue(state) == END

    def test_should_continue_ends_beyond_cap(self):
        state = {"messages": [_ai_with_tool_call()], "tool_rounds": nodes.MAX_TOOL_ROUNDS + 5}
        assert nodes.should_continue(state) == END

    def test_should_continue_ends_when_no_tool_calls(self):
        state = {"messages": [AIMessage(content="直接回答")], "tool_rounds": 0}
        assert nodes.should_continue(state) == END

    def test_missing_tool_rounds_counts_as_zero(self):
        """首次进入时 state 里没有 tool_rounds，不能因此报错或直接收敛"""
        state = {"messages": [_ai_with_tool_call()]}
        assert nodes.should_continue(state) == "tools"

    def test_tool_node_increments_rounds(self):
        state = {"messages": [_ai_with_tool_call()], "tool_rounds": 2}
        with patch.object(nodes, "safe_tool_invoke", return_value="结果"):
            out = nodes.tool_node(state)
        assert out["tool_rounds"] == 3
        assert isinstance(out["messages"][0], ToolMessage)

    def test_tool_node_handles_unknown_tool_and_still_counts(self):
        msg = AIMessage(content="", tool_calls=[{"id": "c9", "name": "不存在的工具", "args": {}}])
        out = nodes.tool_node({"messages": [msg], "tool_rounds": 0})
        assert out["tool_rounds"] == 1
        assert "未知工具" in out["messages"][0].content

    def test_llm_node_drops_tools_when_exhausted(self):
        """轮数用尽时要改用不带工具的 llm，逼它给最终答案

        否则循环会停在一个 tool 消息上，前端拿到的就是工具原文（也是泄题通道）。
        """
        state = {"messages": [HumanMessage(content="问")], "tool_rounds": nodes.MAX_TOOL_ROUNDS}
        with patch.object(nodes, "llm") as plain, \
             patch.object(nodes, "llm_with_tools") as with_tools, \
             patch.object(nodes, "safe_llm_invoke",
                          return_value=AIMessage(content="最终答案")) as invoke:
            nodes.llm_node(state)
        assert invoke.call_count == 1
        called_with = invoke.call_args[0][0]
        assert called_with is plain, "轮数用尽时应该用不带工具的 llm"
        assert with_tools.call_count == 0

    def test_llm_node_keeps_tools_under_cap(self):
        state = {"messages": [HumanMessage(content="问")], "tool_rounds": 1}
        with patch.object(nodes, "llm") as plain, \
             patch.object(nodes, "llm_with_tools") as with_tools, \
             patch.object(nodes, "safe_llm_invoke",
                          return_value=AIMessage(content="答案")) as invoke:
            nodes.llm_node(state)
        assert invoke.call_args[0][0] is with_tools
        assert plain.call_count == 0
