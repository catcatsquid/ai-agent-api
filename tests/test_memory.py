"""对话记忆模块测试"""

import pytest
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)

from course_agent.memory import (
    estimate_tokens,
    count_messages_tokens,
    trim_messages,
    count_turns,
    get_last_human_message,
    _format_conversation,
)


# ============================================================
# Token 估算测试
# ============================================================

class TestTokenEstimation:
    """Token 估算测试"""

    def test_empty_text(self):
        """空文本 token 数为 0"""
        assert estimate_tokens("") == 0

    def test_short_chinese(self):
        """短中文文本估算"""
        # 5个中文字，除以2 ≈ 2
        tokens = estimate_tokens("你好世界")
        assert tokens >= 1
        assert tokens <= 5

    def test_long_text_more_tokens(self):
        """长文本 token 更多"""
        short = estimate_tokens("hello")
        long = estimate_tokens("hello world this is a long text")
        assert long > short


# ============================================================
# 滑动窗口裁剪测试
# ============================================================

class TestTrimMessages:
    """滑动窗口裁剪测试"""

    def _make_messages(self, n: int):
        """生成 n 条交替的用户/AI 消息"""
        msgs = []
        for i in range(n):
            if i % 2 == 0:
                msgs.append(HumanMessage(content=f"问题{i}"))
            else:
                msgs.append(AIMessage(content=f"回答{i}"))
        return msgs

    def test_under_limit_no_trim(self):
        """没超过限制不裁剪"""
        msgs = self._make_messages(5)
        result = trim_messages(msgs, max_messages=10)
        assert len(result) == 5

    def test_over_limit_trims_old(self):
        """超过限制时裁剪最老的消息"""
        msgs = self._make_messages(20)  # 20 条
        result = trim_messages(msgs, max_messages=10)
        assert len(result) == 10
        # 最后一条应该是原来的最后一条（保留最新的）
        assert result[-1].content == "回答19"
        # 第一条应该是第 10 条（裁掉了前 10 条）
        assert result[0].content == "问题10"

    def test_keeps_system_message(self):
        """系统消息永远保留"""
        msgs = [SystemMessage(content="系统提示")] + self._make_messages(20)
        result = trim_messages(msgs, max_messages=10)
        # 系统消息 + 10 条普通消息
        assert len(result) == 11
        assert isinstance(result[0], SystemMessage)
        assert result[0].content == "系统提示"

    def test_keep_system_false(self):
        """keep_system=False 时不保留系统消息"""
        msgs = [SystemMessage(content="系统提示")] + self._make_messages(5)
        result = trim_messages(msgs, max_messages=10, keep_system=False)
        assert len(result) == 5
        assert not any(isinstance(m, SystemMessage) for m in result)

    def test_only_system_messages(self):
        """只有系统消息的情况"""
        msgs = [SystemMessage(content="提示1"), SystemMessage(content="提示2")]
        result = trim_messages(msgs, max_messages=1)
        # 系统消息都保留（keep_system=True）
        assert len(result) == 2


# ============================================================
# 对话轮数统计
# ============================================================

class TestCountTurns:
    """对话轮数统计测试"""

    def test_empty(self):
        assert count_turns([]) == 0

    def test_single_turn(self):
        msgs = [HumanMessage(content="hi"), AIMessage(content="hello")]
        assert count_turns(msgs) == 1

    def test_multiple_turns(self):
        msgs = [
            HumanMessage(content="q1"),
            AIMessage(content="a1"),
            HumanMessage(content="q2"),
            AIMessage(content="a2"),
            HumanMessage(content="q3"),
            AIMessage(content="a3"),
        ]
        assert count_turns(msgs) == 3

    def test_with_tool_messages(self):
        msgs = [
            HumanMessage(content="q1"),
            AIMessage(content="let me search"),
            ToolMessage(content="result", tool_call_id="1"),
            AIMessage(content="a1"),
        ]
        # ToolMessage 不算轮数
        assert count_turns(msgs) == 1


# ============================================================
# 最后一条用户消息
# ============================================================

class TestGetLastHumanMessage:
    """获取最后一条用户消息测试"""

    def test_empty(self):
        assert get_last_human_message([]) is None

    def test_simple(self):
        msgs = [
            HumanMessage(content="第一个"),
            AIMessage(content="回答"),
            HumanMessage(content="第二个"),
        ]
        result = get_last_human_message(msgs)
        assert result is not None
        assert result.content == "第二个"

    def test_no_human(self):
        msgs = [AIMessage(content="回答"), ToolMessage(content="结果", tool_call_id="1")]
        assert get_last_human_message(msgs) is None


# ============================================================
# 对话格式化（给摘要用）
# ============================================================

class TestFormatConversation:
    """对话格式化测试"""

    def test_basic_conversation(self):
        msgs = [
            HumanMessage(content="你好"),
            AIMessage(content="你好！有什么可以帮你的？"),
        ]
        text = _format_conversation(msgs)
        assert "用户: 你好" in text
        assert "AI: 你好" in text

    def test_with_tool_message(self):
        msgs = [
            AIMessage(content="我来查一下", tool_calls=[{"id": "1", "name": "search", "args": {"q": "test"}}]),
            ToolMessage(content="搜索结果", tool_call_id="1", name="search"),
        ]
        text = _format_conversation(msgs)
        assert "调用工具" in text
        assert "[工具 search 返回]" in text
        assert "搜索结果" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
