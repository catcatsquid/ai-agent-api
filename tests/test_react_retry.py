"""ReAct「嘴上说要调、实际没调」的补一轮（1.11.1）

背景：ReAct 每轮让模型自己决定调不调工具，实测约 1/3 概率出现
「文字里写'我先查一下'，但 tool_calls 是空的」——于是这一轮没检索就作答。
用户选的方案：**补一轮强制调用**（`tool_choice="required"`），不是放宽判据。
"""
import json

from langchain_core.messages import AIMessage, HumanMessage

from course_agent import nodes


class TestNarrateDetection:
    def test_it_recognises_the_narration(self):
        for t in ("我先查一下知识库再回答", "让我检索一下这门课的资料",
                  "我来调用一下知识库获取更详细的信息"):
            assert nodes._narrates_tool_use(t), t

    def test_plain_answers_are_not_narration(self):
        for t in ("死锁的四个必要条件是互斥、持有并等待…", ""):
            assert not nodes._narrates_tool_use(t), t


class TestRetryWithForcedToolCall:
    def _llm_node(self, monkeypatch, replies):
        calls = []
        seq = list(replies)

        def fake(fn, msgs, context=None, *a, **k):
            calls.append((context, getattr(fn, "_tool_choice", "auto")))
            return seq.pop(0)

        monkeypatch.setattr(nodes, "safe_llm_invoke", fake)
        out = nodes.llm_node({"messages": [HumanMessage(content="讲讲死锁")],
                              "tool_rounds": 0, "session": {}})
        return out["messages"][-1], calls

    def test_it_retries_once_when_narrated_but_not_called(self, monkeypatch):
        first = AIMessage(content="我先查一下知识库再回答。")
        forced = AIMessage(content="", tool_calls=[
            {"id": "c1", "name": "search_knowledge", "args": {"query": "死锁"}}])
        answer, calls = self._llm_node(monkeypatch, [first, forced])
        assert answer.tool_calls, "补一轮没生效：这一轮还是没发起工具调用"
        assert len(calls) == 2, calls
        assert calls[1][0] == "react_llm_forced", calls
        assert calls[1][1] == "required", f"补的那一轮没有强制调用工具：{calls}"

    def test_it_does_not_retry_when_the_model_already_called_a_tool(self, monkeypatch):
        called = AIMessage(content="", tool_calls=[
            {"id": "c1", "name": "search_knowledge", "args": {"query": "死锁"}}])
        answer, calls = self._llm_node(monkeypatch, [called])
        assert len(calls) == 1, calls

    def test_it_does_not_retry_for_a_plain_answer(self, monkeypatch):
        plain = AIMessage(content="死锁的四个必要条件如下：互斥、持有并等待、不可抢占、循环等待。")
        answer, calls = self._llm_node(monkeypatch, [plain])
        assert len(calls) == 1, calls
        assert not answer.tool_calls

    def test_it_keeps_the_original_answer_if_the_forced_round_still_fails(self, monkeypatch):
        """补的那一轮也没发起调用 → 保留原来那段文字（别把回答吞掉）"""
        first = AIMessage(content="我先查一下知识库再回答。")
        still = AIMessage(content="抱歉，我还是没能取到资料。")
        answer, calls = self._llm_node(monkeypatch, [first, still])
        assert len(calls) == 2
        assert "抱歉" in answer.content or "查一下" in answer.content
