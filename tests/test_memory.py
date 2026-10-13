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
    summarize_old_messages,
    compress_history,
    count_turns,
    get_last_human_message,
    _format_conversation,
    SUMMARY_MARK,
    INTERNAL_LLM_TAG,
)


class _FakeLLM:
    """只实现 invoke 的假 LLM（摘要用）"""

    def __init__(self, content="用户问了死锁和进程调度。", raise_exc=None):
        self.content = content
        self.raise_exc = raise_exc
        self.calls = 0
        self.config_tags = []      # 记录 with_config 收到过哪些 tags

    def with_config(self, **kwargs):
        """真 llm 是 Runnable，带 with_config —— 摘要调用靠它打"内部调用"标签

        这个 tag 不是装饰：LangGraph 的 messages 模式会把图内每一次 LLM 调用都
        推给前端，包括本模块在节点内部跑的摘要。流层靠这个 tag 把它挡掉
        （见 server.sse_content_event 与 tests/test_stream_contract.py）。
        """
        self.config_tags.append(kwargs.get("tags"))
        return self

    def invoke(self, messages, config=None):
        self.calls += 1
        if self.raise_exc:
            raise self.raise_exc

        class R:
            pass

        r = R()
        r.content = self.content
        return r


class TestCompressHistory:
    """摘要压缩（原来整套是死代码，现在 synthesizer 真的在用）"""

    @pytest.fixture(autouse=True)
    def _clear_summary_cache(self):
        """摘要缓存是模块级的，必须逐用例清理，否则用例之间会互相污染"""
        from course_agent import memory as mem

        mem._SUMMARY_CACHE.clear()
        yield
        mem._SUMMARY_CACHE.clear()

    def _msgs(self, n_turns, tag="c"):
        """构造带 id 的消息（LangGraph 的 add_messages 在真实链路里会分配 id）"""
        out = [SystemMessage(content="系统提示")]
        out.append(HumanMessage(content=f"问题0-{tag}", id=f"{tag}-h0"))
        out.append(AIMessage(content=f"回答0-{tag}", id=f"{tag}-a0"))
        for i in range(1, n_turns):
            out.append(HumanMessage(content=f"问题{i}-{tag}", id=f"{tag}-h{i}"))
            out.append(AIMessage(content=f"回答{i}-{tag}", id=f"{tag}-a{i}"))
        return out

    def test_below_threshold_falls_back_to_window(self):
        """没超阈值也必须有窗口保护

        原实现是"没超阈值就原样返回"，等于阈值以下完全没有裁剪 ——
        25 条消息会全量塞给模型。
        """
        msgs = self._msgs(20)  # 41 条消息，超过 MAX_MESSAGES=20
        summary, recent = compress_history(msgs, _FakeLLM(), threshold=999)
        assert summary == ""
        assert len(recent) <= 20, f"未走滑动窗口，仍返回 {len(recent)} 条"

    def test_above_threshold_produces_summary(self):
        llm = _FakeLLM("用户问了死锁的四个必要条件。")
        msgs = self._msgs(25)  # 51 条消息
        summary, recent = compress_history(msgs, llm, threshold=10, keep_last_n=6)
        assert summary == "用户问了死锁的四个必要条件。"
        assert len(recent) == 6
        assert llm.calls == 1

    def test_summary_call_is_tagged_as_internal(self):
        """摘要那次 LLM 调用必须打上"内部调用"标签

        没有这个标签，它会被 LangGraph 的 messages 模式当成正常输出播给用户 ——
        实测确认：它带着**同一个** langgraph_node 流出去，所以按节点做白名单挡不住；
        给这次调用加 stream=False 也没用（只是从 token 流变成一条整消息，照样在流里）。
        流层靠这个 tag 挡它，见 server.sse_content_event。
        """
        llm = _FakeLLM("摘要")
        compress_history(self._msgs(25), llm, threshold=10, keep_last_n=6)
        assert llm.config_tags == [[INTERNAL_LLM_TAG]], (
            f"摘要调用没有打 tag，实际收到的 tags={llm.config_tags}")

    def test_no_llm_means_window_only(self):
        msgs = self._msgs(25)
        summary, recent = compress_history(msgs, None, threshold=10)
        assert summary == ""
        assert len(recent) <= 20

    def test_llm_failure_degrades_to_window(self):
        llm = _FakeLLM(raise_exc=RuntimeError("429"))
        msgs = self._msgs(25)
        summary, recent = compress_history(msgs, llm, threshold=10, keep_last_n=6)
        assert summary == "", "摘要失败应降级，不能让异常冒出去"
        assert recent, "降级后仍要返回可用的消息列表"

    def test_empty_summary_degrades(self):
        llm = _FakeLLM("   ")
        summary, recent = compress_history(self._msgs(25), llm, threshold=10)
        assert summary == ""
        assert recent

    def test_summarize_old_messages_keeps_system_and_marks_summary(self):
        llm = _FakeLLM("历史摘要内容")
        msgs = self._msgs(25)
        result = summarize_old_messages(msgs, llm, threshold=10, keep_last_n=6)
        assert isinstance(result[0], SystemMessage)
        assert "系统提示" in result[0].content
        summary_msgs = [m for m in result if isinstance(m, SystemMessage)
                        and SUMMARY_MARK in m.content]
        assert len(summary_msgs) == 1
        assert "历史摘要内容" in summary_msgs[0].content

    def test_summary_is_reused_within_amortization_window(self):
        """读时压缩会导致每轮都重算摘要，这里验证摊还缓存确实生效

        摘要请求要把几十条老消息全发一遍，比正常请求还贵，
        所以同一段对话每新增 _RESUMMARY_STEP 条才真正重算一次。
        """
        from course_agent import memory as mem

        llm = _FakeLLM("摘要内容")
        msgs = self._msgs(25)
        mem.compress_history(msgs, llm, threshold=10, keep_last_n=6)
        assert llm.calls == 1

        # 只多 2 条老消息（< _RESUMMARY_STEP）→ 复用
        msgs2 = self._msgs(25)
        msgs2.append(HumanMessage(content="追加", id="c-hx"))
        msgs2.append(AIMessage(content="追加回答", id="c-ax"))
        summary, _ = mem.compress_history(msgs2, llm, threshold=10, keep_last_n=6)
        assert llm.calls == 1, "窗口内不该重算摘要"
        assert summary == "摘要内容"

        # 多出足够多 → 重算
        msgs3 = self._msgs(25)
        for i in range(mem._RESUMMARY_STEP):
            msgs3.append(HumanMessage(content=f"继续{i}", id=f"c-hc{i}"))
            msgs3.append(AIMessage(content=f"继续答{i}", id=f"c-ac{i}"))
        mem.compress_history(msgs3, llm, threshold=10, keep_last_n=6)
        assert llm.calls == 2, "超过摊还窗口应重算"

    def test_messages_without_id_are_not_cached(self):
        """没有 id 时不做缓存

        曾用"首条消息内容的哈希"当缓存键，结果两段开头相同的对话会共用摘要
        （测试里立刻暴露）。宁可不缓存也不串摘要。
        """
        from course_agent import memory as mem

        no_id = [HumanMessage(content=f"问题{i}") for i in range(30)]
        llm = _FakeLLM("摘要")
        mem.compress_history(no_id, llm, threshold=5, keep_last_n=4)
        assert mem._SUMMARY_CACHE == {}, "无 id 不应写入缓存"
        mem.compress_history(no_id, llm, threshold=5, keep_last_n=4)
        assert llm.calls == 2, "无 id 时每次都重算（安全优先）"

    def test_summary_cache_keys_by_conversation(self):
        """不同会话不能互相串用摘要"""
        from course_agent import memory as mem

        llm_a = _FakeLLM("会话A摘要")
        mem.compress_history(self._msgs(25, tag="A"), llm_a, threshold=5, keep_last_n=4)

        llm_b = _FakeLLM("会话B摘要")
        summary_b, _ = mem.compress_history(self._msgs(25, tag="B"), llm_b,
                                            threshold=5, keep_last_n=4)
        assert summary_b == "会话B摘要"
        assert llm_b.calls == 1

    def test_summary_cache_is_bounded(self):
        from course_agent import memory as mem

        for i in range(mem._MAX_SUMMARY_CACHE + 20):
            mem._cache_summary(f"k{i}", i, "s")
        assert len(mem._SUMMARY_CACHE) <= mem._MAX_SUMMARY_CACHE

    def test_synthesizer_node_uses_the_summary(self, monkeypatch):
        """synthesizer 必须把摘要拼进 system prompt（否则等于没接）"""
        from course_agent import nodes

        captured = {}

        def fake_invoke(llm_, messages, context=""):
            captured["messages"] = messages

            class R:
                content = "回答"

            return R()

        monkeypatch.setattr(nodes, "safe_llm_invoke", fake_invoke)
        monkeypatch.setattr(nodes, "compress_history",
                            lambda msgs, llm=None: ("这是历史摘要", [HumanMessage(content="最近")]))

        nodes.synthesizer_node({
            "messages": [HumanMessage(content="你好")],
            "plan": [],
        })
        system_prompt = captured["messages"][0].content
        assert "这是历史摘要" in system_prompt
        assert SUMMARY_MARK in system_prompt

    def test_planner_node_also_sees_the_summary(self, monkeypatch):
        """planner 也要拿到摘要

        否则"我最开始问的是什么"这种指代型追问，规划阶段就丢了上下文，
        只会拿上一轮的话题去检索，答案自然跑偏。
        """
        from course_agent import nodes

        captured = {}

        def fake_invoke(llm_, messages, context=""):
            captured["messages"] = messages
            captured["context"] = context

            class R:
                content = '{"steps": []}'

            return R()

        monkeypatch.setattr(nodes, "safe_llm_invoke", fake_invoke)
        monkeypatch.setattr(nodes, "compress_history",
                            lambda msgs, llm=None: ("更早聊过进程与线程", [HumanMessage(content="最近")]))

        nodes.planner_node({
            "messages": [HumanMessage(content="我最开始问的是什么")],
            "plan": [],
        })
        system_prompt = captured["messages"][0].content
        assert "更早聊过进程与线程" in system_prompt

    def test_planner_and_synthesizer_use_the_same_history_scope(self, monkeypatch):
        """两个节点的历史口径要一致，摘要缓存才能命中（否则一轮调两次摘要 LLM）

        这条原来断言的是"函数源码里出现 `not isinstance(m, ToolMessage)` 这段字符串"。
        实测**能被注释骗过**：把真实的过滤删掉、只在注释里留下那句话，
        全量 594 条测试依然全绿。所以改成**行为断言** —— 直接看
        compress_history 实际收到的是哪些消息。
        """
        from course_agent import nodes

        scopes = []

        def fake_compress(msgs, llm):
            scopes.append(list(msgs))
            return "", list(msgs)

        monkeypatch.setattr(nodes, "compress_history", fake_compress)
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, messages, context=None: nodes.AIMessage(content=""))

        state = {
            "messages": [
                nodes.SystemMessage(content="系统提示"),
                nodes.HumanMessage(content="第一个问题"),
                nodes.AIMessage(content="回答"),
                nodes.ToolMessage(content="工具结果", tool_call_id="c1",
                                  name="search_knowledge"),
                nodes.HumanMessage(content="第二个问题"),
            ],
            "plan": [], "tool_rounds": 0,
        }
        nodes.planner_node(state, {"configurable": {"thread_id": "t-scope"}})
        nodes.synthesizer_node(state)

        assert len(scopes) == 2, f"两个节点都该调 compress_history，实际 {len(scopes)} 次"
        for i, msgs in enumerate(scopes, 1):
            kinds = [type(m).__name__ for m in msgs]
            assert "SystemMessage" not in kinds, f"第 {i} 次调用没排除系统提示: {kinds}"
            assert "ToolMessage" not in kinds, f"第 {i} 次调用没排除工具结果: {kinds}"
        assert ([type(m).__name__ for m in scopes[0]]
                == [type(m).__name__ for m in scopes[1]]), (
            "planner 与 synthesizer 的历史口径不一致 —— 摘要缓存会 miss，"
            "每轮要多跑一次摘要 LLM")


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
