"""流式回归测试：`/qa/stream` 的 `content` 事件必须真的能流出来

## 为什么需要这个文件（1.7.2 修的真问题）

原来 `course_agent/llm.py` 的 `SimpleLLM` 是**普通类**，只有 `invoke`/`bind_tools`，
外加一个拿非流式结果逐字切片的**假** `astream`。于是：

  · LangGraph 的 `stream_mode="messages"` 对它永远只推一条**完整 AIMessage**
    （不是 `AIMessageChunk`），被 `sse_content_event` 的"只放行 chunk"判据挡掉；
  · `content` 事件在生产里**从未触发过** → 学生等约 35 秒，整段答案一次性出现。

674 条测试全绿却漏掉了它，原因是流式用例全都用 `GenericFakeChatModel`
（**真** LangChain 模型，自己带 `_stream`）—— 测试里流的那个模型不是生产的那个。

这个文件的判据刻意**只对生产那个对象**成立：

  1. `llm` 必须是 `BaseChatModel` 子类（`messages` 模式认的就是这个）；
  2. 它必须在**假 HTTP 层**上产出 `AIMessageChunk`（证明流式来自 `_stream`，
     不是"把非流式结果切片"——那种假流式在真模型上仍是 35 秒）；
  3. 工具调用必须跟着流出去（`tool_call_chunks` 非空），否则 ReAct 静默退化；
  4. 不绑工具时请求体里**不能**出现 `tools`（planner/synthesizer 的行为不许被改）。
"""
import json
from typing import Annotated, TypedDict

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages

from course_agent import llm as llm_mod
from course_agent.llm import SimpleLLM
from course_agent.memory import INTERNAL_LLM_TAG
from course_agent.server import PLAN_ANSWER_NODES, sse_content_event

ANSWER = "死锁的四个必要条件是互斥、持有并等待、不可抢占、循环等待。"


class _FakeResponse:
    """够用的 requests 响应替身：`iter_lines()` 必须回 **bytes**（生产代码会 decode）"""

    def __init__(self, lines):
        self._lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8")


def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"delta": delta}]}, ensure_ascii=False)


def _fake_api(monkeypatch, lines, captured=None):
    """把 HTTP 层换掉，返回"按 SSE 逐行吐"的响应；captured 收下请求体"""
    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        if captured is not None:
            captured.append(json)
        return _FakeResponse(lines)

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)


def _content_lines(text: str) -> list:
    return [_sse({"content": ch}) for ch in text]


# ============================================================
# ① 类型门槛：messages 模式只对 BaseChatModel 的 _stream 生效
# ============================================================

class TestModelIsARealChatModel:
    def test_llm_is_a_langchain_chat_model(self):
        """不是 BaseChatModel 的话，LangGraph 永远不会吐 AIMessageChunk（就是那个 bug）"""
        assert isinstance(llm_mod.llm, BaseChatModel), (
            "生产 llm 又不是 LangChain ChatModel 了 —— /qa/stream 的 content 事件会再次消失")

    def test_the_global_singleton_has_no_tools_leaked_in(self):
        """bind_tools 不许就地改写单例（planner/synthesizer 会被塞进工具定义）"""
        bound = llm_mod.llm.bind_tools([{"type": "function", "function": {"name": "x"}}])
        assert llm_mod.llm._tools is None, "bind_tools 改写了全局单例：planner 也会带上工具定义"
        assert bound is not llm_mod.llm


# ============================================================
# ② 流式真的来自 _stream：逐片产出，且节点返回的消息不重复
# ============================================================

class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _collect(model, node_name="synthesizer", answer_nodes=PLAN_ANSWER_NODES):
    """把模型放进最小图，走真实的 messages 流，返回 (用户看到的正文, chunk 类型)"""
    def node(state):
        return {"messages": [model.invoke(state["messages"])]}

    g = StateGraph(_State)
    g.add_node(node_name, node)
    g.add_edge(START, node_name)
    g.add_edge(node_name, END)
    app = g.compile()

    seen, kinds = [], []
    for msg, meta in app.stream({"messages": [HumanMessage(content="讲讲死锁")]},
                                stream_mode="messages"):
        kinds.append((type(msg).__name__, meta.get("langgraph_node")))
        line = sse_content_event(msg, meta, answer_nodes)
        if line:
            seen.append(json.loads(line[6:])["content"])
    return "".join(seen), kinds


class TestStreamingActuallyReachesTheUser:
    def test_text_streams_character_by_character(self, monkeypatch):
        _fake_api(monkeypatch, _content_lines(ANSWER))
        text, kinds = _collect(SimpleLLM())
        assert text == ANSWER, f"逐字流没到用户手里：{text!r}"
        assert all(k == "AIMessageChunk" for k, _ in kinds), (
            f"流里出现了非 chunk 的消息（会重复或漏流）：{set(kinds)}")

    def test_the_answer_is_not_sent_twice(self, monkeypatch):
        """节点返回的完整 AIMessage 也会过一遍流 —— 必须是 chunk，否则答案出现两遍"""
        _fake_api(monkeypatch, _content_lines(ANSWER))
        text, _ = _collect(SimpleLLM())
        assert text.count(ANSWER) == 1, f"答案出现了 {text.count(ANSWER)} 遍"

    def test_internal_llm_calls_stay_out_of_the_answer(self, monkeypatch):
        """内部摘要调用（打了 INTERNAL_LLM_TAG）不许流进正文"""
        _fake_api(monkeypatch, _content_lines(ANSWER))
        summary_calls = []

        class _SummaryLLM(SimpleLLM):
            def _stream(self, messages, stop=None, run_manager=None, **kwargs):
                summary_calls.append(1)
                for ch in "【内部摘要】学生之前问过死锁":
                    yield llm_mod.ChatGenerationChunk(message=AIMessageChunk(content=ch))

        def node(state):
            # 模拟 compress_history：节点**内部**的一次调用
            list(_SummaryLLM().with_config(tags=[INTERNAL_LLM_TAG]).stream(state["messages"]))
            return {"messages": [SimpleLLM().invoke(state["messages"])]}

        g = StateGraph(_State)
        g.add_node("synthesizer", node)
        g.add_edge(START, "synthesizer")
        g.add_edge("synthesizer", END)
        app = g.compile()

        seen = []
        for msg, meta in app.stream({"messages": [HumanMessage(content="讲讲死锁")]},
                                    stream_mode="messages"):
            line = sse_content_event(msg, meta, PLAN_ANSWER_NODES)
            if line:
                seen.append(json.loads(line[6:])["content"])
        text = "".join(seen)
        assert summary_calls, "内部摘要调用根本没走到 _stream（这条测试白测了）"
        assert "内部摘要" not in text, f"内部调用流进了正文：{text!r}"
        assert text == ANSWER, f"正文被污染：{text!r}"

    def test_a_model_without_streaming_yields_no_content(self, monkeypatch):
        """反向验证：模型没有流式时，正文就该是空的（证明上面的绿不是碰巧）

        这条**就是 1.7.2 之前的生产状态**：`_stream` 不存在 → 流里只有一条
        完整 AIMessage → 被 sse_content_event 挡掉 → 0 个 content 事件
        （学生等 35 秒，答案一次性出现）。

        刻意用"把生产对象换成普通类"来复现，而不是给 `_stream` 打桩：
        **能不能流，判据就是"模型有没有 `_stream`"**，把判据本身换成替身，
        测的就不是当初那个 bug 了（langchain-core 1.6.2 里 `_stream` 一旦存在
        就会被直接调用，打桩成 raise 只是抛错，不是退回非流式）。
        """
        class _LegacyPlainModel:
            """1.7.2 之前那个 SimpleLLM：普通类，只有 invoke"""

            def invoke(self, messages, config=None):
                return AIMessage(content=ANSWER)

        text, kinds = _collect(_LegacyPlainModel())
        assert text == "", f"没有 chunk 却收到了正文（{text!r}）—— 这条反向验证失效了"
        assert kinds == [("AIMessage", "synthesizer")], (
            f"流里的东西与「旧模型」不符，这条测试没走到要测的那条路：{kinds}")


# ============================================================
# ③ 工具调用必须跟着流出去（ReAct 靠它往前走）
# ============================================================

class TestToolCallsSurviveTheStream:
    def test_tool_call_chunks_are_emitted(self, monkeypatch):
        lines = [
            _sse({"tool_calls": [{"index": 0, "id": "call_1",
                                  "function": {"name": "get_paper_questions", "arguments": '{"paper'}}]}),
            _sse({"tool_calls": [{"index": 0, "function": {"arguments": '_source": "甲"}'}}]}),
        ]
        _fake_api(monkeypatch, lines)
        chunks = list(SimpleLLM(tools=[{"type": "function", "function": {"name": "get_paper_questions"}}])
                      ._stream([HumanMessage(content="出题")]))
        assert chunks, "工具调用一片都没流出来 —— ReAct 会静默退化成「不调工具」"
        assert all(c.message.tool_call_chunks for c in chunks), chunks
        assert [c.message.tool_call_chunks[0]["name"] for c in chunks] == ["get_paper_questions", None]

    def test_langgraph_rebuilds_the_full_tool_call(self, monkeypatch):
        """**这条是 ReAct 的命门**：流式合并后必须拿到完整的 tool_calls

        智谱把工具调用拆片发：id/name 只在第一片，arguments 分几段。
        LangGraph 要能把它们合回 `{name, args, id}` —— 合不回来的话，
        ReAct 要么"调了工具但参数是空的"，要么整个工具调用消失（静默退化）。
        所以这里不看单片的形状，只看**图跑完之后 state 里的那条消息**。
        """
        lines = [
            _sse({"tool_calls": [{"index": 0, "id": "call_1",
                                  "function": {"name": "get_paper_questions", "arguments": '{"paper'}}]}),
            _sse({"tool_calls": [{"index": 0, "function": {"arguments": '_source": "甲"}'}}]}),
        ]
        _fake_api(monkeypatch, lines)
        tools = [{"type": "function", "function": {"name": "get_paper_questions",
                                                   "parameters": {"type": "object", "properties": {}}}}]
        model = SimpleLLM().bind_tools(tools)

        def node(state):
            return {"messages": [model.invoke(state["messages"])]}

        g = StateGraph(_State)
        g.add_node("llm", node)
        g.add_edge(START, "llm")
        g.add_edge("llm", END)
        app = g.compile()
        # 走 messages 流（= 生产里 /qa/stream 那条路），不是普通 invoke。
        # 不查 get_state（那要注入 checkpointer），而是按 LangGraph 自己用的
        # **同一个** 合并规则把 chunk 加起来 —— 图里 add_messages 就是这么合的。
        merged = None
        for msg, meta in app.stream({"messages": [HumanMessage(content="出第 1 组题")]},
                                    stream_mode="messages"):
            if meta.get("langgraph_node") == "llm" and isinstance(msg, AIMessageChunk):
                merged = msg if merged is None else merged + msg

        assert merged is not None, "一个 chunk 都没流出来 —— 走的是非流式那条路（测试没测到东西）"
        calls = merged.tool_calls
        assert len(calls) == 1, f"工具调用没合出来：{calls}"
        assert calls[0]["name"] == "get_paper_questions", calls
        assert calls[0]["args"] == {"paper_source": "甲"}, f"参数被拆坏了：{calls[0]['args']}"
        assert calls[0]["id"] == "call_1"
        assert not merged.invalid_tool_calls, f"有解析不了的工具调用：{merged.invalid_tool_calls}"

    def test_whitespace_only_tool_chunks_do_not_become_content(self, monkeypatch):
        """工具决策的那一轮 content 为空 —— 不能渗出空串/空白污染正文"""
        _fake_api(monkeypatch, [_sse({"tool_calls": [{"index": 0, "id": "c", "function": {
            "name": "search_knowledge", "arguments": "{}"}}]})])
        chunks = list(SimpleLLM(tools=[{"type": "function", "function": {"name": "search_knowledge"}}])
                      ._stream([HumanMessage(content="查")]))
        assert all(not (c.message.content or "").strip() for c in chunks), chunks


# ============================================================
# ④ 工具定义只在绑定时出现（planner/synthesizer 的请求体不许被改）
# ============================================================

class TestToolDefinitionsOnlyWhenBound:
    @pytest.mark.parametrize("bound", [False, True])
    def test_tools_key_presence_follows_binding(self, monkeypatch, bound):
        captured = []
        _fake_api(monkeypatch, _content_lines("好"), captured=captured)
        model = SimpleLLM()
        if bound:
            model = model.bind_tools([{"type": "function", "function": {"name": "search_knowledge"}}])
        list(model.stream([HumanMessage(content="查")]))
        assert captured, "请求根本没发出去"
        assert ("tools" in captured[0]) is bound, (
            f"bound={bound} 时请求体里的 tools 字段不对：{list(captured[0])}")

    def test_non_streaming_path_keeps_working(self, monkeypatch):
        """非流式那条路（invoke）仍然回完整 AIMessage —— nodes 大量依赖它"""
        def fake_post(url, headers=None, json=None, stream=False, timeout=None):
            class _R:
                def raise_for_status(self): return None

                def json(self):
                    return {"choices": [{"message": {"content": "完整答案"}}]}
            return _R()

        monkeypatch.setattr(llm_mod.requests, "post", fake_post)
        msg = SimpleLLM().invoke([HumanMessage(content="hi")])
        assert isinstance(msg, AIMessage) and msg.content == "完整答案"
