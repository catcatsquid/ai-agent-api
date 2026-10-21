"""出题轮**不逐字流式**（2026-10-18 用户实测后拍板）

现象（用户截图）：从题库出题时，气泡里先刷出一大串
`![题目](/qb_images/2024%E5%B9%B4%E7%A7%8B...png)` 的半截 URL 与坏图，
一条 5 题的答案光 URL 就要输出几百 token，看着像卡住。

改法：这一轮只要**真的调了取题工具**（`QUIZ_TOOLS`），正文就不逐字推，
等 `done.answer` 一次给全；`node` 进度事件照旧。

这里守两件事：
  1. 判据函数 `_quiz_turn_update` 认得出两种图的工具轨迹；
  2. **端到端**：`/qa/stream` 的 SSE 里，出题轮没有 content 事件、done 带完整正文；
     而知识问答轮照常流式（不能把所有人都一起"静音"）。
"""
import json

import pytest
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from course_agent import server
from course_agent.quiz_service import QUIZ_TOOLS


class TestQuizTurnDetection:
    @pytest.mark.parametrize("tool", sorted(QUIZ_TOOLS))
    def test_plan_graph_declaration_is_recognised(self, tool):
        """plan 图：executor 返回一条声明了 tool_calls 的 AIMessage"""
        out = {"messages": [AIMessage(content="", tool_calls=[
            {"id": "c1", "name": tool, "args": {}}])]}
        assert server._quiz_turn_update(out) is True

    @pytest.mark.parametrize("tool", sorted(QUIZ_TOOLS))
    def test_react_graph_tool_message_is_recognised(self, tool):
        """react 图：tool_node 返回的 ToolMessage 带 name"""
        out = {"messages": [ToolMessage(content="{}", tool_call_id="c1", name=tool)]}
        assert server._quiz_turn_update(out) is True

    @pytest.mark.parametrize("tool", ["search_knowledge", "read_section",
                                     "explain_question", "prepare_custom_quiz",
                                     "grade_answers"])
    def test_other_tools_keep_streaming(self, tool):
        """讲题 / 讲解 / 自拟出题 / 批改都照常流式 —— 别一锅端"""
        out = {"messages": [AIMessage(content="", tool_calls=[
            {"id": "c1", "name": tool, "args": {}}])]}
        assert server._quiz_turn_update(out) is False

    def test_empty_or_odd_updates_are_safe(self):
        assert server._quiz_turn_update({}) is False
        assert server._quiz_turn_update({"messages": []}) is False
        assert server._quiz_turn_update({"messages": [AIMessage(content="纯文字")]}) is False


class _FakeState:
    def __init__(self, messages):
        self.values = {"messages": messages}


class _FakeGraph:
    """只实现 qa_stream 用到的那两个方法：stream / get_state"""

    def __init__(self, events, final_answer):
        self._events = events
        self._answer = final_answer

    def stream(self, *a, **k):
        yield from self._events

    def get_state(self, config):
        return _FakeState([AIMessage(content=self._answer)])


def _run_stream(monkeypatch, events, final_answer):
    """走真实的 /qa/stream 端点（TestClient + SSE），把事件解出来"""
    from fastapi.testclient import TestClient

    monkeypatch.setattr(server, "graph", _FakeGraph(events, final_answer))
    body = {"query": "出几道题", "mode": "plan", "thread_id": "probe-stream-test"}
    with TestClient(server.app).stream("POST", "/qa/stream", json=body) as resp:
        assert resp.status_code == 200, resp.status_code
        raw = "".join(resp.iter_text())
    return [json.loads(line[6:]) for line in raw.splitlines() if line.startswith("data: ")]


def _quiz_events():
    plan = AIMessage(content="", tool_calls=[
        {"id": "p0", "name": "get_paper_questions", "args": {}}])
    step = AIMessage(content="（调用了工具：get_paper_questions）", tool_calls=plan.tool_calls)
    tool = ToolMessage(content='{"paper": "某卷"}', tool_call_id="p0",
                       name="get_paper_questions")
    return [
        ("updates", {"planner": {"plan": [{"tool": "get_paper_questions", "args": {}}]}}),
        ("updates", {"executor": {"messages": [step, tool]}}),
        # 用 AIMessageChunk（真正的逐字产出）—— 否则"没有 content"是因为类型不对，
        # 不是因为被抑制了，这个用例就成了假绿
        ("messages", (AIMessageChunk(content="第 1 组："),
                      {"langgraph_node": "synthesizer"})),
        ("messages", (AIMessageChunk(content="![题目](/qb_images/x.png)"),
                      {"langgraph_node": "synthesizer"})),
    ]


def _qa_events():
    tool = ToolMessage(content="死锁是……", tool_call_id="p0", name="search_knowledge")
    return [
        ("updates", {"executor": {"messages": [tool]}}),
        ("messages", (AIMessageChunk(content="死锁指"),
                      {"langgraph_node": "synthesizer"})),
        ("messages", (AIMessageChunk(content="多个进程互相等待。"),
                      {"langgraph_node": "synthesizer"})),
    ]


class TestStreamSuppression:
    def test_quiz_turn_has_no_content_events(self, monkeypatch):
        events = _run_stream(monkeypatch, _quiz_events(),
                             "第 1 组：\n![题目](/qb_images/x.png)\n请作答。")
        kinds = [e["type"] for e in events]
        assert "content" not in kinds, f"出题轮不该推正文：{kinds}"
        done = next(e for e in events if e["type"] == "done")
        assert "![题目](/qb_images/x.png)" in done["answer"], "done 必须带完整正文"
        assert kinds[0] == "start" and kinds[-1] == "done"

    def test_quiz_turn_still_reports_progress(self, monkeypatch):
        """不推正文 ≠ 一片空白：取题的进度事件必须照旧发"""
        events = _run_stream(monkeypatch, _quiz_events(), "（略）")
        assert any(e["type"] == "node" for e in events), events

    def test_knowledge_turn_still_streams(self, monkeypatch):
        events = _run_stream(monkeypatch, _qa_events(), "死锁指多个进程互相等待。")
        content = "".join(e["content"] for e in events if e["type"] == "content")
        assert content == "死锁指多个进程互相等待。", content
