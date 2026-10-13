"""服务端会话列表（GET /chat/conversations）

为什么这个接口以前不可能有：会话状态原来在 MemorySaver 里（**进程内存**），
服务一重启就没了，也就没什么可列的。1.2.0 换成落盘的 SqliteSaver 之后，
服务端才第一次真的"存着"这些会话。

这里同时守住 checkpoint_id → 时间的换算：算错了列表的排序和"更新时间"就没意义。
"""
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite import SqliteSaver

from course_agent import server
from course_agent.graph import build_graph, build_react_graph
from course_agent.memory import SUMMARY_MARK
from course_agent.server import _checkpoint_time, app


@pytest.fixture
def tmp_checkpointers(tmp_path, monkeypatch):
    """把 server 看到的 checkpointer/graph 换成临时库里的

    必须换 —— 否则测试会往真实的 runtime/checkpoints_*.sqlite3 里写数据，
    把用户自己的会话列表搅乱。
    """
    def _saver(name):
        return SqliteSaver(sqlite3.connect(str(tmp_path / name), check_same_thread=False))

    plan, react = _saver("plan.sqlite3"), _saver("react.sqlite3")
    monkeypatch.setattr(server, "plan_checkpointer", plan)
    monkeypatch.setattr(server, "react_checkpointer", react)
    monkeypatch.setattr(server, "graph", build_graph(checkpointer=plan))
    monkeypatch.setattr(server, "react_graph", build_react_graph(checkpointer=react))
    return plan, react


class TestCheckpointIdDecoding:
    """checkpoint_id 是 UUIDv6：前 48 位是 60 位时间戳的高 48 位（1582 起的 100ns）"""

    def test_real_checkpoint_ids_decode_to_the_write_time(self, tmp_path):
        """拿真实 checkpointer 写出来的 id 验，不用手造的样本"""
        db = tmp_path / "c.sqlite3"
        g = build_graph(checkpointer=SqliteSaver(
            sqlite3.connect(str(db), check_same_thread=False)))
        before = datetime.now(timezone.utc)
        g.update_state({"configurable": {"thread_id": "t"}},
                       {"messages": [HumanMessage(content="你好", id="m1")]})
        after = datetime.now(timezone.utc)

        con = sqlite3.connect(str(db))
        cid = con.execute("SELECT checkpoint_id FROM checkpoints").fetchone()[0]
        con.close()

        got = datetime.fromisoformat(_checkpoint_time(cid))
        assert before - timedelta(seconds=5) <= got <= after + timedelta(seconds=5), (
            f"checkpoint_id 解出来的时间不对：解出 {got}，实际写在 [{before}, {after}] 之间")

    @pytest.mark.parametrize("bad", ["", "not-a-uuid", "zzzz", None])
    def test_garbage_returns_none_instead_of_raising(self, bad):
        assert _checkpoint_time(bad) is None


class TestConversationList:
    def test_empty_store_gives_an_empty_list(self, tmp_checkpointers):
        assert server.list_server_conversations(10) == []

    def test_a_written_thread_shows_up_with_its_title(self, tmp_checkpointers):
        server.graph.update_state(
            {"configurable": {"thread_id": "t1"}},
            {"messages": [HumanMessage(content="什么是死锁？", id="c1")]})
        rows = server.list_server_conversations(10)
        assert len(rows) == 1, f"应该是 1 个会话，实际 {rows}"
        row = rows[0]
        assert row["thread_id"] == "t1"
        assert row["mode"] == "plan"
        assert row["title"].startswith("什么是死锁")
        assert row["checkpoints"] == 1
        assert row["updated_at"], "没给出更新时间"

    def test_plan_and_react_are_told_apart(self, tmp_checkpointers):
        """两个库都要列出来，且 mode 不能混 —— 前端要据此选对图"""
        server.graph.update_state({"configurable": {"thread_id": "p1"}},
                                  {"messages": [HumanMessage(content="plan 侧", id="a")]})
        server.react_graph.update_state({"configurable": {"thread_id": "r1"}},
                                        {"messages": [HumanMessage(content="react 侧", id="b")]})
        got = {(r["thread_id"], r["mode"]) for r in server.list_server_conversations(10)}
        assert got == {("p1", "plan"), ("r1", "react")}, f"实际 {got}"

    def test_most_recent_comes_first(self, tmp_checkpointers):
        for tid, label in (("old", "最早的"), ("new", "最新的")):
            server.graph.update_state(
                {"configurable": {"thread_id": tid}},
                {"messages": [HumanMessage(content=label, id=tid)]})
        rows = server.list_server_conversations(10)
        assert [r["thread_id"] for r in rows][0] == "new", \
            f"最新的会话没排在最前：{[r['thread_id'] for r in rows]}"

    def test_limit_is_respected(self, tmp_checkpointers):
        for i in range(5):
            server.graph.update_state(
                {"configurable": {"thread_id": f"t{i}"}},
                {"messages": [HumanMessage(content=f"第{i}问", id=f"m{i}")]})
        assert len(server.list_server_conversations(2)) == 2


class TestConversationsEndpoint:
    def test_returns_the_house_shape(self, tmp_checkpointers):
        resp = TestClient(app).get("/chat/conversations?limit=5")
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 0
        assert isinstance(body["data"], list)

    def test_rejects_a_silly_limit(self, tmp_checkpointers):
        """limit 由 Query(ge=1, le=200) 把关，别让 -1 或 10**9 打到库里"""
        client = TestClient(app)
        assert client.get("/chat/conversations?limit=0").status_code == 422
        assert client.get("/chat/conversations?limit=99999").status_code == 422


class TestConversationDetail:
    """取回会话内容（GET /chat/conversations/{thread_id}）

    重点是**只返回能展示的部分**：state 里混着工具结果（检索原文 / 参考答案 /
    批改 JSON）与内部协议声明，它们绝不该通过这个接口回到前端。
    """

    def _write(self, messages, tid="t1", mode="plan"):
        target = server.react_graph if mode == "react" else server.graph
        target.update_state({"configurable": {"thread_id": tid}},
                            {"messages": messages})

    def test_returns_user_and_assistant_messages_in_order(self, tmp_checkpointers):
        self._write([HumanMessage(content="什么是死锁？", id="u1"),
                     AIMessage(content="死锁是……", id="a1")])
        resp = TestClient(app).get("/chat/conversations/t1")
        assert resp.status_code == 200
        assert resp.json()["data"]["messages"] == [
            {"role": "user", "content": "什么是死锁？"},
            {"role": "assistant", "content": "死锁是……"},
        ]

    def test_tool_results_never_come_back(self, tmp_checkpointers):
        """工具结果里是检索原文 / 参考答案 / 批改 JSON —— 这个接口绝不能漏"""
        secret = '{"answer": "D", "reference_points": ["因为等待环"]}'
        self._write([
            HumanMessage(content="批改一下", id="u1"),
            AIMessage(content="", id="a1",
                      tool_calls=[{"id": "c1", "name": "grade_answers", "args": {}}]),
            ToolMessage(content=secret, tool_call_id="c1", name="grade_answers"),
            AIMessage(content="本组 3 题：Q1、Q2、Q3", id="a2"),
        ])
        body = TestClient(app).get("/chat/conversations/t1").json()
        dump = json.dumps(body, ensure_ascii=False)
        assert "reference_points" not in dump, f"工具结果漏出去了：{dump}"
        assert secret not in dump
        # 带 tool_calls 的那条 AIMessage 也不该露出来
        assert [m["role"] for m in body["data"]["messages"]] == ["user", "assistant"]

    def test_internal_summary_is_not_shown(self, tmp_checkpointers):
        """压缩摘要（SystemMessage + SUMMARY_MARK）是内部产物，不是谁的原话"""
        self._write([SystemMessage(content=f"{SUMMARY_MARK}\n学生之前问过死锁", id="s1"),
                     HumanMessage(content="继续", id="u1"),
                     AIMessage(content="好的", id="a1")])
        msgs = TestClient(app).get("/chat/conversations/t1").json()["data"]["messages"]
        assert SUMMARY_MARK not in json.dumps(msgs, ensure_ascii=False)
        assert [m["role"] for m in msgs] == ["user", "assistant"]

    def test_unknown_thread_is_404(self, tmp_checkpointers):
        assert TestClient(app).get("/chat/conversations/does-not-exist").status_code == 404

    def test_wrong_mode_is_404_not_a_silent_empty_chat(self, tmp_checkpointers):
        """thread 在 plan 库里，用 react 读必须 404

        两条图各有各的 checkpointer —— 读错图时**不能**静默返回"空历史"，
        那会让前端以为这条会话是空的（1.0.2 修的就是这类静默失忆）。
        """
        self._write([HumanMessage(content="plan 侧", id="u1")], tid="only-plan")
        client = TestClient(app)
        assert client.get("/chat/conversations/only-plan?mode=plan").status_code == 200
        assert client.get("/chat/conversations/only-plan?mode=react").status_code == 404

    def test_bad_mode_is_rejected(self, tmp_checkpointers):
        assert TestClient(app).get(
            "/chat/conversations/t1?mode=whatever").status_code == 422

    def test_react_thread_reads_from_the_react_graph(self, tmp_checkpointers):
        self._write([HumanMessage(content="react 侧", id="u1"),
                     AIMessage(content="答", id="a1")], tid="r1", mode="react")
        body = TestClient(app).get("/chat/conversations/r1?mode=react").json()
        assert body["data"]["mode"] == "react"
        assert len(body["data"]["messages"]) == 2
