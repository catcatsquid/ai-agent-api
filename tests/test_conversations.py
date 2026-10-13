"""服务端会话列表（GET /chat/conversations）

为什么这个接口以前不可能有：会话状态原来在 MemorySaver 里（**进程内存**），
服务一重启就没了，也就没什么可列的。1.2.0 换成落盘的 SqliteSaver 之后，
服务端才第一次真的"存着"这些会话。

这里同时守住 checkpoint_id → 时间的换算：算错了列表的排序和"更新时间"就没意义。
"""
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from course_agent import server
from course_agent.graph import build_graph, build_react_graph
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
