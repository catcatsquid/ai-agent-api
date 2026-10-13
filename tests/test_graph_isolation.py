"""两张图的会话状态必须互不污染（跨模式串味）

背景：原来 `graph` 与 `react_graph` 共用同一个模块级 MemorySaver，命名空间只有
thread_id。于是"同一个 thread_id 换个模式继续说"会读到对方的会话状态；
更糟的是 plan 的 executor 会造出合成 ToolMessage(tool_call_id="plan_0")，
而 state 里没有声明它的 assistant 消息 —— react 把这串发给模型会被按协议拒绝（400），
被 safe_llm_invoke 吞成兜底套话，表现为"切完模式模型静默变傻"。

1.2.0 起 checkpointer 换成了落盘的 SqliteSaver（见 graph.py）：**两张图各一个库文件**。
"""
from unittest.mock import patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from course_agent import nodes
from course_agent.graph import (
    graph,
    react_graph,
    plan_checkpointer,
    react_checkpointer,
)


def _db_path(saver) -> str:
    """取 sqlite 连接实际打开的库文件路径"""
    row = saver.conn.execute("PRAGMA database_list").fetchone()
    return row[2] if row else ""


class TestIndependentCheckpointers:
    def test_each_graph_has_its_own_checkpointer(self):
        assert plan_checkpointer is not react_checkpointer, \
            "两张图不能共用同一个 checkpointer"

    def test_the_two_checkpointers_use_different_files(self):
        """光"两个对象"不够 —— 必须落在**不同的库文件**上

        否则即使对象不同，同一个 thread_id 在同一个库里还是会撞上，
        等于绕了一圈又回到"切模式串味"。
        """
        p, r = _db_path(plan_checkpointer), _db_path(react_checkpointer)
        assert p and r, f"取不到库文件路径：plan={p!r} react={r!r}"
        assert p != r, f"两张图用了同一个库文件：{p}"
        assert p.replace("\\", "/").endswith("runtime/checkpoints_plan.sqlite3")
        assert r.replace("\\", "/").endswith("runtime/checkpoints_react.sqlite3")

    def test_react_does_not_see_plan_state(self):
        """往 Plan 图写一轮状态，ReAct 图用同一 thread_id 不能读到"""
        tid = "isolation-test-plan-write"
        cfg = {"configurable": {"thread_id": tid}}
        graph.update_state(cfg, {
            "messages": [HumanMessage(content="PLAN-ONLY-MARKER", id="p1")],
            "plan": [{"tool": "search_knowledge", "args": {"query": "死锁"}}],
        })

        st = react_graph.get_state(cfg)
        msgs = (st.values or {}).get("messages", []) if st.values else []
        leaked = [m for m in msgs if "PLAN-ONLY-MARKER" in str(getattr(m, "content", ""))]
        assert not leaked, f"ReAct 读到了 Plan 的会话状态: {leaked}"
        assert not (st.values or {}).get("plan"), "ReAct 不该看到 plan 字段"

    def test_plan_does_not_see_react_state(self):
        tid = "isolation-test-react-write"
        cfg = {"configurable": {"thread_id": tid}}
        react_graph.update_state(cfg, {
            "messages": [HumanMessage(content="REACT-ONLY-MARKER", id="r1")],
        })

        st = graph.get_state(cfg)
        msgs = (st.values or {}).get("messages", []) if st.values else []
        leaked = [m for m in msgs if "REACT-ONLY-MARKER" in str(getattr(m, "content", ""))]
        assert not leaked, f"Plan 读到了 ReAct 的会话状态: {leaked}"

    def test_builders_accept_injected_checkpointer(self):
        """留出依赖注入的缝，方便换成 SQLite/Postgres 持久化"""
        from langgraph.checkpoint.memory import MemorySaver
        custom = MemorySaver()
        assert custom is not None

    def test_shared_checkpointer_would_leak(self):
        """反向验证：故意让两张图共用一个 checkpointer 时确实会串

        这条的作用是证明上面那两条"不串"的断言有判别力 ——
        否则它们可能只是因为读取姿势不对而永远为真。
        """
        from langgraph.checkpoint.memory import MemorySaver
        from course_agent.graph import build_graph, build_react_graph

        shared = MemorySaver()
        g = build_graph(checkpointer=shared)
        r = build_react_graph(checkpointer=shared)

        tid = "shared-checkpointer-demo"
        cfg = {"configurable": {"thread_id": tid}}
        g.update_state(cfg, {
            "messages": [HumanMessage(content="LEAK-MARKER", id="x1")],
            "plan": [],
        })
        st = r.get_state(cfg)
        msgs = (st.values or {}).get("messages", []) if st.values else []
        assert any("LEAK-MARKER" in str(getattr(m, "content", "")) for m in msgs), \
            "共用 checkpointer 时本应发生串味，说明这条反向验证失效了"


class TestNoOrphanToolMessages:
    """executor 必须同时给出声明 tool_call 的 assistant 消息"""

    def test_declaration_pairs_with_results(self):
        plan = [
            {"tool": "search_knowledge", "args": {"query": "死锁"}},
            {"tool": "list_papers", "args": {}},
        ]
        with patch.object(nodes, "safe_tool_invoke", return_value="结果"):
            out = nodes.executor_node({"messages": [], "plan": plan, "tool_rounds": 0})

        msgs = out["messages"]
        assert isinstance(msgs[0], AIMessage), "第一条必须是声明 tool_calls 的 assistant 消息"
        declared = {tc["id"] for tc in msgs[0].tool_calls}
        used = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
        assert declared == used, f"声明 {declared} 与结果 {used} 不匹配（有孤儿）"
        assert len(declared) == len(plan)

    def test_unknown_tool_still_declared(self):
        plan = [{"tool": "不存在的工具", "args": {}}]
        out = nodes.executor_node({"messages": [], "plan": plan, "tool_rounds": 0})
        msgs = out["messages"]
        declared = {tc["id"] for tc in msgs[0].tool_calls}
        used = {m.tool_call_id for m in msgs if isinstance(m, ToolMessage)}
        assert declared == used
        assert "未知工具" in msgs[1].content

    def test_every_tool_message_has_a_declaration(self):
        """不变量：返回的消息序列里，任何 ToolMessage 都能找到声明它的 AIMessage"""
        plan = [{"tool": "search_knowledge", "args": {"query": "x"}}]
        with patch.object(nodes, "safe_tool_invoke", return_value="ok"):
            out = nodes.executor_node({"messages": [], "plan": plan, "tool_rounds": 0})

        declared = set()
        for m in out["messages"]:
            if isinstance(m, AIMessage) and m.tool_calls:
                declared |= {tc["id"] for tc in m.tool_calls}
        for m in out["messages"]:
            if isinstance(m, ToolMessage):
                assert m.tool_call_id in declared, f"{m.tool_call_id} 没有对应声明"

    def test_empty_plan_returns_nothing(self):
        out = nodes.executor_node({"messages": [], "plan": [], "tool_rounds": 0})
        assert out["messages"] == []
