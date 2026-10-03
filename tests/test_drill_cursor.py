"""Plan 图的跨轮会话游标（route A 的核心修复）

背景：Plan 图的 planner_node 组装历史时会**排除 ToolMessage**、并把 assistant 内容
截到 200 字，所以上一轮工具结果里的 `next_group_index` 传不到这一轮。
实测后果：学生说"继续下一组"时 planner 既没卷子也没组号 ——
**时好时坏**（有时能从上一轮发出去的文字里推出"第 1/19 组"，有时推不出），
而间歇性错误比稳定错误更难信任。

修法：给 AgentState 加 `session` 字段，由 executor 从工具结果里抽出游标存进去，
再由 planner_node 组进 prompt。
"""
import json

import pytest

from course_agent import nodes


def _fake_group_result(paper="测试卷", group_index=1, next_index=2, qids=("q1", "q2")):
    return json.dumps({
        "paper": paper, "type": "选择题", "progress": f"第 {group_index}/5 组",
        "group_index": group_index, "group_count": 5, "has_more": True,
        "next_group_index": next_index, "questions": [], "qids": list(qids),
        "reply_text": "…",
    }, ensure_ascii=False)


class TestDrillCursorExtraction:
    def test_executor_records_cursor_from_group_result(self):
        plan = [{"tool": "get_paper_questions",
                 "args": {"paper_source": "测试卷", "group_index": 1}}]
        results = [nodes.ToolMessage(content=_fake_group_result(),
                                     tool_call_id="plan_0", name="get_paper_questions")]
        session = nodes._extract_drill_session({}, plan, results)
        assert session["paper"] == "测试卷"
        assert session["group_index"] == 1
        assert session["next_group_index"] == 2
        assert session["qids"] == ["q1", "q2"]
        assert session["type"] == "选择题"

    def test_cursor_survives_into_next_round(self):
        """第二轮的 executor 应该在已有游标基础上更新，而不是从零开始"""
        first = {"paper": "测试卷", "group_index": 1, "next_group_index": 2, "qids": ["q1"]}
        plan = [{"tool": "get_paper_questions", "args": {"paper_source": "测试卷",
                                                         "group_index": 2}}]
        results = [nodes.ToolMessage(content=_fake_group_result(group_index=2, next_index=3,
                                                                qids=("q6",)),
                                     tool_call_id="plan_0", name="get_paper_questions")]
        session = nodes._extract_drill_session(first, plan, results)
        assert session["group_index"] == 2
        assert session["next_group_index"] == 3
        assert session["qids"] == ["q6"]

    def test_no_cursor_change_for_other_tools(self):
        plan = [{"tool": "list_papers", "args": {}}]
        results = [nodes.ToolMessage(content="1. 某卷子", tool_call_id="plan_0",
                                     name="list_papers")]
        assert nodes._extract_drill_session({}, plan, results) is None

    def test_garbage_result_does_not_break(self):
        """工具返回错误文案/非 JSON 时不能抛异常，也不能污染游标"""
        plan = [{"tool": "get_paper_questions", "args": {}}]
        for bad in ["没能确定是哪份卷子，请从这些里选一个：\nA\nB",
                    '{"error": "题型不存在"}',
                    "{}", ""]:
            results = [nodes.ToolMessage(content=bad, tool_call_id="plan_0",
                                         name="get_paper_questions")]
            assert nodes._extract_drill_session({}, plan, results) is None, bad

    def test_finished_paper_marks_finished(self):
        payload = json.loads(_fake_group_result(group_index=5, next_index=None))
        payload["has_more"] = False
        plan = [{"tool": "get_paper_questions", "args": {}}]
        results = [nodes.ToolMessage(content=json.dumps(payload, ensure_ascii=False),
                                     tool_call_id="plan_0", name="get_paper_questions")]
        session = nodes._extract_drill_session({}, plan, results)
        assert session["finished"] is True


class TestPlannerSeesCursor:
    def test_prompt_includes_cursor_when_present(self, monkeypatch):
        """有游标时，planner 的 prompt 里必须出现卷子和下一组组号"""
        captured = {}

        def fake_invoke(fn, messages, context=None):
            captured["prompt"] = messages[0].content
            return nodes.AIMessage(content='{"need_tools": false, "steps": []}')

        monkeypatch.setattr(nodes, "safe_llm_invoke", fake_invoke)
        state = {
            "messages": [nodes.HumanMessage(content="继续下一组")],
            "plan": [], "tool_rounds": 0,
            "session": {"paper": "2019秋本部计算机网络模拟题", "type": "选择题",
                        "group_index": 1, "next_group_index": 2, "qids": ["A", "B"]},
        }
        nodes.planner_node(state)
        p = captured["prompt"]
        assert "2019秋本部计算机网络模拟题" in p
        assert "group_index=2" in p
        assert "A,B" in p

    def test_prompt_has_no_cursor_block_without_session(self, monkeypatch):
        """没有游标时不要凭空加一段进度块"""
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, messages, context=None: (
                                captured.update(prompt=messages[0].content)
                                or nodes.AIMessage(content='{"need_tools": false, "steps": []}')))
        nodes.planner_node({"messages": [nodes.HumanMessage(content="你好")],
                            "plan": [], "tool_rounds": 0})
        assert "当前刷题进度" not in captured["prompt"]

    def test_finished_paper_tells_planner_not_to_advance(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, messages, context=None: (
                                captured.update(prompt=messages[0].content)
                                or nodes.AIMessage(content='{"need_tools": false, "steps": []}')))
        nodes.planner_node({
            "messages": [nodes.HumanMessage(content="继续")], "plan": [], "tool_rounds": 0,
            "session": {"paper": "某卷", "group_index": 5, "next_group_index": None,
                        "finished": True}})
        p = captured["prompt"]
        assert "已经出完" in p
        assert "下一组要传" not in p


class TestCustomQuizToolIsRegistered:
    def test_plan_path_has_prepare_custom_quiz(self):
        """「另外出题」曾经在 Plan 路径上**根本没有工具**"""
        from course_agent import tools
        assert "prepare_custom_quiz" in tools.tools_by_name

    def test_it_returns_material_not_questions(self):
        from course_agent import tools
        out = json.loads(tools.prepare_custom_quiz.invoke({"topic": "死锁"}))
        assert "material" in out
        assert "guide" in out
        # 它不该返回题目本身
        assert "questions" not in out
        assert out["count"] == 5

    def test_both_surfaces_share_the_implementation(self):
        """助教侧与 Plan 侧必须是同一份实现"""
        from course_agent import tools, tutor_agent

        a = json.loads(tools.prepare_custom_quiz.invoke({"topic": "死锁", "count": 3}))
        sess = tutor_agent._get_session("custom-shared")
        b = tutor_agent.TOOL_HANDLERS["prepare_custom_quiz"](
            {"topic": "死锁", "count": 3}, sess)

        assert a["count"] == b["count"] == 3
        assert a["topic"] == b["topic"]
        # 字段名不同（共享层 guide / 助教侧 _guide），内容必须一致
        assert a["guide"] == b["_guide"]
        assert a["material"] == b["material"]
