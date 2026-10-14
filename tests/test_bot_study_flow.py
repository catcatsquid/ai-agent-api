"""BOT（Plan-and-Execute）那条产品路径的学习/复习行为

这个文件原来叫 `test_surface_parity.py` 的一部分 —— 名字说的是"两个运行面对齐"，
而这里面的测试**只跟 BOT 有关**（planner / synthesizer / 历史裁剪），
跟"对齐"没关系。拆出来是为了让文件名的承诺和内容对上。

这些用例全部来自**真实的学生会话发现**（不是凭空设计的边界）：

- `test_synthesizer_sees_the_session_state`：学生交小测答案那一轮**一个工具都不调**，
  所以 BOT 若只把学习书签注进 planner，写回答的模型就看不到"当前学习位置"。
- `test_missed_questions_forces_a_resend`：学生说"我没看到题目" → 确定性重发刚才那一组，
  不许只口头道歉（这条是从手写助教移植过来的）。
- `test_history_is_pruned_at_a_user_boundary` / `test_old_tool_results_are_compacted`：
  裁剪必须在 **user 处**切开，不能留下"声明了 tool_calls 却没有对应 ToolMessage"的孤儿。
- `test_quiz_answer_during_study_plans_no_tools`：讲课时学生交自拟小测答案 → 不许去调
  `grade_answers`（实测 BOT 在这条上 5 轮错 4 次，还会反问卷名）。

（两个运行面的**一致性**由 `test_surface_parity.py` 守；手写助教自己的行为由
`test_tutor_reference.py` 守。）
"""
import pytest

from course_agent import nodes


class TestBotCaughtUpWithTutor:
    """BOT 在"学习/复习"流程上曾经吃亏 —— 原因不是范式，是**状态注入放的位置**：
    助教注进"写回答的模型"的 system prompt，而 BOT 只注进了 planner。
    另外助教有两个守卫 BOT 一直没有。这里守"已经补齐"。
    """

    def test_synthesizer_sees_the_session_state(self, monkeypatch):
        """写回答的那个模型必须看得到学习书签（学生交小测答案那轮一个工具都不调）"""
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: (
                                captured.update(prompt=msgs[0].content)
                                or nodes.AIMessage(content="ok")))
        state = {
            "messages": [nodes.HumanMessage(content="我答一下：1.B 2.D")],
            "plan": [], "tool_rounds": 0,
            "session": {"study": {"mode": "复习", "course": "操作系统",
                                  "chapter": "死锁", "section": "死锁的概念",
                                  "last_source": "05-死锁.pdf",
                                  "last_read": "第 1-15 块", "next_offset": 15}},
        }
        nodes.synthesizer_node(state)
        p = captured["prompt"]
        assert "当前学习位置" in p, "synthesizer 没看到书签"
        assert "由你自己批" in p, "「小测由你自己批」必须在这一轮就可见"
        assert "不要**调 grade_answers" in p or "不要" in p

    def test_missed_questions_forces_a_resend(self, monkeypatch):
        """学生说"没看到题目" → 确定性重发，不许只口头道歉（移植自手写助教）"""
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: (
                                captured.update(prompt=msgs[0].content)
                                or nodes.AIMessage(content='{"need_tools": true, "steps": []}')))
        out = nodes.planner_node({
            "messages": [nodes.HumanMessage(content="我没看到题目啊")],
            "plan": [], "tool_rounds": 0,
            "session": {"paper": "2026春操作系统期末试题（回忆版）", "type": "选择题",
                        "group_index": 2, "next_group_index": 3}},
            {"configurable": {"thread_id": "t-resend"}})
        plan = out["plan"]
        assert len(plan) == 1 and plan[0]["tool"] == "get_paper_questions"
        assert plan[0]["args"]["group_index"] == 2, "应该重发**刚才那一组**"
        assert "2026春操作系统期末试题（回忆版）" == plan[0]["args"]["paper_source"]

    def test_normal_turn_is_not_overridden(self, monkeypatch):
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: nodes.AIMessage(
                                content='{"need_tools": true, "steps": '
                                        '[{"tool": "search_knowledge", '
                                        '"args": {"query": "死锁"}}]}'))
        out = nodes.planner_node({
            "messages": [nodes.HumanMessage(content="死锁是什么")],
            "plan": [], "tool_rounds": 0,
            "session": {"paper": "某卷", "group_index": 2, "next_group_index": 3}})
        assert out["plan"][0]["tool"] == "search_knowledge", "守卫不该误伤正常请求"

    def test_history_is_pruned_at_a_user_boundary(self):
        """BOT 原来没有任何裁剪 —— 裁剪必须在 user 处切开，不能留孤儿 tool"""
        msgs = []
        for i in range(30):
            msgs.append(nodes.HumanMessage(content=f"u{i}", id=f"h{i}"))
            ai = nodes.AIMessage(content="", tool_calls=[
                {"id": f"c{i}", "name": "search_knowledge", "args": {}}], id=f"a{i}")
            msgs.append(ai)
            msgs.append(nodes.ToolMessage(content="x" * 500, tool_call_id=f"c{i}",
                                          name="search_knowledge", id=f"t{i}"))
        pruned = nodes._prune_old_messages(msgs)
        removes = [p for p in pruned if type(p).__name__ == "RemoveMessage"]
        assert removes, "超长历史没有被裁剪"
        # 被删的必须整段（不会只删掉 tool 而留着声明它的 assistant）
        removed_ids = {p.id for p in removes}
        for i in range(30):
            if f"a{i}" in removed_ids:
                assert f"t{i}" in removed_ids, f"a{i} 被删但 t{i} 留着 → 孤儿 tool"

    def test_old_tool_results_are_compacted(self):
        msgs = [nodes.HumanMessage(content="u", id="h")]
        for i in range(4):
            msgs.append(nodes.AIMessage(content="", tool_calls=[
                {"id": f"c{i}", "name": "get_paper_questions", "args": {}}], id=f"a{i}"))
            msgs.append(nodes.ToolMessage(content="题" * 2000, tool_call_id=f"c{i}",
                                          name="get_paper_questions", id=f"t{i}"))
        pruned = nodes._prune_old_messages(msgs)
        replaced = [p for p in pruned if isinstance(p, nodes.ToolMessage)]
        assert replaced, "较早的工具结果没有被压缩"
        assert all(len(p.content) < 200 for p in replaced)
        # 最近 2 轮必须保留原文
        kept = {m.id for m in msgs if isinstance(m, nodes.ToolMessage)} - \
               {p.id for p in replaced}
        assert len(kept) == nodes._BOT_KEEP_TOOL_TURNS

    def test_quiz_answer_during_study_plans_no_tools(self, monkeypatch):
        """讲课中学生交自拟小测答案 → 不规划工具，交给 synthesizer 自批

        实测：BOT 在这条上 5 轮里错 4 次（去调 grade_answers 并反问卷名）。
        """
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: nodes.AIMessage(
                                content='{"need_tools": true, "steps": '
                                        '[{"tool": "grade_answers", '
                                        '"args": {"user_answer_text": "1.B"}}]}'))
        for answer in ("我答一下：1.B 2.D 3.对 4.银行家算法", "ACDBE", "答案：1.A"):
            out = nodes.planner_node({
                "messages": [nodes.HumanMessage(content=answer)],
                "plan": [], "tool_rounds": 0,
                "session": {"study": {"course": "操作系统", "section": "进程的引入"}}},
                {"configurable": {"thread_id": "t-ans"}})
            assert out["plan"] == [], f"「{answer}」还规划了工具: {out['plan']}"

    def test_real_quiz_answer_still_uses_grade_answers(self, monkeypatch):
        """但真在做题库卷子时，提交答案**必须**走 grade_answers（守卫不能误伤）"""
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: nodes.AIMessage(
                                content='{"need_tools": true, "steps": '
                                        '[{"tool": "grade_answers", "args": '
                                        '{"paper_source": "2026卷", '
                                        '"user_answer_text": "ACDBE"}}]}'))
        out = nodes.planner_node({
            "messages": [nodes.HumanMessage(content="ACDBE")],
            "plan": [], "tool_rounds": 0,
            "session": {"paper": "2026春操作系统期末试题（回忆版）"}},
            {"configurable": {"thread_id": "t-real"}})
        assert out["plan"][0]["tool"] == "grade_answers"

    def test_quiz_answer_pattern_recognition(self):
        for yes in ("1.B 2.D", "ACDBE", "我答一下：1.A", "答案：对错对",
                    "1、对 2、错"):
            assert nodes._looks_like_quiz_answer(yes), yes
        for no in ("死锁是什么", "我要复习操作系统", "继续讲", "没看到题目"):
            assert not nodes._looks_like_quiz_answer(no), no

    def test_short_history_is_left_alone(self):
        msgs = [nodes.HumanMessage(content="u", id="h"),
                nodes.AIMessage(content="a", id="a")]
        assert nodes._prune_old_messages(msgs) == []
