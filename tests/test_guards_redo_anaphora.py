"""守卫 8/9（1.11.2）：重做错题、指代追问

两条都是"模型该调工具却没调/调错"的兜底 —— 提示词在这种窄场景上不稳定，
按项目惯例用代码守住（只在这两个形态生效，不改变正常提问）。
"""
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from course_agent import nodes, progress


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_PATH", tmp_path / "progress.json")
    monkeypatch.setattr(nodes, "safe_llm_invoke", lambda *a, **k: AIMessage(
        content=json.dumps({"need_tools": True, "steps": [
            {"tool": "list_papers", "args": {}}]}, ensure_ascii=False)))
    return tmp_path / "progress.json"


def _run(text, prev=None):
    msgs = []
    if prev:
        msgs.append(HumanMessage(content=prev))
        msgs.append(AIMessage(content="（上一轮回答）"))
    msgs.append(HumanMessage(content=text))
    out = nodes.planner_node({"messages": msgs, "plan": [],
                              "tool_rounds": 0, "session": {}})
    return [s["tool"] for s in (out.get("plan") or [])]


class TestWrongRedoGuard:
    def test_it_forces_practice_wrong_when_ledger_has_wrong_qids(self):
        progress.record_grade("某卷", {"paper": "某卷", "total": 2, "correct": 0,
                                       "wrong": 2, "accuracy": 0.0,
                                       "results": [{"qid": "q1", "correct": False},
                                                   {"qid": "q2", "correct": False}]})
        tools = _run("我要重做错题")
        assert tools and tools[0] == "practice_wrong", tools

    def test_it_does_not_force_when_there_are_no_wrong_qids(self):
        tools = _run("我要重做错题")           # 账本为空
        assert "practice_wrong" not in tools, tools

    def test_normal_quiz_request_is_untouched(self):
        tools = _run("给我出第 1 组题")
        assert "practice_wrong" not in tools, tools


class TestAnaphoraGuard:
    def test_short_reference_question_uses_the_previous_message(self):
        tools = _run("那怎么预防？", prev="死锁是怎么产生的？")
        assert tools == ["search_knowledge"], tools

    def test_it_does_not_fire_without_previous_context(self):
        tools = _run("那怎么预防？")            # 第一句就是指代：没有"上一句"可借
        assert tools != ["search_knowledge"], tools

    def test_it_does_not_fire_when_the_question_names_a_topic(self):
        tools = _run("那死锁怎么预防？", prev="进程是什么？")
        assert "search_knowledge" not in tools or tools != ["search_knowledge"], tools

    def test_long_questions_are_not_treated_as_anaphora(self):
        tools = _run("那你能把刚才讲到的死锁产生条件再详细展开一下吗？", prev="死锁是怎么产生的？")
        assert tools != ["search_knowledge"], tools
