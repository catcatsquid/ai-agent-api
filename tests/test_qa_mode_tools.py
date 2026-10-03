"""LangGraph 模式（知识寻回BOT）的刷题工具测试

它与 LLM 助教模式共用 course_agent/qbank.py，但工具是无状态的，
所以组号必须由调用方携带 —— 曾经这里硬编码 group_index=1，
文档却写"获取下一组"，导致永远只能拿到第 1 组。
"""

import json

import pytest

from course_agent.tools import get_paper_questions, grade_answers, list_papers

PAPER = "2019秋本部计算机网络模拟题"


class TestGetPaperQuestions:
    def _call(self, **kw):
        return json.loads(get_paper_questions.invoke({"paper_source": PAPER, **kw}))

    def test_defaults_to_first_group(self):
        r = self._call()
        assert r["group_index"] == 1
        assert len(r["questions"]) == 5
        assert r["group_count"] == 12
        assert r["next_group_index"] == 2

    def test_can_advance_to_later_groups(self):
        """核心回归：传 group_index 必须真的换组"""
        first = self._call(group_index=1)
        second = self._call(group_index=2)
        third = self._call(group_index=3)
        assert first["qids"] != second["qids"]
        assert second["qids"] != third["qids"]
        assert set(first["qids"]).isdisjoint(second["qids"])
        assert second["group_index"] == 2
        assert second["next_group_index"] == 3

    def test_walk_all_groups_covers_the_paper(self):
        seen = []
        idx = 1
        while True:
            r = self._call(group_index=idx)
            if r.get("finished"):
                assert idx == r["total_groups"] + 1
                break
            seen.extend(r["qids"])
            if not r["has_more"]:
                break
            idx = r["next_group_index"]
        assert len(seen) == 40
        assert len(set(seen)) == 40

    def test_non_positive_group_index_falls_back_to_first(self):
        assert self._call(group_index=0)["group_index"] == 1
        assert self._call(group_index=-3)["group_index"] == 1

    def test_non_integer_group_index_is_rejected_by_schema(self):
        """参数类型由工具 schema 兜住，不该走到我们代码里

        （LangChain 的 @tool 会用 pydantic 校验参数，
          所以模型传个字符串会在入口就被挡掉，而不是悄悄当成第 1 组。）
        """
        from pydantic import ValidationError

        with pytest.raises((ValidationError, ValueError)):
            get_paper_questions.invoke({"paper_source": PAPER, "group_index": "abc"})

    def test_no_answer_leak(self):
        r = self._call()
        for q in r["questions"]:
            assert set(q) == {"no", "type", "topic", "image_url"}

    def test_unknown_paper_is_reported(self):
        out = get_paper_questions.invoke({"paper_source": "根本不存在的卷子"})
        assert "没能确定" in out or "选一个" in out


class TestGradeAnswers:
    def test_scoped_by_qids(self):
        g = json.loads(get_paper_questions.invoke({"paper_source": PAPER, "group_index": 3}))
        r = json.loads(grade_answers.invoke({
            "paper_source": PAPER,
            "user_answer_text": "ABCDA",
            "qids": g["qids"],
        }))
        assert r["total"] == 5
        assert {x["qid"] for x in r["results"]} == set(g["qids"])

    def test_without_qids_still_resolves_by_type(self):
        """不传 qids 时按题型分段解析，选择题答案不能落到综合题上"""
        r = json.loads(grade_answers.invoke({
            "paper_source": PAPER,
            "user_answer_text": "1.A 2.B 3.C",
        }))
        types = {x["type"] for x in r["results"]}
        assert types == {"选择题"}, f"落到了 {types}"

    def test_judge_answers(self):
        r = json.loads(grade_answers.invoke({
            "paper_source": PAPER,
            "user_answer_text": "1.对 2.错 3.对 4.错 5.对",
        }))
        assert r["total"] == 5
        assert {x["type"] for x in r["results"]} == {"判断题"}

    def test_unknown_paper(self):
        r = json.loads(grade_answers.invoke({
            "paper_source": "根本不存在的卷子", "user_answer_text": "1.A"}))
        assert "error" in r


class TestListPapers:
    def test_both_courses_present(self):
        out = list_papers.invoke({})
        assert "操作系统" in out
        assert "计算机网络" in out
        assert "2019秋本部计算机网络期末考试回忆版" in out

    def test_course_filter(self):
        out = list_papers.invoke({"course": "计算机网络"})
        assert "计算机网络" in out
        assert "2026春操作系统期末试题（回忆版）" not in out
