"""A2（按知识点抽题）/ A3（答题三形态）/ A5（重做错题）的确定性判据

三件事都靠**代码**保证，不靠模型自觉：
  · 部分作答 → `grade_answers` 直接回报"还没答的题号"（题号从题库算）
  · 索要答案 → `get_answers` 返回题库原文（模型只负责原样转述）
  · 重做错题 → `practice_wrong` 从进度账本取错题 qid，**重出原题**
  · 按知识点练 → `practice_by_topic` 跨所有卷子抽题，匹配不到就明确说"自己出题"
"""
import json

import pytest

from course_agent import progress, qbank, quiz_service, tools

PAPER = "2025年秋操作系统期末试题（回忆版）"
QIDS = ["2025OS_Q1", "2025OS_Q2", "2025OS_Q3"]


@pytest.fixture(autouse=True)
def _fresh_progress(tmp_path, monkeypatch):
    """每条用例都从空账本开始（错题会跨测试累积，不隔离就会互相污染）"""
    monkeypatch.setattr(progress, "PROGRESS_PATH", tmp_path / "progress.json")
    yield


def _paper():
    paper, _ = quiz_service.resolve_paper(PAPER)
    return paper


def _qid_numbers(qids):
    _, numbers, _ = quiz_service.paper_index(PAPER)
    return [numbers[q] for q in qids]


# ============================================================
# A3：答题三形态
# ============================================================

class TestPartialAnswerReportsWhatIsMissing:
    """①只答部分 → 只批已答 + 提醒还剩哪几道（题号由代码算）"""

    def _grade(self, text, qids=None):
        return json.loads(tools.grade_answers.invoke(
            {"paper_source": PAPER, "user_answer_text": text, "qids": qids}))

    def test_missing_numbers_are_reported(self):
        out = self._grade("1.C", qids=QIDS)
        assert out["total"] == 1, out
        assert out.get("未答题号") == _qid_numbers(QIDS[1:]), out

    def test_all_answered_has_no_missing_note(self):
        out = self._grade("1.C 2.B 3.D", qids=QIDS)
        assert "未答题号" not in out, out

    def test_without_qids_it_does_not_guess(self):
        """没传 qids（= 批整卷）时算不出"本组还差几道"，不许瞎报"""
        out = self._grade("1.C")
        assert "未答题号" not in out, out

    def test_hint_tells_the_model_what_to_do(self):
        out = self._grade("1.C", qids=QIDS)
        hint = out.get("提示", "")
        assert "提醒" in hint and "explain_question" in hint, hint


class TestAskingForAnswersGetsTheBankAnswer:
    """③直接要答案 → 给题库原文，且不改写"""

    def test_answer_matches_the_bank_exactly(self):
        out = json.loads(tools.get_answers.invoke(
            {"paper_source": PAPER, "qids": [QIDS[0]]}))
        bank = {q["qid"]: q for q in qbank.paper_questions(PAPER)}[QIDS[0]]
        expected = qbank.grading_basis(bank)["answer"]
        assert out["answers"][0]["answer"] == expected, out

    def test_it_tells_the_model_not_to_regrade(self):
        out = json.loads(tools.get_answers.invoke(
            {"paper_source": PAPER, "qids": [QIDS[0]]}))
        assert "原样转述" in out["说明"] and "不要顺带批改" in out["说明"]

    def test_single_number_also_works(self):
        out = json.loads(tools.get_answers.invoke(
            {"paper_source": PAPER, "no": "1", "qids": [QIDS[0]]}))
        assert out["count"] == 1 and out["answers"][0]["answer"]


# ============================================================
# A5：重做错题（D4：重出原题，做对就移出）
# ============================================================

class TestPracticeWrong:
    def _set_wrong(self, qids):
        progress.record_quiz({"paper": PAPER, "group_index": 1,
                              "next_group_index": 2, "qids": QIDS})
        progress.record_grade(PAPER, {
            "total": len(qids), "correct": 0, "wrong": len(qids), "accuracy": 0.0,
            "results": [{"qid": q, "correct": False, "no": n}
                        for q, n in zip(qids, _qid_numbers(qids))]})

    def test_it_returns_the_wrong_questions(self):
        self._set_wrong(QIDS[:2])
        out = json.loads(tools.practice_wrong.invoke({"paper_source": PAPER}))
        assert out["count"] == 2, out
        assert [i["qid"] for i in out["items"]] == QIDS[:2]

    def test_it_does_not_leak_stem_or_answer(self):
        self._set_wrong(QIDS[:1])
        out = json.loads(tools.practice_wrong.invoke({"paper_source": PAPER}))
        item = out["items"][0]
        assert set(item) == {"paper", "qid", "no", "type", "topic", "image"}, item
        assert "题目原文" not in json.dumps(out) and "参考答案" not in json.dumps(out)

    def test_getting_them_right_empties_the_wrong_book(self):
        self._set_wrong(QIDS[:1])
        progress.record_grade(PAPER, {"total": 1, "correct": 1, "wrong": 0,
                                      "accuracy": 1.0,
                                      "results": [{"qid": QIDS[0], "correct": True}]})
        out = json.loads(tools.practice_wrong.invoke({"paper_source": PAPER}))
        assert out["count"] == 0, out
        assert "没有记录到错题" in out["hint"]

    def test_nothing_to_redo_is_said_plainly(self):
        out = json.loads(tools.practice_wrong.invoke({"paper_source": PAPER}))
        assert out["count"] == 0 and "hint" in out


# ============================================================
# A2：按知识点抽题
# ============================================================

class TestPracticeByTopic:
    def test_it_finds_questions_across_papers(self):
        out = json.loads(tools.practice_by_topic.invoke(
            {"topic": "死锁", "course": "操作系统", "count": 5}))
        assert out["count"] >= 1, out
        assert out.get("source") == "题库"
        assert len({i["paper"] for i in out["items"]}) >= 1
        for item in out["items"]:
            assert set(item) == {"paper", "qid", "no", "type", "topic", "image"}
            assert "死锁" in item["topic"] or "银行家" in item["topic"], item

    def test_course_filter_is_respected(self):
        out = json.loads(tools.practice_by_topic.invoke(
            {"topic": "传输层", "course": "计算机网络", "count": 3}))
        assert out["count"] >= 1, out
        for item in out["items"]:
            paper = next(p for p in qbank.list_papers() if p["source"] == item["paper"])
            assert paper["course"] == "计算机网络", item

    def test_wrong_course_filter_yields_nothing(self):
        """反例才能证明过滤真的生效：死锁的题都在操作系统卷子里，
        要"计算机网络的死锁题"就该是 0 —— 不过滤的话会照样返回操作系统那些题"""
        out = json.loads(tools.practice_by_topic.invoke(
            {"topic": "死锁", "course": "计算机网络"}))
        assert out["count"] == 0, out

    def test_synonyms_widen_the_match(self):
        """学生说"死锁"，题库里叫"银行家算法/资源分配"的题也该算 —— D5 的同义词表"""
        assert "银行家算法" in quiz_service._topic_terms("死锁")

    def test_count_is_capped(self):
        out = json.loads(tools.practice_by_topic.invoke(
            {"topic": "进程", "count": 50}))
        assert out["count"] <= 10, out

    def test_no_match_says_to_write_custom_questions(self):
        out = json.loads(tools.practice_by_topic.invoke({"topic": "量子力学"}))
        assert out["count"] == 0
        assert "prepare_custom_quiz" in out["hint"], out
