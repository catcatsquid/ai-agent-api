"""共享刷题层（quiz_service）的回归测试

这个模块存在的唯一理由：`tools.py`（Plan/ReAct 图用）和 `tutor_agent.py`
（手写循环用）曾经**各写一份**刷题逻辑，而且已经真实分叉 ——
tools.py 那边的阅卷缺 `basis_warning`，也没把题号传给 `build_llm_basis`。

下面的用例守住两件事：
  ① 两个运行面对同一份输入必须给出一致的结果（不能再各算各的）；
  ② 谁都不许把 grading / 统计逻辑再抄回去（重复的苗头要立刻红）。
"""
import json
import threading

from course_agent import quiz_service, tools, tutor_agent

PAPER = "2019秋本部计算机网络模拟题"


class TestBothSurfacesShareOneImplementation:
    def test_grading_outcome_is_identical_across_surfaces(self):
        """同一个作答，Plan/ReAct 工具与手写助教的核心结果必须一致"""
        answer = "1.D 2.B 3.D 4.A 5.C"

        mode_a = json.loads(tools.grade_answers.invoke({
            "paper_source": PAPER,
            "user_answer_text": answer,
        }))

        sess = tutor_agent._get_session("shared-grading")
        # 先出一组题，让会话知道当前 qids（模拟真实流程）
        tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": PAPER, "qtype": "选择题"}, sess)
        mode_b = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"paper_source": PAPER, "user_answer_text": answer}, sess)

        assert mode_a["total"] == mode_b["total"] > 0
        assert mode_a["correct"] == mode_b["correct"]
        assert mode_a["wrong"] == mode_b["wrong"]
        assert mode_a["needs_manual"] == mode_b["needs_manual"]
        assert mode_a["accuracy"] == mode_b["accuracy"]
        # 逐题结论也要一致
        assert [r["correct"] for r in mode_a["results"]] == \
               [r["correct"] for r in mode_b["results"]]

    def test_group_payload_matches_across_surfaces(self):
        """同一组题，两边给题号/知识点/图片必须一致"""
        mode_a = json.loads(tools.get_paper_questions.invoke(
            {"paper_source": PAPER, "group_index": 2}))

        sess = tutor_agent._get_session("shared-group")
        mode_b = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": PAPER, "qtype": "选择题", "group_index": 2}, sess)

        assert mode_a["qids"] == mode_b["qids"]
        assert [q["no"] for q in mode_a["questions"]] == \
               [q["no"] for q in mode_b["questions"]]
        assert [q["topic"] for q in mode_a["questions"]] == \
               [q["topic"] for q in mode_b["questions"]]

    def test_tutor_reply_text_comes_from_the_shared_layer(self):
        """助教的成品文本必须来自共享层 —— 比对**结构**，不比对文案

        原来这条用的是 `"答案怎么写都行" in reply_text`：那是拿字词当代理，
        于是**面向学生的措辞一改就红**，而"到底是不是同一份实现"并没被守住
        （实测：用户把一句面向学生的提示改成「请享用。」就让这条红了，那本是允许的）。

        现在改成与共享层对**同一组**的输出逐题对齐，并要求成品文本真的带上
        题号与图片 —— 措辞随便改，这条都不会误报。
        """
        sess = tutor_agent._get_session("shared-reply")
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": PAPER, "qtype": "选择题"}, sess)

        paper, _ = quiz_service.resolve_paper(PAPER)
        shared = quiz_service.group_payload(paper, qtype="选择题",
                                            group_index=r["group_index"])
        assert [q["image_url"] for q in r["questions"]] == \
               [q["image_url"] for q in shared["questions"]], "助教没走共享层"
        assert [q["topic"] for q in r["questions"]] == \
               [q["topic"] for q in shared["questions"]]
        assert r["reply_text"].count("![题目](") == len(r["questions"])

    def test_nobody_reintroduces_a_private_grader(self):
        """两个运行面都不许再自己实现阅卷/统计"""
        assert not hasattr(tools, "_grade_with_llm")
        assert not hasattr(tools, "_call_grader")
        assert not hasattr(tools, "_STATS_FILE")
        assert not hasattr(tutor_agent, "_grade_with_llm")
        assert not hasattr(tutor_agent, "_chat_json")
        assert not hasattr(tutor_agent, "_STATS_FILE")
        # 唯一实现就在 quiz_service
        assert callable(quiz_service.grade_with_llm)
        assert callable(quiz_service.chat_json)


class TestExplainIsShared:
    QID = "2019网模拟_Q1"

    def test_explain_payload_returns_question_and_answer(self):
        paper, _ = quiz_service.resolve_paper(PAPER)
        payload = quiz_service.explain_payload(paper, qids=[self.QID])
        assert payload["questions"], "讲题时必须真的给出题面"
        q = payload["questions"][0]
        assert q["题目原文"], "没有题面模型就讲不了题（这正是当初'我不会'只会说空话的根因）"
        assert "参考答案" in q and "参考要点" in q
        assert "讲解要求" in payload

    def test_whole_paper_default_vs_current_group_only(self):
        """没说讲哪道题时：Plan/ReAct 给整卷；助教收敛到当前组，不知道就报错"""
        paper, _ = quiz_service.resolve_paper(PAPER)

        whole = quiz_service.explain_payload(paper)
        assert len(whole["questions"]) > 5, "默认应当覆盖整卷"

        # 助教有"当前这一组"，所以能用 restrict_to 把歧义题号收敛掉
        narrow = quiz_service.explain_payload(
            paper, no=[1], restrict_to=[self.QID],
            fallback_qids=[self.QID], whole_paper_if_unspecified=False)
        assert len(narrow["questions"]) == 1

        unknown = quiz_service.explain_payload(
            paper, restrict_to=None, fallback_qids=None,
            whole_paper_if_unspecified=False)
        assert unknown.get("error") == "不知道要讲哪道题"

    def test_ambiguous_numbers_are_reported_not_guessed(self):
        """题号 1 在选择题/填空题/判断题/综合题里都存在 → 必须报歧义，不能猜"""
        paper, _ = quiz_service.resolve_paper(PAPER)
        payload = quiz_service.explain_payload(paper, no=[1])
        assert "questions" not in payload, "歧义时不该擅自挑一道题讲"
        assert payload.get("error") == "题号在不同题型里重复，需要指明是哪一道"
        assert payload.get("ambiguous")

    def test_restrict_to_disambiguates(self):
        """同一个歧义题号，给了 restrict_to 就能定位（这是助教侧的关键差异）"""
        paper, _ = quiz_service.resolve_paper(PAPER)
        payload = quiz_service.explain_payload(paper, no=[1], restrict_to=[self.QID])
        assert "questions" in payload
        assert len(payload["questions"]) == 1


class TestDivergenceIsFixed:
    """合并前两边已经真实分叉，这些用例守住合并后的行为"""

    def test_grade_with_llm_applies_basis_warning(self, monkeypatch):
        """参考答案存疑时必须在点评里说出来

        分叉点就在这：tools.py 那份 `_grade_with_llm` 没有这段，
        所以同一个"参考答案缺失"的填空题，走 Plan/ReAct 时学生会看到
        一句没有免责说明的点评，走手写助教时才有说明。
        """
        monkeypatch.setattr(quiz_service, "chat_json",
                            lambda *a, **k: {"items": [{"no": 1, "correct": True,
                                                        "comment": "要点答到了"}]})
        q = {"qid": "X", "type": "填空题", "answer": "", "topic": "t"}
        r = {"no": 1, "needs_llm": True, "basis_warning": "题库参考答案缺失"}
        out = quiz_service.grade_with_llm([(q, "学生作答", r)])
        assert len(out) == 1
        assert "题库参考答案缺失" in out[0]["explanation"]
        assert out[0]["correct"] is True
        assert "needs_llm" not in out[0]

    def test_basis_no_comes_from_result_not_recomputed(self, monkeypatch):
        """传给阅卷模型的题号必须用 result 里算好的那个

        分叉点之二：tools.py 那份调的是 `build_llm_basis(q)`（不传 no），
        题号会退回 q_num/display_num，跨题型就会串号。
        """
        seen = {}

        def fake_chat_json(system, user, max_tokens=1500):
            seen["user"] = user
            return {"items": []}

        monkeypatch.setattr(quiz_service, "chat_json", fake_chat_json)
        q = {"qid": "X", "type": "填空题", "answer": "参考答案", "topic": "t",
             "q_num": "4.1", "display_num": 7}
        r = {"no": 3, "needs_llm": True}
        quiz_service.grade_with_llm([(q, "作答", r)])
        assert '"no": 3' in seen["user"], f"题号没按 result 里的 3 传: {seen['user'][:200]}"

    def test_llm_failure_marks_manual_not_wrong(self, monkeypatch):
        """阅卷模型失败时必须标成"需人工核对"，绝不能静默判错"""
        def boom(*a, **k):
            raise RuntimeError("网络抖动")

        monkeypatch.setattr(quiz_service, "chat_json", boom)
        q = {"qid": "X", "type": "填空题", "answer": "A", "topic": "t"}
        r = {"no": 1, "needs_llm": True}
        out = quiz_service.grade_with_llm([(q, "作答", r)])
        assert out[0]["correct"] is None
        assert out[0]["needs_manual"] is True

    def test_both_surfaces_agree_on_stateful_session_flow(self):
        """同一会话连续出题 → 批改，两边的批次必须指向同一组题"""
        sess = tutor_agent._get_session("flow-consistency")
        served = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": PAPER, "qtype": "选择题"}, sess)
        assert len(served["qids"]) == 5

        # 同一次作答：显式带上这组 qids，两边结果必须一致
        text = "1.D 2.B 3.D 4.A 5.C"
        mode_a = json.loads(tools.grade_answers.invoke({
            "paper_source": PAPER, "user_answer_text": text, "qids": served["qids"]}))
        mode_b = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"paper_source": PAPER, "user_answer_text": text, "qids": served["qids"]}, sess)
        assert mode_a["accuracy"] == mode_b["accuracy"]
        assert mode_a["results"] == mode_b["results"]


class TestPracticeStatsAreLocked:
    def test_concurrent_increments_are_not_lost(self, monkeypatch, tmp_path_factory=None):
        """读-改-写必须持锁：原来 12 个并发只能记到 3 次"""
        import pathlib

        path = pathlib.Path(__file__).parent / "_tmp_stats_concurrent.json"
        path.unlink(missing_ok=True)
        monkeypatch.setattr(quiz_service, "_STATS_FILE", path)
        try:
            def bump():
                quiz_service.record_practice("并发测试卷")

            threads = [threading.Thread(target=bump) for _ in range(12)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert quiz_service.practice_count("并发测试卷") == 12
        finally:
            path.unlink(missing_ok=True)

    def test_record_then_read(self, monkeypatch):
        import pathlib

        path = pathlib.Path(__file__).parent / "_tmp_stats_roundtrip.json"
        path.unlink(missing_ok=True)
        monkeypatch.setattr(quiz_service, "_STATS_FILE", path)
        try:
            assert quiz_service.practice_count("P") == 0
            assert quiz_service.record_practice("P") == 1
            assert quiz_service.record_practice("P") == 2
            assert quiz_service.practice_count("P") == 2
        finally:
            path.unlink(missing_ok=True)

    def test_corrupt_file_does_not_crash(self, monkeypatch):
        import pathlib

        path = pathlib.Path(__file__).parent / "_tmp_stats_corrupt.json"
        path.write_text("{ 这不是 JSON", encoding="utf-8")
        monkeypatch.setattr(quiz_service, "_STATS_FILE", path)
        try:
            assert quiz_service.practice_count("P") == 0
            assert quiz_service.record_practice("P") == 1
        finally:
            path.unlink(missing_ok=True)
