"""全局学习进度（`course_agent/progress.py`）的测试

判据选的是**会真的出错**的那几处：
  · 评测线程（eval-* / probe-*）不许写 —— 否则用户的学习进度被评测刷满
  · 并发写不许丢更新（`practice_stats` 上真实发生过：12 次并发只记到 3 次）
  · 进度文件损坏/不存在时**绝不能让一整轮问答崩掉**
  · "刷到一半"的定义 = 记录写到第几组（用户拍板），刷完是 next_group_index=None
  · 开场建议由代码生成且**明说不要再问"学到哪"**（这是这个功能的本意）

文件隔离：PROGRESS_PATH 指向 tmp_path，绝不碰真实 runtime/progress.json。
"""
import json
import threading

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from course_agent import nodes, progress, tools


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(progress, "PROGRESS_PATH", tmp_path / "progress.json")
    return tmp_path / "progress.json"


STUDY = {"mode": "学习", "course": "操作系统", "chapter": "死锁",
         "section": "死锁：四个必要条件", "source": "03-OS调度.pdf", "status": "讲解中"}


# ============================================================
# ① 空账本 / 坏文件：不许崩
# ============================================================

class TestItNeverBreaksATurn:
    def test_empty_store_gives_empty_text(self):
        assert progress.load()["courses"] == {}
        assert progress.summary() == ""
        assert progress.brief() == ""          # 没进度就别说"已知进度"

    def test_corrupt_file_is_treated_as_empty(self, _isolated):
        _isolated.write_text("{这不是 JSON", encoding="utf-8")
        assert progress.load()["papers"] == {}
        assert progress.brief() == ""
        # 坏文件之后还能正常写（不许从此永久坏掉）
        progress.record_study(STUDY)
        assert "操作系统" in progress.summary()

    def test_non_dict_json_is_tolerated(self, _isolated):
        _isolated.write_text("[1,2,3]", encoding="utf-8")
        assert progress.summary() == ""

    def test_missing_dir_is_created(self, tmp_path, monkeypatch):
        nested = tmp_path / "a" / "b" / "progress.json"
        monkeypatch.setattr(progress, "PROGRESS_PATH", nested)
        progress.record_study(STUDY)
        assert nested.exists()


# ============================================================
# ② 评测/探针线程不写
# ============================================================

class TestEvalThreadsNeverWrite:
    @pytest.mark.parametrize("tid", ["eval-quiz_no_leak-0-abc", "probe-1234abcd"])
    def test_ephemeral_thread_writes_nothing(self, tid):
        progress.record_study(STUDY, thread_id=tid)
        progress.record_quiz({"paper": "某卷", "group_index": 1,
                              "next_group_index": 2, "qids": ["a"]}, thread_id=tid)
        progress.record_grade("某卷", {"total": 5, "correct": 4}, thread_id=tid)
        assert progress.load()["courses"] == {}
        assert progress.load()["papers"] == {}

    def test_real_thread_still_writes(self):
        progress.record_study(STUDY, thread_id="3f2a-正常会话")
        assert progress.load()["courses"]


# ============================================================
# ③ 学到哪
# ============================================================

class TestStudyProgress:
    def test_summary_mentions_course_and_section(self):
        progress.record_study(STUDY)
        text = progress.summary()
        assert "操作系统" in text and "死锁" in text and "讲解中" in text

    def test_visits_accumulate(self):
        for _ in range(3):
            progress.record_study(STUDY)
        data = progress.load()
        sec = list(data["courses"]["操作系统"]["sections"].values())[0]
        assert sec["visits"] == 3

    def test_empty_fields_do_not_wipe_previous_values(self):
        """实测模型会传 chapter="" 把上一轮记住的章抹掉 —— 进度账本不许被抹

        而且**不许新开一条**：`chapter=""` 时如果按 `source|""` 建 key，会多出
        一条重复的节（"已学了几节"就会数错）。所以这里同时断言**只有一条**。
        """
        progress.record_study(STUDY)
        progress.record_study({**STUDY, "chapter": "", "section": ""})
        sections = progress.load()["courses"]["操作系统"]["sections"]
        assert len(sections) == 1, f"空章名不许新开一条：{sections}"
        sec = list(sections.values())[0]
        assert sec["chapter"] == "死锁"
        assert sec["title"] == "死锁：四个必要条件"
        assert sec["visits"] == 2

    def test_reading_offset_is_remembered_on_the_same_section(self):
        progress.record_study(STUDY)
        progress.record_read("操作系统", "03-OS调度.pdf", 15)
        data = progress.load()
        sections = data["courses"]["操作系统"]["sections"]
        assert len(sections) == 1, f"同一份课件不许记成两条：{sections}"
        sec = list(sections.values())[0]
        assert sec["last_offset"] == 15
        assert "读到第 15 块之后" in progress.summary()

    def test_two_courses_are_both_listed(self):
        progress.record_study(STUDY)
        progress.record_study({**STUDY, "course": "计算机网络",
                               "chapter": "传输层", "source": "Ch5.pdf"})
        text = progress.summary()
        assert "操作系统" in text and "计算机网络" in text

    def test_summary_is_capped(self):
        for i in range(20):
            progress.record_study({**STUDY, "course": f"课{i}",
                                   "chapter": f"章{i}", "source": f"s{i}.pdf"})
        assert len(progress.summary(max_lines=5).splitlines()) <= 5


# ============================================================
# ④ 刷到哪 / 刷到一半（用户拍板的定义）
# ============================================================

class TestQuizProgress:
    def _quiz(self, gi=1, nxt=2):
        progress.record_quiz({"paper": "2025年秋操作系统期末试题（回忆版）",
                              "type": "选择题", "group_index": gi,
                              "next_group_index": nxt, "qids": ["q1", "q2", "q3"]})

    def test_half_done_means_next_group_is_known(self):
        self._quiz(gi=1, nxt=2)
        text = progress.summary()
        assert "做到第 1 组" in text and "下一组是第 2 组" in text

    def test_finished_means_no_next_group(self):
        self._quiz(gi=3, nxt=None)
        data = progress.load()
        rec = data["papers"]["2025年秋操作系统期末试题（回忆版）"]
        assert rec["finished"] is True
        assert "已刷完" in progress.summary()

    def test_grade_marks_that_group_and_records_score(self):
        self._quiz(gi=2, nxt=3)
        progress.record_grade("2025年秋操作系统期末试题（回忆版）",
                              {"total": 5, "correct": 4, "wrong": 1, "accuracy": 0.8})
        rec = progress.load()["papers"]["2025年秋操作系统期末试题（回忆版）"]
        assert rec["groups"]["2"]["status"] == "已批改"
        assert rec["groups"]["2"]["score"] == "4/5"
        assert "上次 4/5" in progress.summary()

    def test_grade_without_quiz_does_not_crash(self):
        progress.record_grade("某卷", {"total": 5, "correct": 3})
        assert progress.load()["papers"]["某卷"]["last_grade"]["correct"] == 3


class TestWrongQuestions:
    """A5 错题记录（D4）：做错留下、做对移出；需人工判定的（correct=None）不算错"""

    PAPER = "2025年秋操作系统期末试题（回忆版）"

    def _grade(self, rows):
        progress.record_grade(self.PAPER, {
            "total": len(rows),
            "correct": sum(1 for r in rows if r[1] is True),
            "wrong": sum(1 for r in rows if r[1] is False),
            "accuracy": 0.5,
            "results": [{"qid": q, "correct": c, "no": i + 1}
                        for i, (q, c) in enumerate(rows)]})

    def _wrong(self):
        return progress.load()["papers"][self.PAPER].get("wrong_qids") or []

    def test_wrong_qids_are_recorded(self):
        self._grade([("q1", False), ("q2", True), ("q3", False)])
        assert self._wrong() == ["q1", "q3"]

    def test_getting_it_right_removes_it(self):
        """D4：做对就移出（否则"重做错题"永远做不完）"""
        self._grade([("q1", False), ("q2", False)])
        self._grade([("q1", True)])
        assert self._wrong() == ["q2"], self._wrong()

    def test_still_wrong_stays(self):
        self._grade([("q1", False)])
        self._grade([("q1", False)])
        assert self._wrong() == ["q1"]

    def test_manual_grading_is_not_counted_as_wrong(self):
        """correct=None 是"没判出来"，不能凭猜塞进错题"""
        self._grade([("q1", None), ("q2", False)])
        assert self._wrong() == ["q2"]

    def test_summary_shows_the_count(self):
        self._grade([("q1", False), ("q2", False)])
        assert "错题 2 道" in progress.summary()

    def test_it_is_visible_through_the_tool(self):
        self._grade([("q1", False)])
        data = json.loads(tools.get_learning_progress.invoke({}))
        assert data["detail"]["papers"][self.PAPER]["wrong_qids"] == ["q1"]


# ============================================================
# ⑤ 开场建议（这个功能的本意）
# ============================================================

class TestOpeningBrief:
    def test_brief_contains_progress_and_says_dont_ask_again(self):
        progress.record_study(STUDY)
        text = progress.brief()
        assert "不要再问" in text, f"没告诉模型别重问学到哪：{text!r}"
        assert "死锁" in text, "建议里没带上真实进度"

    def test_brief_offers_a_next_step_for_half_done_paper(self):
        progress.record_quiz({"paper": "某卷", "group_index": 1,
                             "next_group_index": 2, "qids": ["q1"]})
        text = progress.brief()
        assert "刷题" in text and "第 2 组" in text, text

    def test_brief_always_offers_the_four_modes(self):
        """用户要的是四选一：接着学 / 自拟练习 / 整套卷子 / 复习

        有半刷的卷子时**不许**把"自拟练习"顶掉（那样只剩三个选项）。
        """
        progress.record_study(STUDY)
        progress.record_quiz({"paper": "某卷", "group_index": 1,
                             "next_group_index": 2, "qids": ["q1"]})
        text = progress.brief()
        for k in ("接着学", "自拟练习", "刷题", "复习"):
            assert k in text, f"缺了「{k}」这个选项：{text}"

    def test_brief_suggests_review_when_sections_exist(self):
        progress.record_study(STUDY)
        assert "复习" in progress.brief()

    def test_brief_without_papers_suggests_custom_practice(self):
        progress.record_study(STUDY)
        assert "AI 出题" in progress.brief() or "自拟" in progress.brief()


# ============================================================
# ⑥ 并发与写入原子性
# ============================================================

class TestConcurrencyAndAtomicity:
    def test_concurrent_writes_do_not_lose_updates(self):
        """`practice_stats` 真实踩过：12 个并发调用只记到 3 次（读-改-写互相覆盖）"""
        n = 12

        def work(i):
            progress.record_study({**STUDY, "section": f"第 {i} 次"})

        threads = [threading.Thread(target=work, args=(i,)) for i in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        sec = list(progress.load()["courses"]["操作系统"]["sections"].values())[0]
        assert sec["visits"] == n, f"丢了更新：只记到 {sec['visits']} 次"

    def test_no_temp_file_left_behind(self, _isolated):
        progress.record_study(STUDY)
        assert not _isolated.with_suffix(".json.tmp").exists()

    def test_written_file_is_valid_utf8_json(self, _isolated):
        progress.record_study(STUDY)
        data = json.loads(_isolated.read_text(encoding="utf-8"))
        assert data["user"] == "local"
        assert data["version"] == progress._VERSION
        # 中文不许被转成 \uXXXX（人工看盘的时候要能直接读）
        assert "操作系统" in _isolated.read_text(encoding="utf-8")


# ============================================================
# ⑦ 接进图里：新会话说"我想学习"时，提示词里必须已有进度
# ============================================================

class TestItReachesTheModel:
    """这个功能的本意：开新窗口说"我想学习"，Agent 不再问"你学到哪了"""

    def _capture(self, monkeypatch, node, user="我想学习", session=None):
        captured = {}

        def fake_invoke(fn, msgs, context=None, *a, **k):
            captured["prompt"] = "\n".join(
                getattr(m, "content", "") or "" for m in msgs)
            return AIMessage(content='{"need_tools": false, "steps": []}')

        monkeypatch.setattr(nodes, "safe_llm_invoke", fake_invoke)
        state = {"messages": [HumanMessage(content=user)], "plan": [],
                 "tool_rounds": 0, "session": session or {}}
        node(state)
        return captured.get("prompt", "")

    def test_planner_sees_the_progress_when_asking_to_study(self, monkeypatch):
        progress.record_study(STUDY)
        prompt = self._capture(monkeypatch, nodes.planner_node)
        assert "已知进度" in prompt, f"planner 没拿到进度：{prompt[-300:]!r}"
        assert "死锁" in prompt
        assert "不要再问" in prompt, "没告诉它别重问学到哪"

    def test_synthesizer_sees_it_too(self, monkeypatch):
        progress.record_study(STUDY)
        prompt = self._capture(monkeypatch, nodes.synthesizer_node)
        assert "已知进度" in prompt and "死锁" in prompt

    def test_it_is_not_injected_mid_quiz_answer(self, monkeypatch):
        """学生正在答题（"1.C 2.B"）时不许插进度 —— 会把这一轮带偏，也白费 token"""
        progress.record_study(STUDY)
        prompt = self._capture(monkeypatch, nodes.planner_node,
                               user="1.C 2.B 3.D",
                               session={"study": {"course": "操作系统"}})
        assert "已知进度" not in prompt, prompt[-300:]

    def test_no_progress_no_noise(self, monkeypatch):
        """没有进度、**也不是开场**时不许插进度块（空会话不是注入的理由）"""
        prompt = self._capture(monkeypatch, nodes.planner_node, user="死锁是什么？")
        assert "已知进度" not in prompt

    def test_empty_store_still_gets_an_opening_brief_on_the_kickoff(self, monkeypatch):
        """空账本也要有开场块（用户 2026-10-18 实测的要求）

        原文：「我说我想学操作系统（没有明确说要学哪章），它并没有先给我展示
        当前的进度然后给我推荐」。旧实现是"没进度就直接开讲"，实测它讲了一节
        **记忆里的**通用内容；现在改成先说"还没有记录 + 建议从第一章开始"。
        """
        prompt = self._capture(monkeypatch, nodes.planner_node)
        assert "还没有任何学习记录" in prompt
        assert "不要开始讲内容" in prompt

    def test_a_concept_question_on_a_fresh_thread_gets_no_progress_block(self, monkeypatch):
        """回归（评测 `knowledge_answer_streams` 抓到的）：新会话里问概念题，
        进度块一注入就把"某卷做到第 1 组"也带进去 → planner 误判成刷题，
        真的调了 get_paper_questions + grade_answers。空会话不是注入的理由。
        """
        progress.record_study(STUDY)
        prompt = self._capture(monkeypatch, nodes.planner_node,
                              user="死锁的四个必要条件是什么？")
        assert "已知进度" not in prompt, prompt[-300:]

    def test_the_tool_returns_the_summary(self):
        progress.record_quiz({"paper": "某卷", "group_index": 1,
                              "next_group_index": 2, "qids": ["q1"]})
        data = json.loads(tools.get_learning_progress.invoke({}))
        assert "做到第 1 组" in data["summary"]
        assert data["detail"]["papers"], data

    def test_the_tool_says_when_there_is_nothing(self):
        data = json.loads(tools.get_learning_progress.invoke({}))
        assert "还没有" in data["summary"]


# ============================================================
# ⑧ 开场守卫：只说"我想学习"时不许直接开讲（用户明确要的行为）
# ============================================================

class TestKickoffAsksBeforeTeaching:
    """实测：不加守卫时，学生只说"我想学习"，它直接 read_section 讲了一整节，
    学生根本没机会在「接着学 / 自拟练习 / 刷卷子 / 复习」里选。"""

    def _plan(self, monkeypatch, user, session=None):
        monkeypatch.setattr(
            nodes, "safe_llm_invoke",
            lambda *a, **k: AIMessage(content=json.dumps(
                {"need_tools": True, "steps": [
                    {"tool": "read_section", "args": {"course": "操作系统", "section": 1}}]},
                ensure_ascii=False)))
        out = nodes.planner_node({"messages": [HumanMessage(content=user)],
                                  "plan": [], "tool_rounds": 0,
                                  "session": session or {}})
        return out.get("plan") or []

    def test_kickoff_with_progress_plans_no_tool(self, monkeypatch):
        progress.record_study(STUDY)
        plan = self._plan(monkeypatch, "我想学习")
        assert plan == [], f"开场不该调工具（应先让他选模式）：{plan}"
        assert nodes.route_after_plan({"plan": plan}) == "synthesizer"

    def test_kickoff_with_progress_block_is_injected(self, monkeypatch):
        """不调工具之后，回答由 synthesizer 用 CHITCHAT + 进度块写出来"""
        progress.record_study(STUDY)
        captured = {}

        def fake_invoke(fn, msgs, context=None, *a, **k):
            captured["p"] = "\n".join(getattr(m, "content", "") or "" for m in msgs)
            return AIMessage(content="好的")

        monkeypatch.setattr(nodes, "safe_llm_invoke", fake_invoke)
        nodes.synthesizer_node({"messages": [HumanMessage(content="我想学习")],
                                "plan": [], "tool_rounds": 0, "session": {}})
        assert "已知进度" in captured["p"] and "不要开始讲内容" in captured["p"]

    def test_without_progress_it_also_asks_first(self, monkeypatch):
        """没有进度时**也要先问**（用户 2026-10-18 实测要求，这条改了产品行为）

        旧行为是"没进度就直接开讲"（先给资料地图），实测后果：学生只说想学
        操作系统，它直接讲了一节"操作系统基本概念"，一条进度都没报、也没给他
        选"继续/复习/刷题"的机会。现在空账本同样先展示状态 + 建议。
        """
        plan = self._plan(monkeypatch, "我想学习")
        assert plan == [], f"空账本的开场也不许直接开讲：{plan}"

    def test_a_vague_course_request_is_a_kickoff_too(self, monkeypatch):
        """用户原话「我们来学习一下操作系统吧」—— 旧判据（固定句式 + 长度≤20）漏了它"""
        plan = self._plan(monkeypatch, "我们来学习一下操作系统吧！")
        assert plan == [], f"这种最常见的开场也不许直接开讲：{plan}"

    def test_i_want_to_learn_os_is_a_kickoff(self):
        """「我想学操作系统」只有"想学"、没有"学习"，旧判据同样漏了"""
        assert nodes._is_study_kickoff("我想学操作系统")
        assert nodes._is_study_kickoff("学操作系统吧")

    def test_a_concrete_target_is_not_a_kickoff(self, monkeypatch):
        """说清楚学什么时照常规划（"我想学习死锁那一章" 不许被当成开场）"""
        progress.record_study(STUDY)
        plan = self._plan(monkeypatch, "我想学习死锁那一章")
        assert plan, "说了具体章节就该照常规划"

    def test_answering_questions_is_not_a_kickoff(self, monkeypatch):
        progress.record_study(STUDY)
        assert self._plan(monkeypatch, "1.C 2.B 3.D"), "答题轮次不许被当成开场"


class TestOpeningNeverDropsAnOption:
    """实测：提示里写了"四条全部列出"，模型仍然只说三条（漏掉「自拟练习」）"""

    def test_missing_modes_are_detected(self):
        progress.record_study(STUDY)
        assert "自拟练习" in progress.missing_modes("1. 接着学 2. 刷题 3. 复习")

    def test_ensure_modes_appends_what_is_missing(self):
        progress.record_study(STUDY)
        out, added = progress.ensure_modes("1. 接着学\n2. 刷题\n3. 复习")
        assert len(added) == 1 and added[0].startswith("自拟练习"), added
        assert "自拟练习" in out
        assert out.startswith("1. 接着学"), "原有内容不许被重排"
        assert "2. 自拟" not in out, "补进来的那条不许带原编号"

    def test_custom_practice_option_names_the_current_chapter(self):
        """W2：出题选项要带上**当前章节**，否则模型只能从进度里自己猜 topic"""
        progress.record_study(STUDY)
        text = progress.brief()
        line = next(ln for ln in text.splitlines() if "自拟练习" in ln)
        assert "死锁" in line, line
        assert "操作系统" in line, line

    def test_nothing_is_added_when_all_four_are_there(self):
        progress.record_study(STUDY)
        text = "\n".join(
            ln for ln in progress.brief().splitlines() if ln[:1].isdigit())
        out, added = progress.ensure_modes(text)
        assert added == [] and out == text

    def test_synthesizer_output_is_patched_end_to_end(self, monkeypatch):
        """真跑一遍 synthesizer：只给了三条 → 出来必须是四条"""
        progress.record_study(STUDY)
        monkeypatch.setattr(
            nodes, "safe_llm_invoke",
            lambda *a, **k: AIMessage(content="1. 接着学\n2. 刷题\n3. 复习"))
        out = nodes.synthesizer_node({
            "messages": [HumanMessage(content="我想学习")],
            "plan": [], "tool_rounds": 0, "session": {}})
        answer = out["messages"][-1].content
        for k in ("接着学", "自拟练习", "刷题", "复习"):
            assert k in answer, f"缺了「{k}」：{answer}"


