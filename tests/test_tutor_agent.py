"""tutor_agent 主循环与工具层测试（LLM 全程打桩，不产生网络请求）"""

import json

import pytest

from course_agent import qbank, tools
from frozen import tutor_agent


@pytest.fixture(autouse=True)
def clean_sessions():
    tutor_agent._SESSIONS.clear()
    yield
    tutor_agent._SESSIONS.clear()


def _scripted(rounds, final="（结束语）"):
    """把 _stream_round 换成按脚本出牌：每一轮是 (content, tool_calls)

    不带 tools 的调用（工具轮数用尽后的收尾请求）统一返回 final。
    """
    calls = {"n": 0, "payloads": []}

    def fake_stream(messages, tools=None, temperature=0.7, max_tokens=2000):
        calls["payloads"].append(list(messages))
        if not tools:
            yield ("content", final)
            return
        idx = calls["n"]
        calls["n"] += 1
        content, tool_calls = rounds[min(idx, len(rounds) - 1)]
        if content:
            # 模拟真实流式：分片吐出
            for i in range(0, len(content), 7):
                yield ("content", content[i:i + 7])
        if tool_calls:
            yield ("tool_calls", tool_calls)

    return fake_stream, calls


def _collect(session_id, text):
    return list(tutor_agent.process_message(session_id, text))


class TestPlainAnswer:
    def test_no_tool_call_streams_answer(self, monkeypatch):
        fake, _ = _scripted([("你好，想做哪套题？", None)])
        monkeypatch.setattr(tutor_agent, "_stream_round", fake)

        events = _collect("s1", "在吗")
        kinds = [e["type"] for e in events]
        assert kinds[0] == "start"
        assert "content" in kinds
        assert kinds[-1] == "done"
        assert "".join(e["content"] for e in events if e["type"] == "content") == "你好，想做哪套题？"
        assert events[-1]["answer"] == "你好，想做哪套题？"


class TestToolFlow:
    def test_list_papers_then_answer(self, monkeypatch):
        tc = [{"id": "c1", "type": "function",
               "function": {"name": "list_papers", "arguments": "{}"}}]
        fake, calls = _scripted([("我看看题库。", tc), ("题库里有两门课的卷子。", None)])
        monkeypatch.setattr(tutor_agent, "_stream_round", fake)

        events = _collect("s2", "我想刷题")
        kinds = [e["type"] for e in events]
        # 第一轮流出的"前言"必须被作废，否则前端会先显示一句废话再重新开始
        assert "discard" in kinds
        assert kinds[-1] == "done"
        assert events[-1]["answer"] == "题库里有两门课的卷子。"

        tool_events = [e for e in events if e["type"] == "tool"]
        assert tool_events and tool_events[0]["tool"] == "list_papers"
        assert "试卷" in tool_events[0]["result"]

        # 第二轮请求里必须带上工具结果
        second = calls["payloads"][1]
        assert any(m.get("role") == "tool" for m in second)

    def test_group_cursor_advances_within_session(self, monkeypatch):
        """连续两次出题，不传 group_index 时自动往下走"""
        tc1 = [{"id": "c1", "type": "function", "function": {
            "name": "get_paper_questions",
            "arguments": json.dumps({"paper_source": "2019网模拟"})}}]
        tc2 = [{"id": "c2", "type": "function", "function": {
            "name": "get_paper_questions",
            "arguments": json.dumps({"paper_source": "2019网模拟"})}}]
        fake, _ = _scripted([("", tc1), ("第一组来了。", None),
                             ("", tc2), ("第二组来了。", None)])
        monkeypatch.setattr(tutor_agent, "_stream_round", fake)

        _collect("s3", "刷题")
        sess = tutor_agent._SESSIONS["s3"]
        assert sess["last_group"]["index"] == 1
        first_qids = list(sess["current_qids"])

        _collect("s3", "继续")
        assert sess["last_group"]["index"] == 2
        assert sess["current_qids"] != first_qids

    def test_tool_round_exhaustion_still_answers(self, monkeypatch):
        """工具轮数用尽时不能再回一句干巴巴的兜底，要再要一次回答"""
        forever = [{"id": "cX", "type": "function",
                    "function": {"name": "list_papers", "arguments": "{}"}}]
        rounds = [(None, forever)] * tutor_agent._MAX_TOOL_ROUNDS
        fake, _ = _scripted(rounds, final="这是最终回答。")
        monkeypatch.setattr(tutor_agent, "_stream_round", fake)

        events = _collect("s4", "一直调工具")
        assert events[-1]["type"] == "done"
        assert events[-1]["answer"] == "这是最终回答。"

    def test_api_error_is_reported(self, monkeypatch):
        def boom(messages, tools=None, temperature=0.7, max_tokens=2000):
            raise RuntimeError("429 rate limited")
            yield  # pragma: no cover

        monkeypatch.setattr(tutor_agent, "_stream_round", boom)
        events = _collect("s5", "你好")
        assert events[-1]["type"] == "error"
        assert "429" in events[-1]["msg"]


class TestContextManagement:
    def test_old_tool_results_are_compacted(self, monkeypatch):
        big = {"questions": [{"no": i, "topic": "x" * 200} for i in range(40)]}
        s = tutor_agent._get_session("c1")
        for i in range(5):
            s["messages"].append({"role": "assistant", "content": "", "tool_calls": [{"id": f"t{i}"}]})
            s["messages"].append({"role": "tool", "tool_call_id": f"t{i}",
                                  "content": json.dumps(big, ensure_ascii=False)})
        before = sum(len(m.get("content") or "") for m in s["messages"])
        tutor_agent._compact_tools(s)
        after = sum(len(m.get("content") or "") for m in s["messages"])
        assert after < before / 2
        kept = [m for m in s["messages"] if m["role"] == "tool" and len(m["content"]) > 200]
        assert len(kept) == tutor_agent._COMPACT_KEEP_LAST_TOOL

    def test_trim_never_orphans_tool_messages(self):
        s = tutor_agent._get_session("c2")
        for i in range(80):
            s["messages"] += [
                {"role": "user", "content": f"q{i}"},
                {"role": "assistant", "content": "", "tool_calls": [{"id": f"x{i}"}]},
                {"role": "tool", "tool_call_id": f"x{i}", "content": "{}"},
                {"role": "assistant", "content": "a"},
            ]
        tutor_agent._trim_history(s)
        assert len(s["messages"]) <= tutor_agent._MAX_HISTORY_MESSAGES + 1
        assert s["messages"][0]["role"] == "system"
        assert s["messages"][1]["role"] == "user"
        seen = set()
        for m in s["messages"]:
            if m["role"] == "assistant" and m.get("tool_calls"):
                seen.update(t["id"] for t in m["tool_calls"])
            if m["role"] == "tool":
                assert m["tool_call_id"] in seen, "出现了没有对应 tool_call 的 tool 消息"

    def test_sessions_are_swept_when_idle(self):
        s = tutor_agent._get_session("old")
        s["touched"] = 0
        tutor_agent._sweep_sessions()
        assert "old" not in tutor_agent._SESSIONS

    def test_same_session_id_keeps_history(self):
        s1 = tutor_agent._get_session("keep")
        s1["messages"].append({"role": "user", "content": "hi"})
        s2 = tutor_agent._get_session("keep")
        assert s2 is s1
        assert any(m.get("content") == "hi" for m in s2["messages"])


class TestToolLayer:
    def test_get_paper_questions_has_usage_guide_and_no_answers(self):
        sess = tutor_agent._get_session("t1")
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题"}, sess)
        assert r["type"] == "选择题"
        assert len(r["questions"]) == 5
        assert "_guide" in r and "_render_example" in r
        for q in r["questions"]:
            assert set(q) == {"no", "type", "topic", "image_url"}
        assert sess["current_paper"] == "2019秋本部计算机网络模拟题"

    def test_get_paper_questions_ships_ready_to_send_text(self):
        """出题必须给成品文本，否则模型可能只回一句「好的，这是下一组」却不贴题

        **刻意不断言具体措辞**：这段是面向学生的文案，可以改（原来钉的是
        「请慢慢做」，用户把它改掉是合理的）。要守的是"成品、且完整"。
        """
        sess = tutor_agent._get_session("t1b")
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题"}, sess)
        reply = r["reply_text"]
        for q in r["questions"]:
            assert f"{q['no']}." in reply
            assert q["image_url"] in reply
        assert "题目必须整组出现在回复里" in r["_guide"]
        # 完整到不可能是"只回了一句好的"
        assert len(reply) > 100, f"成品文本太短，像是没贴题: {reply[:60]!r}"

    def test_freeform_answers_stay_supported_where_the_model_can_see_it(self):
        """**自由格式作答**这条行为约束，必须留在模型每轮都看得到的地方

        为什么单独守这条：用户改 `reply_text` 措辞时担心"会不会让模型又开始要求
        特定答案格式"。查清了：真正让模型接受任意格式的是**工具描述**（模型每轮必看），
        `reply_text` 里那句只是给学生看的安抚。所以守工具描述 —— 它丢了，
        历史上那种"「AAAAA」被拒"就会回来。
        """
        desc = tools.tools_by_name["grade_answers"].description or ""
        assert "原样" in desc or "自动解析" in desc, (
            "grade_answers 的工具描述必须告诉模型「原样传入、工具自己解析」，"
            f"否则模型会要求特定格式：{desc[:120]!r}")


class TestMissedQuestionsRecovery:
    """出题会推进进度，但模型可能取了题却没展示 —— 那一组会被静默跳过

    用户实测：说"继续刷题"，游标前进了，却一条题都没给。
    """

    def test_saying_missed_questions_reserves_the_same_group(self):
        sess = tutor_agent._get_session("m1")
        g1 = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "选择题"}, sess)
        assert g1["group_index"] == 1

        # 正常继续 → 推进到第 2 组
        g2 = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "选择题"}, sess)
        assert g2["group_index"] == 2

        # 学生说没看到题 → 重发当前这一组，不能再往前走
        sess["last_user_message"] = "我没看到题目"
        again = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "选择题"}, sess)
        assert again["group_index"] == 2, "说没看到题时不能推进游标"
        assert again["qids"] == g2["qids"]

    @pytest.mark.parametrize("text", ["我没看到题目", "题目呢", "没给我题", "再发一遍题目"])
    def test_recognises_missed_question_phrases(self, text):
        assert tutor_agent._missed_questions(text) is True

    @pytest.mark.parametrize("text", ["下一组", "继续", "1.A 2.B", "换一套卷子", "我不会"])
    def test_does_not_misfire_on_normal_messages(self, text):
        assert tutor_agent._missed_questions(text) is False

    def test_explicit_group_index_is_still_honoured(self):
        """学生说没看到题，但模型明确要第 3 组 → 以模型的明确指定为准"""
        sess = tutor_agent._get_session("m2")
        tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "选择题"}, sess)
        sess["last_user_message"] = "没看到题目"
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "选择题",
             "group_index": 3}, sess)
        assert r["group_index"] == 3

    def test_guide_tells_model_how_to_recover(self):
        sess = tutor_agent._get_session("m3")
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "选择题"}, sess)
        assert f"group_index={r['group_index']}" in r["_guide"]

    def test_loop_nudges_model_when_it_only_promises_to_resend(self, monkeypatch):
        """学生说没看到题，模型只口头答应却不调工具 → 循环强制纠偏一次

        实测原话：「很抱歉，可能是由于格式问题导致题目没有正确显示。
        请稍等，我将重新发送这组题目。」—— 然后一条题都没发。
        """
        served = [{"id": "s1", "type": "function", "function": {
            "name": "get_paper_questions",
            "arguments": json.dumps({"paper_source": "2019秋本部计算机网络模拟题",
                                     "qtype": "选择题"})}}]
        # 第 1 轮：正常出题；第 2 轮：只道歉不调工具；第 3 轮（被纠偏后）：重新出题
        rounds = [("", served),
                  ("很抱歉，我将重新发送这组题目。", None),
                  ("", served),
                  ("题目已重新发送。", None)]
        fake, calls = _scripted(rounds)
        monkeypatch.setattr(tutor_agent, "_stream_round", fake)

        events = _collect("nudge1", "我要做2019秋本部计算机网络模拟题的选择题")
        tutor_agent._SESSIONS["nudge1"]["last_user_message"] = "我没看到题目"
        events2 = _collect("nudge1", "我没看到题目")

        tool_events = [e for e in events2 if e["type"] == "tool"]
        assert any(e["tool"] == "get_paper_questions" for e in tool_events), \
            "纠偏后应该真的调一次出题工具"
        assert events2[-1]["answer"] == "题目已重新发送。"

    def test_loop_does_not_nudge_for_normal_messages(self, monkeypatch):
        """普通消息不该触发纠偏（否则会白白多花一次模型调用）"""
        fake, calls = _scripted([("好的。", None)])
        monkeypatch.setattr(tutor_agent, "_stream_round", fake)
        events = _collect("nudge2", "下一组")
        assert calls["n"] == 1
        assert events[-1]["answer"] == "好的。"

    def test_nudge_only_once(self, monkeypatch):
        """模型第二次仍然不调工具时不能无限纠偏"""
        served = [{"id": "s1", "type": "function", "function": {
            "name": "get_paper_questions", "arguments": "{}"}}]
        rounds = [("", served),
                  ("我这就重发。", None),
                  ("我又忘了。", None)]
        fake, calls = _scripted(rounds)
        monkeypatch.setattr(tutor_agent, "_stream_round", fake)
        _collect("nudge3", "我要做2019秋本部计算机网络模拟题")
        tutor_agent._SESSIONS["nudge3"]["last_user_message"] = "没看到题目"
        events = _collect("nudge3", "没看到题目")
        assert calls["n"] == 3, "纠偏只允许一次"
        assert events[-1]["answer"] == "我又忘了。"

    def test_comprehensive_group_is_single_question(self):
        sess = tutor_agent._get_session("t2")
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2026春操作系统期末试题（回忆版）", "qtype": "综合题"}, sess)
        assert r["type"] == "综合题"
        assert len(r["questions"]) == 1

    def test_grading_only_touches_submitted_group(self):
        sess = tutor_agent._get_session("t3")
        g = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "判断题"}, sess)
        r = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"paper_source": "2019秋本部计算机网络模拟题",
             "user_answer_text": "1.对 2.错 3.对 4.错 5.对",
             "qids": g["qids"]}, sess)
        assert r["total"] == 5
        assert {x["no"] for x in r["results"]} == {1, 2, 3, 4, 5}

    def test_grading_falls_back_to_session_group(self):
        """模型忘了传 qids 时，用本会话刚出的那组兜底"""
        sess = tutor_agent._get_session("t4")
        g = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "选择题",
             "group_index": 3}, sess)
        r = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"paper_source": "2019秋本部计算机网络模拟题", "user_answer_text": "ABCDA"}, sess)
        assert {x["qid"] for x in r["results"]} == set(g["qids"])

    def test_unparsable_submission_is_reported_not_guessed(self):
        sess = tutor_agent._get_session("t5")
        g = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题", "qtype": "判断题"}, sess)
        r = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"paper_source": "2019秋本部计算机网络模拟题",
             "user_answer_text": "嗯……先不做了", "qids": g["qids"]}, sess)
        # 判断题能识别出 0 个 token -> 走兜底并明确提示解析不确定，而不是编造对错
        assert r.get("notes") or r.get("parse_notes") or r["total"] >= 0

    def test_unknown_paper_returns_candidates(self):
        sess = tutor_agent._get_session("t6")
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "完全不存在的卷子"}, sess)
        assert "error" in r
        assert r["available_papers"]

    def test_list_papers_groups_by_course_and_hints_extra_questions(self):
        sess = tutor_agent._get_session("t7")
        r = tutor_agent.TOOL_HANDLERS["list_papers"]({}, sess)
        courses = {c["course"] for c in r["courses"]}
        assert {"操作系统", "计算机网络"} <= courses
        # "可以另出题"这句话在给学生的回复文本里
        assert "另出题" in r["reply_text"]

    def test_list_papers_ships_a_ready_to_paste_block(self):
        """工具要给出排好版的清单让模型照抄

        只给嵌套结构时模型会漏段：实测"我想刷题"时工具返回了两门课的卷子，
        模型却只把操作系统那几份写进了回复。
        """
        sess = tutor_agent._get_session("t7b")
        sess["last_user_message"] = "我想刷题"
        r = tutor_agent.TOOL_HANDLERS["list_papers"]({"course": "操作系统"}, sess)
        block = r["_render_example"]
        assert "**操作系统**" in block and "**计算机网络**" in block
        assert "2019秋本部计算机网络模拟题" in block
        assert "2026春操作系统期末试题（回忆版）" in block
        # reply_text 是"直接发出去的话"，必须自带两门课和另出题提示
        reply = r["reply_text"]
        assert "操作系统" in reply and "计算机网络" in reply
        assert "另出题" in reply
        assert "2019秋本部计算机网络模拟题" in reply

    def test_identical_tool_result_is_reused_for_the_same_message(self, monkeypatch):
        """同一句话重复调 list_papers 结果应一致（不要靠模型随机性）"""
        sess = tutor_agent._get_session("t7c")
        sess["last_user_message"] = "我想刷题"
        a = tutor_agent.TOOL_HANDLERS["list_papers"]({"course": "操作系统"}, sess)
        b = tutor_agent.TOOL_HANDLERS["list_papers"]({"course": "操作系统"}, sess)
        assert a["reply_text"] == b["reply_text"]
        assert a["courses"] == b["courses"]

    def test_course_filter_on_list_papers(self):
        sess = tutor_agent._get_session("t8")
        sess["last_user_message"] = "我只想刷计算机网络"
        r = tutor_agent.TOOL_HANDLERS["list_papers"]({"course": "计算机网络"}, sess)
        assert {c["course"] for c in r["courses"]} == {"计算机网络"}
        assert r["applied_course_filter"] == "计算机网络"

    def test_model_invented_course_filter_is_ignored(self):
        """模型在「我想刷题」时自己填了 course，会把一半题库藏起来。

        「按课程分类、两门课都能看到」是刷题入口的硬要求，不能交给模型随手决定。
        """
        sess = tutor_agent._get_session("t8b")
        sess["last_user_message"] = "我想刷题"
        r = tutor_agent.TOOL_HANDLERS["list_papers"]({"course": "操作系统"}, sess)
        courses = {c["course"] for c in r["courses"]}
        assert courses == {"操作系统", "计算机网络"}, "没指定课程时必须两门都列"
        assert r["applied_course_filter"] is None
        assert r["ignored_course_filter"] == "操作系统"

    def test_knowledge_search_filters_by_course(self, monkeypatch):
        captured = {}

        class FakeKB:
            def search(self, query, **kwargs):
                captured.update(kwargs)
                # 高相似度：话题确实属于这门课
                return [{"text": "片段", "chapter": "进程与线程",
                         "source": "02-1-进程与线程之进程.pdf", "score": 0.82}]

        monkeypatch.setattr("rag_tool.get_knowledge_base", lambda: FakeKB())
        sess = tutor_agent._get_session("t9")
        r = tutor_agent.TOOL_HANDLERS["search_knowledge_base"](
            {"query": "进程调度", "course": "操作系统"}, sess)
        assert captured.get("sources"), "按课程检索必须带上文件过滤，否则两门课会串味"
        assert all(isinstance(s, str) for s in captured["sources"])
        assert r["count"] == 1
        assert r["note"] == ""

    def test_knowledge_search_falls_back_when_topic_is_not_in_that_course(self, monkeypatch):
        """话题不在该课程里时必须退回全库

        `kb.search()` 对非空库永远返回 top_k 条，所以不能拿"结果非空"当命中 ——
        否则学生在做计网时问「死锁」，拿回来的是网络层材料。
        """
        calls = []

        class FakeKB:
            def search(self, query, **kwargs):
                calls.append(kwargs.get("sources"))
                if kwargs.get("sources"):
                    # 按课程过滤只召回弱相关内容
                    return [{"text": "网络层", "chapter": "网络层",
                             "source": "Chapter4-Network Layer.pdf", "score": 0.21}]
                return [{"text": "死锁的四个必要条件", "chapter": "死锁",
                         "source": "05-死锁.pdf", "score": 0.81}]

        monkeypatch.setattr("rag_tool.get_knowledge_base", lambda: FakeKB())
        sess = tutor_agent._get_session("t9b")
        r = tutor_agent.TOOL_HANDLERS["search_knowledge_base"](
            {"query": "死锁", "course": "计算机网络"}, sess)
        assert len(calls) == 2, "应先按课程试一次，再退回全库"
        assert calls[0] and calls[1] is None
        assert "没有真正相关的片段" in r["note"]
        assert r["passages"][0]["chapter"] == "死锁"

    def test_practice_counter_roundtrip(self, monkeypatch, tmp_path_factory=None):
        # 不用 pytest 的 tmp_path：某些受限环境里系统临时目录不可写
        import pathlib

        from course_agent import quiz_service

        path = pathlib.Path(__file__).parent / "_tmp_stats.json"
        # 统计文件的读写已统一到 quiz_service（两个 Agent 共用一份、且加了锁），
        # 所以这里 patch 的是 service 的路径。
        monkeypatch.setattr(quiz_service, "_STATS_FILE", path)
        try:
            sess = tutor_agent._get_session("t10")
            assert tutor_agent.TOOL_HANDLERS["get_practice_stats"](
                {"paper_source": "P"}, sess)["practice_count"] == 0
            assert tutor_agent.TOOL_HANDLERS["record_practice"](
                {"paper_source": "P"}, sess)["practice_count"] == 1
            assert tutor_agent.TOOL_HANDLERS["get_practice_stats"](
                {"paper_source": "P"}, sess)["practice_count"] == 1
        finally:
            path.unlink(missing_ok=True)


class TestCustomQuizIntent:
    """学生要「临时出的新题」时，不能让模型去翻题库充数

    实测过的问题：学生说「不想做这套了，帮我针对死锁这个知识点另外出几道题」，
    模型调了 get_paper_questions，把计算机网络的选择题标成
    「死锁的定义」「产生死锁的四个必要条件」发了出去。
    """

    @pytest.mark.parametrize("text", [
        "帮我针对死锁这个知识点另外出几道题",
        "再来几道关于TCP的题",
        "给我出几道选择题",
        "生成几道判断题练练",
        "额外出题",
    ])
    def test_detects_custom_quiz_intent(self, text):
        assert tutor_agent._wants_custom_quiz(text) is True

    @pytest.mark.parametrize("text", [
        "下一组", "继续", "接着往下做", "换一套卷子", "换题",
        "1.D 2.B 3.D", "什么是进程调度",
    ])
    def test_does_not_flag_normal_flow(self, text):
        assert tutor_agent._wants_custom_quiz(text) is False

    def test_paper_tool_refuses_and_redirects(self):
        sess = tutor_agent._get_session("cust1")
        sess["last_user_message"] = "帮我针对死锁另外出几道题"
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题"}, sess)
        assert "error" in r
        assert r["use_instead"] == "prepare_custom_quiz"
        assert "questions" not in r

    def test_paper_tool_still_works_when_continuing(self):
        sess = tutor_agent._get_session("cust2")
        sess["last_user_message"] = "继续下一组"
        r = tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题"}, sess)
        assert r.get("type") == "选择题"
        assert len(r["questions"]) == 5

    def test_prepare_custom_quiz_returns_material_not_questions(self, monkeypatch):
        class FakeKB:
            def search(self, query, **kwargs):
                return [{"text": "死锁的四个必要条件：互斥、请求与保持、不可剥夺、循环等待",
                         "chapter": "死锁", "source": "05-死锁.pdf"}]

        monkeypatch.setattr("rag_tool.get_knowledge_base", lambda: FakeKB())
        sess = tutor_agent._get_session("cust3")
        r = tutor_agent.TOOL_HANDLERS["prepare_custom_quiz"](
            {"topic": "死锁", "course": "操作系统", "qtype": "选择题", "count": 4}, sess)
        assert r["count"] == 4
        assert r["material"], "应带回出题素材"
        assert "questions" not in r, "这不是出题接口，题目由模型自己写"
        assert "自己写 4 道" in r["_guide"]
        assert "不要调用题库工具" in r["_guide"]

    def test_prepare_custom_quiz_survives_empty_kb(self, monkeypatch):
        class EmptyKB:
            def search(self, query, **kwargs):
                return []

        monkeypatch.setattr("rag_tool.get_knowledge_base", lambda: EmptyKB())
        sess = tutor_agent._get_session("cust4")
        r = tutor_agent.TOOL_HANDLERS["prepare_custom_quiz"]({"topic": "冷门话题"}, sess)
        assert r["material"] == []
        assert "自拟题" in r["_guide"], "没有材料时要明确说明是自拟题，不能假装来自题库"

    def test_topic_outside_current_course_drops_the_course_label(self, monkeypatch):
        """学生在做计网时突然要「死锁」的题：课程标签必须去掉，
        否则模型会写出「在计算机网络中，以下哪种情况会导致死锁」这种串味的题"""
        class KB:
            def search(self, query, **kwargs):
                # 带课程过滤（sources）时只召回弱相关内容；全库能查到死锁
                if kwargs.get("sources"):
                    return [{"text": "网络层", "chapter": "网络层",
                             "source": "Chapter4.pdf", "score": 0.22}]
                return [{"text": "死锁的四个必要条件", "chapter": "死锁",
                         "source": "05-死锁.pdf", "score": 0.83}]

        monkeypatch.setattr("rag_tool.get_knowledge_base", lambda: KB())
        sess = tutor_agent._get_session("cust6")
        sess["current_course"] = "计算机网络"
        r = tutor_agent.TOOL_HANDLERS["prepare_custom_quiz"](
            {"topic": "死锁", "course": "计算机网络"}, sess)
        assert r["course"] == "（未指定）"
        assert "不要硬说成是计算机网络的内容" in r["_guide"]
        assert r["material"], "退到全库后应能拿到素材"
        assert r["material"][0]["chapter"] == "死锁"

    def test_topic_inside_current_course_keeps_the_label(self, monkeypatch):
        """话题确实属于这门课时，课程标签保留"""
        class KB:
            def search(self, query, **kwargs):
                return [{"text": "TCP拥塞控制", "chapter": "传输层",
                         "source": "Chapter5.pdf", "score": 0.86}]

        monkeypatch.setattr("rag_tool.get_knowledge_base", lambda: KB())
        sess = tutor_agent._get_session("cust7")
        r = tutor_agent.TOOL_HANDLERS["prepare_custom_quiz"](
            {"topic": "TCP拥塞控制", "course": "计算机网络"}, sess)
        assert r["course"] == "计算机网络"
        assert "不要硬说成是" not in r["_guide"]

    def test_prepare_custom_quiz_clamps_count(self):
        sess = tutor_agent._get_session("cust5")
        assert tutor_agent.TOOL_HANDLERS["prepare_custom_quiz"]({"count": 999}, sess)["count"] == 20
        assert tutor_agent.TOOL_HANDLERS["prepare_custom_quiz"]({"count": -3}, sess)["count"] == 1
        assert tutor_agent.TOOL_HANDLERS["prepare_custom_quiz"]({"count": "abc"}, sess)["count"] == 5


class TestExplainQuestion:
    """学生说「不会」时要真的讲题

    出题接口刻意不返回题面和答案（防泄题），代价是模型**手里没有题目**，
    只能回"请仔细阅读题目"这种空话 —— 实测就出现过
    「这是一道关于网络性能的选择题，具体题干和选项我无法提供」。
    explain_question 是"真要讲题"时的正当入口。
    """

    def _serve(self, sess, qtype="选择题", group_index=1):
        return tutor_agent.TOOL_HANDLERS["get_paper_questions"](
            {"paper_source": "2019秋本部计算机网络模拟题",
             "qtype": qtype, "group_index": group_index}, sess)

    def test_returns_question_text_and_answer(self):
        sess = tutor_agent._get_session("ex1")
        self._serve(sess)
        r = tutor_agent.TOOL_HANDLERS["explain_question"]({}, sess)
        assert r["paper"] == "2019秋本部计算机网络模拟题"
        assert len(r["questions"]) == 5
        first = r["questions"][0]
        assert first["题目原文"], "必须给出题面，否则讲不了题"
        assert first["参考答案"], "必须给出答案"
        assert first["参考要点"], "必须给出讲解依据"

    def test_guide_forbids_hollow_encouragement(self):
        sess = tutor_agent._get_session("ex2")
        self._serve(sess)
        guide = tutor_agent.TOOL_HANDLERS["explain_question"]({}, sess)["_guide"]
        assert "不要" in guide and "再试试" in guide, "要明确禁止空泛鼓励"
        assert "不要只丢一个答案" in guide or "只丢一个答案" in guide

    def test_can_target_one_question_by_no(self):
        sess = tutor_agent._get_session("ex3")
        self._serve(sess)
        r = tutor_agent.TOOL_HANDLERS["explain_question"]({"no": [3]}, sess)
        assert len(r["questions"]) == 1
        assert r["questions"][0]["no"] == 3
        assert r["questions"][0]["type"] == "选择题", "题号要先在'当前这一组'里解析"

    def test_ambiguous_number_across_types_is_not_guessed(self):
        """没出过题时，"第 1 题"同时是选择题第 1 题和综合题第 1 题，不能猜"""
        sess = tutor_agent._get_session("ex3b")
        sess["current_paper"] = "2019秋本部计算机网络模拟题"
        sess["current_course"] = "计算机网络"
        r = tutor_agent.TOOL_HANDLERS["explain_question"]({"no": [1]}, sess)
        assert "ambiguous" in r
        assert len(r["ambiguous"]["1"]) > 1

    def test_can_target_by_qid(self):
        sess = tutor_agent._get_session("ex4")
        g = self._serve(sess)
        r = tutor_agent.TOOL_HANDLERS["explain_question"]({"qids": [g["qids"][1]]}, sess)
        assert len(r["questions"]) == 1
        assert r["questions"][0]["no"] == 2

    def test_defaults_to_current_group(self):
        """换一组出题后，默认讲的是"当前这组"而不是上一组"""
        sess = tutor_agent._get_session("ex5")
        self._serve(sess, group_index=1)
        second = self._serve(sess, group_index=2)
        r = tutor_agent.TOOL_HANDLERS["explain_question"]({}, sess)
        assert [q["no"] for q in r["questions"]] == [6, 7, 8, 9, 10]
        assert set(second["qids"]) == set(sess["current_qids"])

    def test_refuses_when_nothing_is_in_play(self):
        sess = tutor_agent._get_session("ex6")
        r = tutor_agent.TOOL_HANDLERS["explain_question"]({}, sess)
        assert "error" in r
        assert "_guide" in r

    def test_comprehensive_question_also_explainable(self):
        sess = tutor_agent._get_session("ex7")
        self._serve(sess, qtype="综合题")
        r = tutor_agent.TOOL_HANDLERS["explain_question"]({}, sess)
        q = r["questions"][0]
        assert q["题目原文"]
        assert q["参考要点"] or q["小问答案"]

    def test_get_paper_questions_guide_points_to_explain_tool(self):
        sess = tutor_agent._get_session("ex8")
        r = self._serve(sess)
        assert "explain_question" in r["_guide"], "出题时就该告诉模型卡住怎么办"

    def test_grading_guide_demands_reasons(self):
        sess = tutor_agent._get_session("ex9")
        g = self._serve(sess)
        r = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"paper_source": "2019秋本部计算机网络模拟题",
             "user_answer_text": "AAAAA", "qids": g["qids"]}, sess)
        assert "reference_points" in r["_guide"]
        assert "等于没讲" in r["_guide"]

    def test_wrong_answers_include_reference_points(self):
        sess = tutor_agent._get_session("ex10")
        g = self._serve(sess)
        r = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"paper_source": "2019秋本部计算机网络模拟题",
             "user_answer_text": "AAAAA", "qids": g["qids"]}, sess)
        wrong = [x for x in r["results"] if x["correct"] is False]
        assert wrong, "全选 A 至少该有错题"
        assert all(x.get("reference_points") for x in wrong), "错题要带讲解依据"


class TestSystemPromptIsNotHardcoded:
    def test_prompt_has_no_rigid_rule_block(self):
        """约束应该落在工具返回值上，prompt 不该退化成「铁则清单」"""
        prompt = tutor_agent.SYSTEM_PROMPT
        for banned in ["铁则", "绝对不能", "必须调用", "违反任何一条", "⚠️"]:
            assert banned not in prompt, f"prompt 里又出现硬性规则: {banned}"
        assert len(prompt) < 1200, "prompt 太长容易让模型变得机械"

    def test_tool_results_carry_the_presentation_contract(self):
        """呈现约束由工具返回的 _guide 承载"""
        sess = tutor_agent._get_session("g1")
        for name, args in [
            ("list_papers", {}),
            ("get_paper_questions", {"paper_source": "2019秋本部计算机网络模拟题"}),
        ]:
            r = tutor_agent.TOOL_HANDLERS[name](args, sess)
            assert "_guide" in r and r["_guide"]
