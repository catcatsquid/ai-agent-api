"""用户实测报的 bug 的回归测试（AI 出 5 道选择题，学生答 ACDBA）

用户贴出来的 trace：
    📋 1. [grade_answers] grade_answers
    🔧 步骤 1/2: 工具1
    🔧 步骤 2/2: grade_answers
    🔧 {"error": "没能确定试卷: 2025OS_Q1,2025OS_Q2,2025OS_Q3,2025OS_Q4,2025OS_Q5"}
    很抱歉…我将再次尝试批改你的答案…
    get_paper_answers {"paper_source": "…", "answers": ["ACDBA"]}

四个缺陷：
  1. planner 把 qids 列表填进了 `paper_source`（我的提示没说填哪个参数）
  2. grade_answers 只会拿这串东西当卷名解析 → 直接失败
  3. 批改失败后模型**编造了一个不存在的工具名**并写进了回复
  4. 前端多出一个不存在的步骤「工具1」（AIMessage 被当工具结果渲染）
"""
import json

from langchain_core.messages import AIMessage, ToolMessage

from course_agent import nodes, quiz_service, server, tools

QIDS = ["2025OS_Q1", "2025OS_Q2", "2025OS_Q3", "2025OS_Q4", "2025OS_Q5"]
PAPER = "2025年秋操作系统期末试题（回忆版）"
BAD = ",".join(QIDS)


class TestQidsInPaperSource:
    """缺陷 1+2：把 qids 填进 paper_source 必须能被**确定性**纠正"""

    def test_qids_are_recognized_in_text(self):
        assert quiz_service.qids_in_text(BAD) == QIDS
        assert quiz_service.qids_in_text("") == []
        assert quiz_service.qids_in_text(PAPER) == []   # 正常卷名不该被误认

    def test_normalize_moves_qids_out_of_paper_source(self):
        ps, qids = quiz_service.normalize_grade_args(BAD)
        assert ps == PAPER, "没用 qid 反查出卷子"
        assert qids == QIDS
        # 数组形式（JSON 转义过的）也要认得出来 —— 提示里就是这么给的
        assert quiz_service.normalize_grade_args(json.dumps(QIDS, ensure_ascii=False))[1] == QIDS

    def test_normalize_keeps_correct_input_untouched(self):
        ps, qids = quiz_service.normalize_grade_args(PAPER, ["2025OS_Q1"])
        assert ps == PAPER
        assert qids == ["2025OS_Q1"]

    def test_normalize_falls_back_to_session_paper(self):
        """空 paper_source 才退回会话里的当前试卷；模型明确说的卷名不覆盖"""
        assert quiz_service.normalize_grade_args("", None, PAPER)[0] == PAPER
        assert quiz_service.normalize_grade_args("别的卷", None, PAPER)[0] == "别的卷"

    def test_normalize_resolves_paper_from_qids_alone(self):
        """没有卷名、只有 qids 时，用 qid 反查出卷子（确定性）"""
        ps, qids = quiz_service.normalize_grade_args("", QIDS)
        assert ps == PAPER
        assert qids == QIDS

    def test_grade_answers_actually_grades(self):
        """端到端：拿用户那个错输入调工具，必须真的批改出结果"""
        d = json.loads(tools.grade_answers.invoke(
            {"paper_source": BAD, "user_answer_text": "ACDBA"}))
        assert "error" not in d, d
        assert d["total"] == 5
        assert d["correct"] + d["wrong"] + d["needs_manual"] == 5


class TestListArgsTolerateTheWaysModelsWriteThem:
    """缺陷 5（2026-10-16 评测 `dont_know_calls_explain` 抓到）：

    模型讲第 1 题时调成 `explain_question(paper_source=…, no="1")`，
    而 schema 写的是 `list[int]` → pydantic `ValidationError: Input should be a
    valid list`，工具**根本没执行**，学生只得到一段绕开题目的泛泛而谈。

    与 `read_section` 的 `"int | str"` 同一条纪律：schema 层放行现实里的写法，
    容错放实现里。下面既测"收敛函数本身"，也**真的调工具**（否则 schema 层
    还可能挡在外面 —— 这正是当初那条 bug 的形态）。
    """

    def test_single_value_becomes_a_one_item_list(self):
        assert tools._as_list(1) == [1]
        assert tools._as_list(1.0) == [1]
        assert tools._as_list("1") == ["1"]
        assert tools._as_list("1,2") == ["1", "2"]
        assert tools._as_list("1、2") == ["1", "2"]
        assert tools._as_list("[1,2]") == [1, 2]
        assert tools._as_list('["a","b"]') == ["a", "b"]

    def test_lists_and_none_pass_through(self):
        assert tools._as_list(["2025OS_Q1"]) == ["2025OS_Q1"]
        assert tools._as_list(None) is None
        assert tools._as_list([]) == []

    def test_explain_question_accepts_a_single_number(self):
        """实测那个坏输入：no="1"（字符串单值）必须能**进到工具里**

        注意判据不是"一定讲出题"：这个题库里题号 1 在选择题/综合题里都有，
        `explain_payload` 会**按设计**报"题号在不同题型里重复，需要指明是哪一道"。
        要守的是"schema 没把它挡在外面" —— 所以只要不是 pydantic 的
        ValidationError（那种连工具都没执行）就算过。
        """
        d = json.loads(tools.explain_question.invoke(
            {"paper_source": PAPER, "no": "1", "qids": QIDS[0]}))
        assert "error" not in d, d
        assert d.get("questions"), f"没讲出题来：{str(d)[:200]}"

    def test_explain_question_accepts_a_comma_string_no(self):
        d = json.loads(tools.explain_question.invoke(
            {"paper_source": PAPER, "no": "1,2", "qids": QIDS[0]}))
        assert "error" not in d, d
        assert d.get("questions"), str(d)[:200]

    def test_explain_question_reports_ambiguity_instead_of_crashing(self):
        """只给一个歧义题号、不给 qids 时：要给"请指明"的**人话**，而不是 ValidationError"""
        raw = tools.explain_question.invoke({"paper_source": PAPER, "no": 1})
        d = json.loads(raw)
        assert d.get("error") == "题号在不同题型里重复，需要指明是哪一道", d

    def test_grade_answers_accepts_a_string_qids(self, monkeypatch):
        """批改那边同类错法：qids 传成 JSON 字符串 / 逗号串

        判据是**传进 quiz_service 的到底是什么**，不是最终批改结果：
        qids 没被收敛成 list 时，`normalize_grade_args` 里的 `[q for q in (qids or [])]`
        会把字符串当**可迭代对象逐字符遍历**（`["2","0","2","5",…]`）。
        实测过：这种情况下 total/correct 可能**碰巧一样**（字符都不匹配 → 静默退化成
        "整卷按题型分段解析"，而 '1.C' 正好还是第 1 题）—— 所以拿批改结果当判据
        是个安慰剂（变异验证时抓到），必须直接看参数。
        """
        seen = []
        real = quiz_service.normalize_grade_args

        def spy(paper_source, qids=None, *a, **k):
            seen.append(qids)
            return real(paper_source, qids, *a, **k)

        monkeypatch.setattr(quiz_service, "normalize_grade_args", spy)
        for raw in (json.dumps([QIDS[0]], ensure_ascii=False), QIDS[0]):
            d = json.loads(tools.grade_answers.invoke(
                {"paper_source": PAPER, "user_answer_text": "1.C", "qids": raw}))
            assert "error" not in d, (raw, d)

        assert seen == [[QIDS[0]], [QIDS[0]]], (
            f"qids 没被收敛成 list 就传进 quiz_service 了（会被逐字符遍历）：{seen}")


class TestHallucinatedToolNameIsStripped:
    """缺陷 3：模型编造的工具名不能泄漏给学生"""

    LEAK = ('很抱歉，似乎在批改你的答案时出现了一些问题。\n\n你的答案：ACDBA\n\n'
            'get_paper_answers {"paper_source": "某卷", "answers": ["ACDBA"]}')

    def test_fake_tool_name_is_stripped(self):
        out = nodes.strip_tool_call_text(self.LEAK)
        assert "get_paper_answers" not in out
        assert "ACDBA" in out, "学生自己的内容不能被误删"

    def test_real_tool_names_still_stripped(self):
        for name in ("search_knowledge", "grade_answers", "read_section",
                     "list_papers", "explain_question"):
            t = f'{name} {{"q": "x"}} 下面是解释。'
            assert name not in nodes.strip_tool_call_text(t), name

    def test_normal_chinese_text_is_not_touched(self):
        """别误伤正常回复：中文里不会出现「蛇形标识符 + JSON 对象」"""
        for t in ("正确答案是 C，因为就绪态不能由阻塞态转化而来。",
                  "我们来看 process { 这个符号 } 的含义。",
                  "得分：5/5（100%）"):
            assert nodes.strip_tool_call_text(t) == t.strip(), t


class TestPlannerGetsACopyableCall:
    """缺陷 1 的根治：给 planner 一句可以直接照抄的调用

    实测「描述式提示」它不跟（`用上面的 qids 调 grade_answers` → 填进 paper_source）；
    给可照抄的一句才会照做 —— 这套办法在 read_section 的「继续讲」上已验证过。
    """

    def test_template_names_every_parameter(self):
        tpl = nodes._grade_call_template({"paper": PAPER, "qids": QIDS})
        assert f'paper_source="{PAPER}"' in tpl
        assert "user_answer_text=" in tpl
        assert f"qids={json.dumps(QIDS, ensure_ascii=False)}" in tpl

    def test_cursor_block_uses_the_template_and_warns(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: (
                                captured.update(prompt=msgs[0].content)
                                or nodes.AIMessage(content='{"need_tools": false, "steps": []}')))
        nodes.planner_node({
            "messages": [nodes.HumanMessage(content="ACDBA")],
            "plan": [], "tool_rounds": 0,
            "session": {"paper": PAPER, "type": "选择题", "group_index": 1,
                        "next_group_index": 2, "qids": QIDS}})
        p = captured["prompt"]
        assert "绝对不要把它填进 paper_source" in p
        assert json.dumps(QIDS, ensure_ascii=False) in p


class TestNoPhantomToolStep:
    """缺陷 4：AIMessage（带 tool_calls）不该被当成工具结果渲染"""

    def test_only_tool_results_are_kept(self):
        msgs = [AIMessage(content="", tool_calls=[
                    {"id": "c1", "name": "grade_answers", "args": {}}], id="a1"),
                ToolMessage(content='{"total": 5}', tool_call_id="c1",
                            name="grade_answers", id="t1")]
        kept = server._tool_result_messages(msgs)
        assert len(kept) == 1
        assert kept[0].name == "grade_answers"
        assert not any(getattr(m, "name", "") == "" for m in kept)

    def test_empty_and_none_are_safe(self):
        assert server._tool_result_messages([]) == []
        assert server._tool_result_messages(None) == []


class TestFabricatedImagesAreStripped:
    """实测：模型没取到课件、凭记忆讲课那一轮，自己编了题**和图片**：
    `![进程与线程测试题](https://example.com/process_thread_question.png)`
    提示词里早就写了「绝不允许编造图片链接」，照样编 —— 所以用代码兜。
    """

    def test_fake_image_is_dropped(self):
        t = ("接下面是小测：\n![进程与线程测试题]"
             "(https://example.com/process_thread_question.png)\n请作答。")
        cleaned, dropped = quiz_service.sanitize_images(t)
        assert "example.com" not in cleaned
        assert dropped == ["https://example.com/process_thread_question.png"]
        assert "请作答" in cleaned, "正文不能被误删"

    def test_real_question_image_is_kept(self):
        t = "1. 进程状态转换  ![题目](/qb_images/2025%E5%B9%B4..._Q1_p1.png)"
        cleaned, dropped = quiz_service.sanitize_images(t)
        assert cleaned == t
        assert dropped == []

    def test_static_images_are_kept(self):
        t = "![示意图](/static/img/x.png)"
        assert quiz_service.sanitize_images(t)[0] == t

    def test_plain_text_untouched(self):
        t = "正确答案是 C。没有图片。"
        assert quiz_service.sanitize_images(t) == (t, [])


class TestStudyContractReachesTheSynthesizerWithoutReadSection:
    """实测：只调了 list_course_sections、没调 read_section 时，
    会话里没有学习书签 → 「以课件为准 / 收尾 / 不要编题编图」全都不会注入 →
    模型凭记忆讲课还编题编图。所以 list_course_sections 也要报"在学哪门课"。
    """

    def test_list_course_sections_reports_the_course(self):
        out = tools.list_course_sections.invoke({"course": "操作系统"})
        assert "[study-course] course=操作系统" in out

    def test_extraction_sets_the_bookmark(self):
        out = tools.list_course_sections.invoke({"course": "计算机网络"})
        sess = nodes._extract_study_session(
            {}, [{"tool": "list_course_sections", "args": {}}],
            [ToolMessage(content=out, tool_call_id="p0", name="list_course_sections")])
        assert sess["study"]["course"] == "计算机网络"

    def test_contract_is_injected_from_that_bookmark_alone(self, monkeypatch):
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: (
                                captured.update(prompt=msgs[0].content)
                                or AIMessage(content="ok")))
        nodes.synthesizer_node({
            "messages": [nodes.HumanMessage(content="我要复习操作系统")],
            "plan": [], "tool_rounds": 0,
            "session": {"study": {"course": "操作系统", "mode": "复习"}}})
        p = captured["prompt"]
        assert "本节结束" in p, "收尾契约没注入"
        assert "不要编造" in p or "编造" in p or "quiz" in p.lower() or "小测" in p


class TestStudyRequestAlwaysReadsMaterial:
    """守卫：学生在听课/复习，但计划里没有取材料的动作 → 确定性补一步 read_section

    实测那一轮只调了 list_course_sections 就开讲，内容不在课件里。
    """

    def _plan(self, monkeypatch, steps):
        monkeypatch.setattr(
            nodes, "safe_llm_invoke",
            lambda fn, msgs, context=None: AIMessage(
                content=json.dumps({"need_tools": True, "steps": steps},
                                   ensure_ascii=False)))
        return nodes.planner_node({
            # 2026-10-18 起「我要复习操作系统」算**开场**（先说进度、让他选模式，不调工具），
            # 这里要测的是"讲了具体哪一章"的那条守卫，所以说清章节。
            "messages": [nodes.HumanMessage(content="我要复习操作系统死锁那一章")],
            "plan": [], "tool_rounds": 0, "session": {}})

    def test_forces_read_section_when_only_map_was_planned(self, monkeypatch):
        out = self._plan(monkeypatch, [{"tool": "list_course_sections",
                                        "args": {"course": "操作系统"}}])
        tools_ = [s["tool"] for s in out["plan"]]
        assert "read_section" in tools_, tools_
        rs = next(s for s in out["plan"] if s["tool"] == "read_section")
        assert rs["args"]["course"] == "操作系统"

    def test_continues_from_the_bookmark_instead_of_restarting(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "safe_llm_invoke",
            lambda fn, msgs, context=None: AIMessage(content=json.dumps(
                {"need_tools": True, "steps": [{"tool": "record_study_position",
                                                "args": {}}]})))
        out = nodes.planner_node({
            "messages": [nodes.HumanMessage(content="继续讲")],
            "plan": [], "tool_rounds": 0,
            "session": {"study": {"course": "操作系统",
                                  "last_source": "05-死锁.pdf", "next_offset": 15}}})
        rs = next(s for s in out["plan"] if s["tool"] == "read_section")
        assert rs["args"]["source"] == "05-死锁.pdf"
        assert rs["args"]["offset"] == 15

    def test_does_not_touch_non_study_questions(self, monkeypatch):
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: AIMessage(
                                content='{"need_tools": true, "steps": '
                                        '[{"tool": "list_papers", "args": {}}]}'))
        out = nodes.planner_node({
            "messages": [nodes.HumanMessage(content="我想刷题")],
            "plan": [], "tool_rounds": 0, "session": {}})
        assert [s["tool"] for s in out["plan"]] == ["list_papers"]

    def test_quiz_answer_turn_is_left_to_the_other_guard(self, monkeypatch):
        """交答案那一轮不能被"补 read_section"污染（那条守卫要求不调工具）"""
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: AIMessage(
                                content='{"need_tools": true, "steps": '
                                        '[{"tool": "grade_answers", "args": {}}]}'))
        out = nodes.planner_node({
            "messages": [nodes.HumanMessage(content="我答一下：1.B 2.D")],
            "plan": [], "tool_rounds": 0,
            "session": {"study": {"course": "操作系统", "section": "x"}}})
        assert out["plan"] == [], out["plan"]


class TestClosingMarkerIsGuaranteedByCode:
    """「本节结束」这条契约换过三种写法、量了三轮：
    提示中段 1/3 → 末尾+一句要求 2/3 → 给可照抄格式 3/4。
    而「出小测」一直 4/4。所以最后一级：缺了就由代码补，插在小测标题之前。
    """

    def test_inserts_before_the_quiz_heading(self):
        t = "## 进程的引入\n…讲解…\n\n### 本节小测\n1. 什么是进程？"
        out, added = quiz_service.ensure_closing(t)
        assert added
        assert out.index("本节结束") < out.index("本节小测"), "顺序必须是 讲解→本节结束→小测"

    def test_appends_when_there_is_no_quiz_heading(self):
        out, added = quiz_service.ensure_closing("## 进程的引入\n…讲解…")
        assert added and out.endswith("**本节结束**")

    def test_does_not_touch_text_that_already_has_it(self):
        t = "…讲解…\n\n**本节结束**\n\n【本节小测】\n1. x"
        assert quiz_service.ensure_closing(t) == (t, False)

    def test_empty_text_is_safe(self):
        assert quiz_service.ensure_closing("") == ("", False)

    def test_synthesizer_補上收尾标记(self, monkeypatch):
        """端到端：模型答了课但没写收尾 → synthesizer 输出里必须有"""
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: AIMessage(
                                content="进程是程序的一次执行过程。\n\n"
                                        "### 本节小测\n1. 什么是进程？"))
        out = nodes.synthesizer_node({
            # 「我要复习操作系统」2026-10-18 起算开场（先报进度、让他选），
            # 这里要测的是"讲了课但没写收尾标记"，所以说清章节。
            "messages": [nodes.HumanMessage(content="继续讲操作系统死锁那一章")],
            "plan": [], "tool_rounds": 0,
            "session": {"study": {"course": "操作系统", "mode": "复习"}}})
        text = out["messages"][0].content
        assert "本节结束" in text
        assert text.index("本节结束") < text.index("本节小测")

    def test_synthesizer_不在学习模式时不动它(self, monkeypatch):
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, msgs, context=None: AIMessage(
                                content="这是普通回答，没有收尾标记。"))
        out = nodes.synthesizer_node({
            "messages": [nodes.HumanMessage(content="你好")],
            "plan": [], "tool_rounds": 0, "session": {}})
        assert "本节结束" not in out["messages"][0].content


class TestSynthesizerHandlesToolErrors:
    """缺陷 3 的配套：工具报错时不许「假装重试」或编造调用"""

    def test_error_guidance_appears_only_on_error(self, monkeypatch):
        def run(tool_content):
            captured = {}
            monkeypatch.setattr(nodes, "safe_llm_invoke",
                                lambda fn, msgs, context=None: (
                                    captured.update(prompt=msgs[0].content)
                                    or AIMessage(content="ok")))
            nodes.synthesizer_node({
                "messages": [nodes.HumanMessage(content="ACDBA"),
                             ToolMessage(content=tool_content, tool_call_id="c1",
                                         name="grade_answers")],
                "plan": [], "tool_rounds": 0, "session": {}})
            return captured["prompt"]

        on_error = run('{"error": "没能确定试卷", "candidates": ["某卷"]}')
        assert "不要**说「我将再次尝试」" in on_error or "我将再次尝试" in on_error
        assert "编造工具名" in on_error

        ok = run('{"total": 5, "correct": 3}')
        assert "编造工具名" not in ok, "没出错时不该加这段约束"
