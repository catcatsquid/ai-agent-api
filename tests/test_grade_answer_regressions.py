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
