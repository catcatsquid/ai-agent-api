"""手写助教（`tutor_agent.py`）的行为 —— **冻结的对照实现**

它的定位（`chat.html` 侧栏那段注释写得更完整）：2026-10 起界面上不再有它的入口，
只保留模块与测试，供"手写循环 vs LangGraph"对照学习用；新能力只加到 BOT，它不再演进。

这个文件原来埋在 `test_surface_parity.py` 里，而那个名字说的是"两个运行面对齐"——
这些用例其实**只跟助教自己有关**（工具守卫、状态注入、会话书签），拆出来让名字对上内容。

⚠️ 这些测试全部**直接调 `tutor_agent` 的函数**，不经过 HTTP。
所以 1.1.0 误删 `POST /tutor/stream` 时，它们一条都没红 ——
端点那一层现在由 `test_route_surface.py` 的路由清单守。

两个运行面的**一致性**（工具对齐、共用 quiz_service、提示词契约）在
`test_surface_parity.py`；BOT 那条路的学习流程在 `test_bot_study_flow.py`。
"""
from course_agent import tutor_agent


class TestTutorStudyFlow:
    """助教在学习/复习流程里的行为（含"共用实现"那一侧）"""

    def test_section_map_is_internal_not_a_student_facing_reply(self):
        """地图是**内部工作资料**，不能当成"成品文本"发给学生

        实测教训：我原来给它配了一份排好版的 `reply_text`，而模型被教过
        "工具给了成品文本就直接转发" —— 于是学生说「我要复习操作系统」，
        收到的是一张目录。现在它不给 reply_text，并且明确要求接着开讲。
        """
        r = tutor_agent.TOOL_HANDLERS["list_course_sections"]({"course": "操作系统"}, {})
        assert "reply_text" not in r, "别再把目录当学生可见的成品文本"
        assert "material_map" in r
        assert "不要把它作为回复内容" in r["_guide"]
        assert "立刻调 read_section" in r["_guide"]

    def test_tutor_read_section_returns_same_content_shape(self):
        r = tutor_agent.TOOL_HANDLERS["read_section"](
            {"course": "操作系统", "chapter": "死锁", "limit": 2}, {})
        assert "[study-state]" in r["text"], "机器可读行不能只有 BOT 那边有"
        assert "05-死锁.pdf" in r["text"]
        assert "本节结束" in r["text"]

    def test_tutor_bookmark_writes_into_session(self):
        """手写循环有 session，可以直接写（Plan 图那边是从工具结果里抽）"""
        sess = {}
        tutor_agent.TOOL_HANDLERS["record_study_position"](
            {"course": "操作系统", "section": "死锁的概念", "mode": "复习"}, sess)
        assert sess["study"]["course"] == "操作系统"
        assert sess["study"]["mode"] == "复习"
        assert sess["study_mode"] == "复习"

    def test_tutor_read_section_records_position_automatically(self):
        sess = {}
        tutor_agent.TOOL_HANDLERS["read_section"](
            {"course": "操作系统", "chapter": "死锁", "limit": 15, "mode": "复习"}, sess)
        study = sess["study"]
        assert study["course"] == "操作系统"
        assert study["last_source"] == "05-死锁.pdf"
        assert study["next_offset"] == 15
        assert study["mode"] == "复习"

    def test_tutor_corrects_chapter_in_course_field(self):
        sess = {}
        tutor_agent.TOOL_HANDLERS["record_study_position"](
            {"course": "死锁", "section": "x"}, sess)
        assert sess["study"]["course"] == "操作系统"
        assert sess["study"]["chapter"] == "死锁"


class TestTutorToolGuards:
    """工具层守卫 —— 全部来自"代入学生视角跑一轮"的真实发现"""

    def test_quiz_answers_are_guarded_during_study(self):
        """讲课/复习中学生交小测答案 → 不能走题库批改

        实测模型习惯性调 grade_answers，然后反问"你在做哪份卷子"，
        把学习流程硬拽回刷题流程。提示词它不跟，所以在工具入口拦。
        """
        r = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"user_answer_text": "1.B 2.D"},
            {"study": {"course": "操作系统", "section": "进程的引入"}})
        assert "不走题库批改" in r.get("error", "")
        assert "由你自己批" in r.get("_guide", "")
        assert "哪份卷子" in r.get("_guide", "")   # 明确禁止反问卷名

    def test_normal_quiz_grading_still_works(self):
        """守卫不能误伤正常刷题批改"""
        r = tutor_agent.TOOL_HANDLERS["grade_answers"](
            {"paper_source": "2026卷", "user_answer_text": "ACDBE"}, {})
        assert "不走题库批改" not in str(r.get("error", ""))
        assert "accuracy" in r, "正常批改的结果字段不见了"

    def test_study_state_is_injected_into_the_system_prompt(self):
        """"小测自己批""没听懂就重取原文"必须在**交答案那一轮**就可见 ——
        那轮学生不会调 read_section，工具返回里的契约到不了模型手里。"""
        p = tutor_agent._system_prompt_for({
            "study": {"mode": "复习", "course": "操作系统", "chapter": "死锁",
                      "section": "死锁的四个必要条件",
                      "last_source": "05-死锁.pdf", "last_read": "第 1-15 块",
                      "next_offset": 15},
            "current_paper": None})
        assert "当前学习位置" in p
        assert "由你自己批" in p
        assert "offset=15" in p
        assert "没听懂" in p
        assert "不串课" in p

    def test_quiz_tools_are_hidden_while_teaching(self):
        """讲课时**不给**题库工具 —— 最强的一种纠偏：不给它，它就没法调错

        实测三级升级：① 在 system prompt 里写"不要用 grade_answers" → 不跟；
        ② 在工具入口加守卫、返回"这是小测，你自己批" → **还是不跟**，照样问学生卷名；
        ③ 干脆在讲课状态（无当前试卷）下不提供这些工具。
        """
        teaching = {"study": {"course": "操作系统", "section": "进程的引入"}}
        names = {t["function"]["name"]
                 for t in tutor_agent._tools_for(teaching)}
        for t in ("grade_answers", "explain_question", "get_paper_questions"):
            assert t not in names, f"讲课中还给了 {t}"
        # 但学习三件套和检索要有
        for t in ("read_section", "list_course_sections", "record_study_position",
                  "search_knowledge_base"):
            assert t in names, f"讲课中缺了 {t}"

    def test_quiz_tools_come_back_for_real_quiz(self):
        """刷题时（有当前试卷）题库工具必须都在"""
        names = {t["function"]["name"] for t in tutor_agent._tools_for(
            {"current_paper": "2026春操作系统期末试题（回忆版）"})}
        for t in ("grade_answers", "explain_question", "get_paper_questions"):
            assert t in names, f"刷题时缺了 {t}"

    def test_quiz_tools_available_when_not_studying(self):
        names = {t["function"]["name"] for t in tutor_agent._tools_for({})}
        assert "grade_answers" in names

    def test_no_injection_when_not_studying(self):
        p = tutor_agent._system_prompt_for({"current_paper": None})
        assert p == tutor_agent.SYSTEM_PROMPT, "没在学习就不该注入状态"
