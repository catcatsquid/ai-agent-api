"""两个运行面的能力必须对齐 —— 这条测试的存在本身是有原因的

实测教训：我给 LangGraph 那条路（BOT）加了「学习/复习」三件套，
**忘了给手写助教加**。而"刷题工具各写一份、真实分叉"这个坑我**之前已经踩过一次**
（所以才有了 quiz_service）。同一个错误犯第二次，说明缺的不是记性，是**测试**。

这个文件守两件事：
  1. 两个运行面的**工具能力**必须对齐（差异只能是"有意保留"的那几个）；
  2. 同名能力必须走**同一份实现**（用同一个 quiz_service 函数，而不是各写一份）。
"""
import inspect
import re
from pathlib import Path

import pytest

from course_agent import quiz_service, tools, tutor_agent

ROOT = Path(__file__).resolve().parent.parent

# 有意保留的差异：这两条路的历史定位不同（BOT 有联网/统计，助教没有）
INTENTIONAL_BOT_ONLY = {"web_search", "get_course_info"}
# 同一个能力在两个面上叫不同名字
ALIASES = {"search_knowledge": "search_knowledge_base"}


def _bot_tools() -> set:
    return set(tools.tools_by_name)


def _tutor_tools() -> set:
    return set(tutor_agent.TOOL_HANDLERS)


class TestToolParity:
    def test_study_tools_exist_on_both_surfaces(self):
        """学习/复习三件套必须两个面都有 —— 这就是漏掉过的那次"""
        for t in ("list_course_sections", "read_section", "record_study_position"):
            assert t in _bot_tools(), f"BOT 缺 {t}"
            assert t in _tutor_tools(), f"手写助教缺 {t}（曾经真的缺过）"

    def test_quiz_tools_exist_on_both_surfaces(self):
        for t in ("list_papers", "get_paper_questions", "grade_answers",
                  "prepare_custom_quiz", "explain_question"):
            assert t in _bot_tools() and t in _tutor_tools(), f"{t} 没对齐"

    def test_only_intentional_differences_remain(self):
        """除了明确有意保留的（以及同一能力的别名），两边不该再有差异"""
        only_bot = _bot_tools() - _tutor_tools() - set(ALIASES)
        only_tutor = _tutor_tools() - _bot_tools() - set(ALIASES.values())
        assert only_bot <= INTENTIONAL_BOT_ONLY, f"BOT 多出的工具没登记: {only_bot}"
        assert only_tutor <= set(ALIASES.values()), f"助教多出的工具没登记: {only_tutor}"

    def test_tutor_schemas_match_handlers(self):
        """助教是手写 schema 的 —— schema 与 handler 必须一一对应

        （LangGraph 那边是 @tool 自动生成，天然一致；手写这边最容易漏。）
        """
        names = {t["function"]["name"] for t in tutor_agent.TOOLS}
        assert names == _tutor_tools(), (
            f"schema 有但没 handler: {names - _tutor_tools()}；"
            f"handler 有但没 schema: {_tutor_tools() - names}")


class TestSharedImplementation:
    """同名能力必须走同一份实现 —— 否则就是"两份实现"，早晚分叉"""

    def test_read_section_delegates_to_quiz_service(self):
        for mod in (tools, tutor_agent):
            src = inspect.getsource(mod)
            assert "quiz_service.study_read" in src, f"{mod.__name__} 没走共享实现"
        # 关键：两头都不许自己再写一套"按章找课件 / 分段读"的逻辑
        for mod in (tools, tutor_agent):
            src = inspect.getsource(mod)
            assert "def find_chapter" not in src, f"{mod.__name__} 自己写了章节查找"
            assert "def section_chunks" not in src, f"{mod.__name__} 自己写了分段读取"

    def test_study_contract_comes_from_the_shared_layer(self):
        """学习/复习的呈现契约（含模式差异）只能是 quiz_service 那一份"""
        for mode in ("学习", "复习"):
            g = quiz_service.study_guide(mode)
            assert f"模式：{mode}" in g
        assert "快速过" in quiz_service.study_guide("复习")
        assert "细讲" in quiz_service.study_guide("学习")
        # 题量差异要体现出来，但**不能写死**成一个数
        assert "5~8" in quiz_service.study_guide("复习")
        assert "2~3" in quiz_service.study_guide("学习")
        assert "由你" in quiz_service.study_guide("学习")

    def test_tutor_and_bot_agree_on_section_index(self):
        idx = quiz_service.section_index("操作系统")
        names = [s["source"] for s in idx["ppt"]]
        assert "05-死锁.pdf" in names
        # 助教侧返回的也是同一份（字段名不叫 reply_text 了 —— 见下一条测试）
        r = tutor_agent.TOOL_HANDLERS["list_course_sections"]({"course": "操作系统"}, {})
        assert r["sections"]["ppt"] == idx["ppt"]
        assert "05-死锁.pdf" in r["material_map"]

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


class TestBothPromptsCarryThePrinciples:
    """教学原则的**细节只有一份出处**（quiz_service.study_guide），
    两个 prompt 只留指针 —— 助教那条甚至有测试卡着提示词长度上限
    （prompt 堆太长会让模型变没主见）。

    所以这里守的是：契约完整 + 两边 prompt 都指向契约。
    """

    KEYS = ("本节结束", "小测", "错在哪", "重讲", "全称", "不串课")

    def test_contract_carries_every_principle(self):
        for mode in ("学习", "复习"):
            g = quiz_service.study_guide(mode)
            for k in self.KEYS:
                assert k in g, f"{mode} 契约缺「{k}」"
            assert "作业" in g

    def test_both_prompts_point_at_the_contract(self):
        for who, prompt in (("BOT", _bot_prompt()),
                            ("助教", tutor_agent.SYSTEM_PROMPT)):
            assert "read_section" in prompt, f"{who} prompt 没指引去用 read_section"
            assert "契约" in prompt, f"{who} prompt 没说「照契约走」"

    def test_no_prompt_duplicates_the_numbers(self):
        """细节不许在 prompt 里再抄一遍（两处写同样的数字，改一处必漏另一处）"""
        for who, prompt in (("BOT", _bot_prompt()),
                            ("助教", tutor_agent.SYSTEM_PROMPT)):
            assert "5~8" not in prompt, f"{who} prompt 抄了题量数字"
            assert "2~3" not in prompt, f"{who} prompt 抄了题量数字"

    def test_mode_difference_lives_in_the_contract(self):
        assert "快速过" in quiz_service.study_guide("复习")
        assert "细讲" in quiz_service.study_guide("学习")
        assert "5~8" in quiz_service.study_guide("复习")
        assert "2~3" in quiz_service.study_guide("学习")
        assert "由你" in quiz_service.study_guide("学习")


def _bot_prompt() -> str:
    from course_agent import nodes
    return nodes.SYNTHESIZER_PROMPT


class TestStudentSessionFindings:
    """这四条全部来自"代入学生视角跑一轮"的真实发现"""

    def test_chinese_chapter_maps_to_english_ppt(self):
        """说"复习传输层"必须落到 Chapter5-Transport Layer.pdf

        实测：中文名对不上英文课件名 → 模型自己挑，挑成了网络层，
        讲的内容跟学生要的完全不是一回事。
        """
        cases = [("传输层", "Chapter5-Transport Layer.pdf"),
                 ("网络层", "Chapter4-Network Layer.pdf"),
                 ("应用层", "Chapter6-Application Layer.pdf"),
                 ("数据链路层", "Chapter3-Data Link Layer & LAN.pdf"),
                 ("物理层", "Chapter2-Physical Layer.pdf"),
                 ("无线", "Chapter7-Wireless & Mobile Networks.pdf")]
        for cn, src in cases:
            hit = quiz_service.find_chapter("计算机网络", cn)
            assert hit and hit["source"] == src, f"「{cn}」没落到 {src}"

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

    def test_section_means_chapter_number_for_chapter_named_courses(self):
        """计网课件是 ChapterN- 命名的，学生说"传输层"时模型会想成 Chapter5。

        实测：它把 5 传进 section，而按**位置**数第 5 份是 Chapter4（课件从
        Chapter0 起，差一位）→ 读了网络层、却凭记忆讲 TCP/UDP。
        所以这类课程里 section 优先按**章号**解释。
        """
        r = quiz_service.study_read(course="计算机网络", section=5, limit=2)
        assert "Chapter5-Transport Layer.pdf" in r["text"], "section=5 应该落到 Chapter5"
        assert "按章号定位" in r["text"]
        r4 = quiz_service.study_read(course="计算机网络", section=4, limit=2)
        assert "Chapter4-Network Layer.pdf" in r4["text"]

    def test_section_still_means_position_when_no_chapter_numbering(self):
        """操作系统课件是 02-1-/03- 命名，没有 ChapterN —— 仍按位置数"""
        r = quiz_service.study_read(course="操作系统", section=1, limit=2)
        assert "02-1-进程与线程之进程.pdf" in r["text"]
        assert "本课课件 1/" in r["text"], "这类课程应该仍按位置编号"

    def test_chapter_lookup_beats_section(self):
        """给了章名就以章名为准（学生按章提问时最准）"""
        r = quiz_service.study_read(course="计算机网络", chapter="传输层",
                                    section=4, limit=2)
        assert "Chapter5-Transport Layer.pdf" in r["text"]
        assert "按章定位" in r["text"]

    def test_guide_warns_against_guessing_the_section_number(self):
        """实测：学生说"传输层"，它自己数成第 5 份 → 读了网络层那章。
        所以两边都要明确要求"按章提问时必须传 chapter，不要数编号"。"""
        bot_guide = tutor_agent.TOOL_HANDLERS["list_course_sections"](
            {"course": "计算机网络"}, {})["_guide"]
        assert "chapter" in bot_guide
        assert "不要自己数编号" in bot_guide
        assert "不要自己数编号" in tutor_agent._system_prompt_for(
            {"study": {"course": "计算机网络", "section": "x"}})

    def test_no_injection_when_not_studying(self):
        p = tutor_agent._system_prompt_for({"current_paper": None})
        assert p == tutor_agent.SYSTEM_PROMPT, "没在学习就不该注入状态"
