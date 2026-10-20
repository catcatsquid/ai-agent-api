"""两个运行面的**一致性** —— 这条测试的存在本身是有原因的

实测教训：我给 LangGraph 那条路（BOT）加了「学习/复习」三件套，**忘了给手写助教加**。
而"刷题工具各写一份、真实分叉"这个坑我**之前已经踩过一次**（所以才有了 quiz_service）。
同一个错误犯第二次，说明缺的不是记性，是**测试**。

这个文件只守两件事：

1. 两个运行面的**工具能力**必须对齐（差异只能是"有意保留"的那几个）；
2. 同名能力必须走**同一份实现**（用同一个 quiz_service 函数，而不是各写一份），
   以及那份共享契约（study_guide）的内容只允许有一处出处。

⚠️ 2026-10 起手写助教不再是产品路径上的入口（界面上撤掉了切换按钮），定位是
**冻结的对照实现**。所以"对齐"这件事的意义变了：不是"两边都要能给学生用"，
而是**共享的那一层不能被任何一边绕开**（绕开就会重新长出第二份实现）。

文件名的边界（拆分后）：

- 这份：两面一致性 + 共享契约内容
- `test_bot_study_flow.py`：BOT 那条路自己的学习流程行为
- `test_tutor_reference.py`：手写助教自己的行为与工具守卫
"""
import inspect
import json
import re
from pathlib import Path

import pytest

from course_agent import nodes, quiz_service, tools
from frozen import tutor_agent

ROOT = Path(__file__).resolve().parent.parent

# 有意保留的差异：这两条路的历史定位不同（BOT 有联网/统计，助教没有）
INTENTIONAL_BOT_ONLY = {"web_search", "get_course_info", "get_learning_progress",
                        "get_answers", "practice_wrong", "practice_by_topic"}
# ↑ `get_learning_progress`（1.8.0）：跨会话学习进度是 BOT 侧新加的能力；
#   `get_answers` / `practice_wrong` / `practice_by_topic`（1.8.1）：答题三形态与
#   按知识点抽题也是 BOT 侧新加的能力。手写助教是**冻结的对照实现**，
#   不追这些功能 —— 想加就得显式改这一行。
# 同一个能力在两个面上叫不同名字
ALIASES = {"search_knowledge": "search_knowledge_base"}

# 造假数据的课程结构（验证"两头走同一份实现"用；必须**按课程隔离**，
# 否则 section_index("操作系统") 也能查到计网课件，纠偏结果会失真）
_FAKE_PPTS_BY_COURSE = {
    "操作系统": [
        {"source": "05-死锁.pdf", "chapter": "死锁", "chunks": 20},
    ],
    "计算机网络": [
        {"source": "Chapter5-Transport Layer.pdf", "chapter": "Transport Layer", "chunks": 30},
        {"source": "Chapter4-Network Layer.pdf", "chapter": "Network Layer", "chunks": 25},
    ],
}


def _patch_section_index(monkeypatch):
    monkeypatch.setattr(
        quiz_service, "section_index",
        lambda course="": {"ppt": list(_FAKE_PPTS_BY_COURSE.get(course, []))})


def _bot_tools() -> set:
    return set(tools.tools_by_name)


def _tutor_tools() -> set:
    return set(tutor_agent.TOOL_HANDLERS)


def _bot_prompt() -> str:
    return nodes.SYNTHESIZER_PROMPT


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

    def test_the_docstring_tool_list_matches_reality(self):
        """`tools.py` 文件头那份"工具列表"必须与实际注册的工具**完全一致**

        它腐烂过一次：文件头写 9 个，加了「学习/复习三件套 + 自拟题」之后实际变成
        13 个，而文件头一直没改 —— 照着它数工具就会数错（我自己就被误导过一次）。

        断言比对的是**解析出来的条目名**，不是搜文本：正文里提到某个工具名不算数。
        """
        listed = set(re.findall(r"^\s+- ([a-z_][a-z0-9_]*):",
                                tools.__doc__ or "", re.M))
        actual = set(tools.tools_by_name)
        assert listed == actual, (
            "tools.py 文件头的工具列表与实际注册的对不上：\n"
            f"  只在文档里: {sorted(listed - actual)}\n"
            f"  只在代码里: {sorted(actual - listed)}\n"
            "  —— 加/删工具时请同步改文件头那份清单。")


class TestSharedImplementation:
    """同名能力必须走同一份实现 —— 否则就是"两份实现"，早晚分叉"""

    def test_read_section_delegates_to_quiz_service(self):
        for mod in (tools, tutor_agent):
            src = inspect.getsource(mod)
            assert "quiz_service.study_read" in src, f"{mod.__name__} 没走共享实现"
        # 关键：两头都不许自己再写一套"按章找课件 / 分段读"的逻辑
        for mod in (tools, tutor_agent):
            src = inspect.getsource(mod)
            for bad in ("find_chapter", "section_chunks"):
                # 两个名字都要查：`def _find_chapter` 这种**加下划线的变体**曾经
                # 躲过了这里（原来只判 "def find_chapter"，字符串不匹配），
                # 于是 tools.py 里一份不带中英对照表的旧副本长期没被发现，
                # BOT 侧对计网章节（传输层→Chapter5）纠偏失效。
                assert f"def {bad}" not in src and f"def _{bad}" not in src, \
                    f"{mod.__name__} 自己写了 {bad}（应走 quiz_service 的共享实现）"

    @pytest.mark.parametrize("chapter_in_course, expected", [
        ("传输层", "计算机网络"),
        ("网络层", "计算机网络"),
        ("死锁", "操作系统"),
    ])
    def test_chapter_in_course_field_is_corrected_identically(
            self, monkeypatch, chapter_in_course, expected):
        """模型把**章名**填进 course 时，两头必须纠成同一个课程名

        实测：计网课件是英文名（Chapter5-Transport Layer.pdf），学生说的是中文
        （传输层）。quiz_service.find_chapter 为此带了一张中英对照表，
        而 tools.py 里那份不带表的旧副本让 BOT 侧纠不过来 ——
        书签的 course 字段会留成"传输层"，于是两头的学习记录对不上。
        """
        _patch_section_index(monkeypatch)
        args = {"course": chapter_in_course, "section": "测试小节", "mode": "学习"}

        bot = json.loads(tools.record_study_position.invoke(dict(args)))["recorded"]
        tutor = tutor_agent.TOOL_HANDLERS["record_study_position"](
            dict(args), {})["recorded"]

        assert bot["course"] == tutor["course"], (
            f"两头纠偏结果不一致（输入 course={chapter_in_course!r}）："
            f"BOT={bot['course']!r}  助教={tutor['course']!r}")
        assert bot["course"] == expected
        assert bot["chapter"] == chapter_in_course, "章名应落到 chapter 字段"

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


class TestSharedCoreBehaviour:
    """共享核心（quiz_service）的行为契约 —— 两个面都建立在它上面

    这些不比较"两边是否一致"，而是钉住那份共享契约本身：它变了，两个面一起变。
    """

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
