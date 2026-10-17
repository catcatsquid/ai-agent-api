"""Plan 模式的 prompt 与工具集合必须一致（止血修复的守卫）

背景（实测）：Plan 模式在刷题流程上是坏的 ——
  · 出题时**伪造题目**并编出 `https://example.com/net1.jpg` 这种假图片链接；
  · 说"继续下一组"时丢掉会话游标，回复变成"我去挑一份卷子…"；
  · 说"我不会"时**不调 explain_question**，围绕知识点泛泛而谈；
  · 正文里还漏出 `get_paper_questions {"paper_source": ...}` 这种工具调用原文。

根因之一是 prompt 与工具集合**对不上**：
  · PLANNER_PROMPT 写"你有 8 个工具"，实际注册 9 个；
  · 把 `get_paper_questions` 描述成"获取指定试卷的**所有**题目"（实际一次只给一组，
    而且完全没提 group_index）；
  · `explain_question` 注册了却既不在 prompt 里、也不在 has_paper_tools 白名单里；
  · 没有任何一句禁止模型自己编题。

这些用例用"枚举 + 一致性"的方式守住，避免以后加工具时又忘改 prompt。
"""
import re

import pytest

from course_agent import nodes, tools


class TestPlannerPromptCoversEveryTool:
    def test_every_registered_tool_is_mentioned(self):
        """注册了的工具必须出现在 PLANNER_PROMPT 里

        实测踩过：`explain_question` 注册了但 prompt 没提、白名单也漏了，
        于是 Plan 模式下"学生说不会"这条路径基本不可能被规划出来。
        """
        prompt = nodes.PLANNER_PROMPT
        missing = [name for name in tools.tools_by_name if name not in prompt]
        assert not missing, f"这些工具注册了但 PLANNER_PROMPT 没提: {missing}"

    def test_tool_count_statement_matches_reality(self):
        """prompt 里写的工具个数必须和实际注册数一致"""
        m = re.search(r"你有\s*(\d+)\s*个工具", nodes.PLANNER_PROMPT)
        assert m, "PLANNER_PROMPT 里找不到工具数量声明"
        assert int(m.group(1)) == len(tools.tools_by_name), (
            f"prompt 说 {m.group(1)} 个，实际注册 {len(tools.tools_by_name)} 个")

    def test_quiz_tools_are_in_the_no_research_whitelist(self):
        """刷题类工具都必须在 has_paper_tools 白名单里

        漏掉的话 route_after_executor 会把它们的结果当成"检索不足"，多跑一次联网搜索。
        """
        src = open("course_agent/nodes.py", encoding="utf-8").read()
        m = re.search(r"has_paper_tools = any\((.*?)\)\n", src, re.S)
        assert m, "找不到 has_paper_tools"
        whitelist = set(re.findall(r'"(\w+)"', m.group(1)))
        # `tool` 是 lambda 里的字段名，不是工具名
        whitelist.discard("tool")
        expected = {"list_papers", "get_paper_questions", "grade_answers",
                    "explain_question", "get_practice_stats", "record_practice"}
        missing = expected - whitelist
        assert not missing, f"has_paper_tools 白名单漏了: {missing}"


class TestPlannerPromptDoesNotLieAboutTools:
    def test_get_paper_questions_is_described_as_one_group(self):
        """不能把 get_paper_questions 说成"获取所有题目"

        实测后果很严重：planner 以为一次能拿到整卷，结果只拿到一组，
        synthesizer 拿不到题就开始**自己编题**。
        """
        prompt = nodes.PLANNER_PROMPT
        assert "获取指定试卷的所有题目" not in prompt, "又出现了「所有题目」这个错误描述"
        # 必须说明"一次一组"
        assert re.search(r"一次只返回一组|不是整卷|5 题一组", prompt), \
            "没有说明 get_paper_questions 一次只返回一组"

    def test_group_index_is_documented(self):
        """必须告诉模型可以传 group_index，否则它只会反复要第 1 组"""
        assert "group_index" in nodes.PLANNER_PROMPT
        assert "next_group_index" in nodes.PLANNER_PROMPT

    def test_continue_intent_maps_to_get_paper_questions(self):
        """说"继续/下一组"要重新调 get_paper_questions，而不是换个卷子"""
        prompt = nodes.PLANNER_PROMPT
        assert "继续" in prompt
        assert re.search(r"继续[^\n]*get_paper_questions|get_paper_questions[^\n]*继续", prompt), \
            "没有把「继续」和 get_paper_questions 关联起来"

    def test_topic_practice_maps_to_practice_by_topic(self):
        """按知识点的练习必须指向 practice_by_topic（不是拉清单、也不是直接自拟）

        实测（评测用例 `quiz_a2_topic_practice`）：工具加了但提示词还写着
        prepare_custom_quiz，模型就照旧去自拟（2/2 次），新功能等于没接上。
        判据看**那条规则所在的段落**（规则会折行，不能只比一行）。
        """
        lines = nodes.PLANNER_PROMPT.splitlines()
        hits = [i for i, ln in enumerate(lines) if "practice_by_topic" in ln]
        assert hits, "按知识点的练习没指向 practice_by_topic"
        # 任一处出现都要满足：附近同时写了「不要」和 list_papers（规则会折行）
        blocks = ["\n".join(lines[i:i + 5]) for i in hits]
        assert any("list_papers" in b and "不要" in b for b in blocks), (
            "没在同一条规则里说明「按知识点练习不要拉试卷清单」：" + repr(blocks[0]))


class TestNoFabricationRules:
    @pytest.mark.parametrize("prompt_name", ["PLANNER_PROMPT", "SYNTHESIZER_PROMPT"])
    def test_forbids_inventing_questions(self, prompt_name):
        """两个 prompt 都要明确禁止自己编题"""
        prompt = getattr(nodes, prompt_name)
        assert re.search(r"不要编|不许编|绝不允许自己编|只能来自工具", prompt), \
            f"{prompt_name} 没有禁止编题的规则"

    def test_synthesizer_forbids_placeholder_image_urls(self):
        """必须点名禁止占位图片地址（实测 Plan 模式编出过 example.com）"""
        prompt = nodes.SYNTHESIZER_PROMPT
        assert "example.com" in prompt, "没有点名禁止占位图片地址"

    def test_synthesizer_forbids_leaking_tool_call_text(self):
        """工具调用本身不能被写进回复（实测漏出过 JSON 原文）"""
        assert re.search(r"不要把工具调用|工具调用本身", nodes.SYNTHESIZER_PROMPT)

    def test_synthesizer_points_to_explain_question(self):
        """讲题必须走 explain_question，而不是围绕知识点泛泛而谈"""
        assert "explain_question" in nodes.SYNTHESIZER_PROMPT


class TestRouteAfterExecutorWithExplain:
    def test_explain_question_does_not_trigger_web_search(self):
        """规划了 explain_question 时不该补搜"""
        state = {
            "messages": [],
            "plan": [{"tool": "explain_question", "args": {"paper_source": "X"}}],
            "tool_rounds": 1,
        }
        assert nodes.route_after_executor(state) == "synthesizer"

    def test_pure_knowledge_question_still_falls_back(self):
        """纯知识点问题且结果为空时，仍然要能补搜（别把原有能力关掉）"""
        state = {
            "messages": [],
            "plan": [{"tool": "search_knowledge", "args": {"query": "x"}}],
            "tool_rounds": 1,
        }
        assert nodes.route_after_executor(state) == "web_search_fallback"
