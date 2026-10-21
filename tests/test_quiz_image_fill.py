"""出题轮的图片由**代码**补，模型不再重打那几百 token 的图片链接

2026-10-18 用户实测：出题一轮 33 秒，正文里 5 条 `![题目](/qb_images/…%E5%B9%B4…png)`
（百分号编码的中文文件名，一条 100+ 字符）占了输出的大头。

第一步（已做）：这一轮不逐字流式，别再刷屏。
第二步（本文件守的）：模型只写「题号 + 知识点」，图片由 `attach_quiz_images`
按题号补 —— 既省 token，又保证"有几道题就有几张图"（以前靠提示词求模型）。
"""
import json

import pytest
from langchain_core.messages import HumanMessage, ToolMessage

from course_agent import nodes, qbank, quiz_service

PLAN = {'paper': '某卷', 'type': '选择题', 'group_index': 1}


def _group_payload(n=3):
    """造一份和 get_paper_questions 同形的结果"""
    return {
        "paper": "某卷", "type": "选择题", "group_index": 1, "group_count": 2,
        "progress": "第 1/2 组（选择题 3 题）",
        "questions": [
            {"no": i, "type": "选择题", "topic": f"知识点{i}",
             "image_url": f"/qb_images/q{i}.png"}
            for i in range(1, n + 1)
        ],
        "qids": [f"Q{i}" for i in range(1, n + 1)],
        "has_more": True, "next_group_index": 2,
        "reply_text": "…", "render_example": "…",
    }


class TestExtractFromToolResults:
    def test_group_questions_use_their_own_numbers(self):
        got = quiz_service.quiz_questions_with_images([json.dumps(_group_payload())])
        assert [q["no"] for q in got] == [1, 2, 3]
        assert got[0]["image_url"] == "/qb_images/q1.png"
        assert got[0]["topic"] == "知识点1"

    def test_cross_paper_items_are_renumbered_in_order(self):
        """practice_by_topic 的 items 来自不同卷子，no 会撞 → 按展示顺序 1..N"""
        payload = {"source": "题库", "count": 2, "items": [
            {"paper": "A卷", "no": 5, "topic": "死锁", "image": "/qb_images/a5.png"},
            {"paper": "B卷", "no": 5, "topic": "银行家算法", "image": "/qb_images/b5.png"},
        ]}
        got = quiz_service.quiz_questions_with_images([json.dumps(payload)])
        assert [q["no"] for q in got] == [1, 2], got
        assert [q["image_url"] for q in got] == ["/qb_images/a5.png", "/qb_images/b5.png"]

    def test_junk_and_other_tools_are_ignored(self):
        got = quiz_service.quiz_questions_with_images(
            ["不是 JSON", json.dumps({"source": "题库", "count": 0}),
             json.dumps({"paper": "某卷", "group_index": 1})])
        assert got == []


class TestAttachImages:
    def test_images_are_attached_by_question_number(self):
        payload = _group_payload(3)
        answer = "第 1/2 组（选择题 3 题）：\n1. 知识点1\n2. 知识点2\n3. 知识点3\n\n请作答。"
        out = quiz_service.attach_quiz_images(
            answer, quiz_service.quiz_questions_with_images([json.dumps(payload)]))
        lines = [ln for ln in out.splitlines() if ln.startswith(("1.", "2.", "3."))]
        assert len(lines) == 3
        for i, ln in enumerate(lines, 1):
            assert f"![题目](/qb_images/q{i}.png)" in ln, ln

    def test_model_written_urls_are_replaced(self):
        """模型自己写了（往往写错的）长链接 → 一律换成题库里的那个"""
        got = quiz_service.quiz_questions_with_images([json.dumps(_group_payload(2))])
        answer = "1. 知识点1 ![题目](/qb_images/WRONG_long.png)\n2. 知识点2"
        out = quiz_service.attach_quiz_images(answer, got)
        assert "WRONG_long" not in out
        assert out.count("![题目](") == 2, out

    def test_missing_lines_are_filled(self):
        """模型漏了一道题 → 也要把图和题号补上（保证"几道题几张图"）"""
        got = quiz_service.quiz_questions_with_images([json.dumps(_group_payload(3))])
        answer = "第 1/2 组：\n1. 知识点1\n2. 知识点2\n\n请作答。"
        out = quiz_service.attach_quiz_images(answer, got)
        assert out.count("![题目](") == 3, out
        assert "3. 知识点3" in out, out

    def test_no_questions_leaves_text_alone(self):
        assert quiz_service.attach_quiz_images("随便一段话", []) == "随便一段话"

    @pytest.mark.parametrize("line", ["1. 进程管理", "**2、死锁**", "- 3) 银行家算法"])
    def test_common_list_styles_are_understood(self, line):
        got = [{"no": 1 if line.startswith("1") else (2 if "2" in line[:6] else 3),
                "topic": "x", "image_url": "/qb_images/x.png"}]
        out = quiz_service.attach_quiz_images(f"清单：\n{line}\n", got)
        assert "![题目](/qb_images/x.png)" in out, out


class TestToolHintsMatchTheNewDivision:
    """工具返回值本身也在"教"模型怎么回答 —— 它的措辞必须和新分工一致"""

    def test_get_paper_questions_does_not_ask_the_model_to_copy_urls(self):
        from course_agent import tools

        raw = tools.get_paper_questions.invoke(
            {"paper_source": "2025年秋操作系统期末试题（回忆版）", "group_index": 1})
        hint = json.loads(raw)["render_hint"]
        assert "不要写" in hint and "系统" in hint, hint
        assert "图片必须用 reply_text 里的" not in hint, f"又在教它抄 URL：{hint}"

    def test_redaction_removes_the_long_urls(self):
        payload = json.dumps(_group_payload(3), ensure_ascii=False)
        got = quiz_service.quiz_questions_with_images([payload])
        red = quiz_service.redact_quiz_image_urls(payload, got)
        assert "/qb_images/" not in red, red
        assert red.count("（系统按题号补图）") == 3, red

    def test_redaction_drops_the_whole_image_markdown(self):
        """只换 URL 的话模型会抄 `![题目]（图片由系统补）` 这种半成品，再被当成编造的图摘掉"""
        got = [{"no": 1, "topic": "进程管理", "image_url": "/qb_images/q1.png"}]
        src = "1. 进程管理  ![题目](/qb_images/q1.png)"
        red = quiz_service.redact_quiz_image_urls(src, got)
        assert "![题目]" not in red, red
        assert "1. 进程管理" in red


class TestSynthesizerFillsTheImages:
    def test_quiz_turn_gets_images_without_the_model_writing_urls(self, monkeypatch):
        payload = json.dumps(_group_payload(3), ensure_ascii=False)
        captured = {}

        def fake(fn, msgs, context=None, *a, **k):
            captured["p"] = msgs[0].content
            return nodes.AIMessage(
                content="第 1/2 组（选择题 3 题）：\n1. 知识点1\n2. 知识点2\n3. 知识点3\n\n"
                        "请作答，我来逐题批改。")

        monkeypatch.setattr(nodes, "safe_llm_invoke", fake)
        out = nodes.synthesizer_node({
            "messages": [nodes.HumanMessage(content="给我出第 1 组题"),
                         ToolMessage(content=payload, tool_call_id="c1",
                                     name="get_paper_questions")],
            "plan": [], "tool_rounds": 0, "session": {"paper": "某卷"}})
        text = out["messages"][0].content
        assert text.count("![题目](") == 3, text
        assert "/qb_images/q1.png" in text and "/qb_images/q3.png" in text
        # 提示词里明确"不许自己写链接"
        assert "不要写图片链接" in captured["p"], captured["p"][-400:]

    def test_knowledge_turn_is_untouched(self, monkeypatch):
        monkeypatch.setattr(
            nodes, "safe_llm_invoke",
            lambda fn, msgs, context=None, *a, **k: nodes.AIMessage(
                content="死锁的四个必要条件：互斥、持有并等待、不可剥夺、循环等待。"))
        out = nodes.synthesizer_node({
            "messages": [nodes.HumanMessage(content="死锁的四个必要条件是什么"),
                         ToolMessage(content=json.dumps({"hits": []}), tool_call_id="c1",
                                     name="search_knowledge")],
            "plan": [], "tool_rounds": 0, "session": {}})
        assert "![题目](" not in out["messages"][0].content
