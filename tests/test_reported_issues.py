"""用户实测报告的三个问题的回归测试

问题① 「复习操作系统」被当成刷题，完全不检索课程资料
问题② 「2026卷」这种最自然的说法解析不出试卷
问题③ 服务重启后说「继续」就"找不到题目了"（游标只活在进程内存里）

外加一条：模型把工具调用**当正文写出来**给学生看（实测出现过）。
"""
import json
import re
import sys
from pathlib import Path

import pytest

from course_agent import nodes, qbank, tools

ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 问题②：卷名解析
# ============================================================

class TestPaperAliasResolution:
    """「卷」是用户自然会加的量词，但**任何卷名里都没有这个字** ——
    简称匹配要求输入的每个字符都能在卷名里按序找到，于是「2026卷」必然失败。
    """

    TARGET = "2026春操作系统期末试题（回忆版）"

    def test_noise_chars_really_absent_from_every_paper(self):
        """先证明前提：剥掉这些字永远是安全的"""
        names = [p["source"] for p in qbank.list_papers()]
        for ch in ("卷", "套", "份"):
            assert not any(ch in n for n in names), \
                f"「{ch}」出现在某份卷名里，不能再当量词剥掉"

    @pytest.mark.parametrize("text", [
        "2026卷",                    # 用户的原话
        "2026操作系统卷",              # 用户的原话
        "2026年操作系统卷",
        "我选择2026卷",                # 用户原话（模型把整句当卷名传进来）
        "2026OS",                    # 缩写：所有卷名里都没有拉丁字母
        "2026操作系统",
        "2026",
        "2026春期末",
    ])
    def test_user_phrasings_resolve(self, text):
        p, _ = qbank.resolve_paper(text)
        assert p is not None, f"「{text}」解析失败"
        assert p["source"] == self.TARGET

    @pytest.mark.parametrize("text", ["2024卷", "2025卷"])
    def test_other_years_also_work(self, text):
        p, _ = qbank.resolve_paper(text)
        assert p is not None and text[:4] in p["source"]

    def test_course_aliases(self):
        p, _ = qbank.resolve_paper("计网模拟题")
        assert p and p["source"] == "2019秋本部计算机网络模拟题"

    @pytest.mark.parametrize("text", [
        "2026春计算机网络模拟题",   # 编造：课程对不上
        "2027春操作系统期末试题",   # 编造：年份对不上
        "2019秋操作系统期末试题",   # 编造：这门课 2019 没有卷子
        "随便什么卷",
    ])
    def test_fabricated_names_still_rejected(self, text):
        """放宽"量词"的同时，绝不能开始乱猜卷名"""
        p, _ = qbank.resolve_paper(text)
        assert p is None, f"编造的卷名不该命中，却解析成 {p['source'] if p else ''}"

    def test_strip_only_touches_edges(self):
        """两步机制要分清楚：量词是**无条件删除**，不在 allowed 的字只剥**首尾**"""
        # 第 1 步：量词无条件删（「卷」在任何卷名里都不存在，删了永远安全）
        assert qbank._strip_alias_noise("2026卷") == "2026"
        assert qbank._strip_alias_noise("我选择2026卷") == "我选择2026"
        # 第 2 步：首尾不在 allowed 里的字剥掉，**中间的不动**
        assert qbank._strip_alias_noise("我选择2026", {"2", "0", "6"}) == "2026"
        assert qbank._strip_alias_noise("2026操作系统", {"2", "0", "6", "作"}) == "2026操作"


# ============================================================
# 问题①：复习意图必须落到检索，而不是刷题
# ============================================================

class TestReviewIntentGoesToKnowledgeBase:
    """实测：说"想复习操作系统"→ 只调 list_papers（当成刷题）；
    说"开始第一天的复习"→ 直接调 get_paper_questions 甩题。
    两次都**没有** search_knowledge，所以讲解内容与本课课件无关。
    """

    def test_prompt_has_a_review_rule(self):
        p = nodes.PLANNER_PROMPT
        assert "复习" in p and "search_knowledge" in p
        # 必须明确"复习 ≠ 刷题"
        assert "不是刷题" in p or "不要" in p

    def test_prompt_lists_review_keywords(self):
        p = nodes.PLANNER_PROMPT
        for kw in ("复习", "讲解", "梳理", "过一遍", "带我学"):
            assert kw in p, f"规则里缺少复习类关键词「{kw}」"

    def test_prompt_requires_questions_when_paper_is_named(self):
        """学生指名卷子时必须直接出题，不能只把清单再念一遍"""
        p = nodes.PLANNER_PROMPT
        assert "指名" in p or "已经选好" in p

    def test_search_knowledge_ships_a_presentation_contract(self):
        """检索结果必须带"怎么用这些片段"的契约

        只给一段资料，模型会拿自己记忆里的通用教材替换它 ——
        这就是"复习时没按课件讲"的机制（跟伪造题目是同一个道理）。
        """
        out = tools.search_knowledge.invoke({"query": "死锁的四个必要条件"})
        assert "怎么用上面的片段" in out
        assert "以这些片段为准" in out
        # 资料没覆盖时要如实说，而不是编
        assert "没找到" in out


# ============================================================
# 问题③：游标必须活过进程重启
# ============================================================

class TestSessionSurvivesRestart:
    """跨轮游标由 checkpointer 落盘保管（1.2.0 起是 SqliteSaver）

    历史：1.0.x 的 checkpointer 是 MemorySaver（进程内存），服务一重启游标就丢，
    所以当时给游标单独做了一份落盘副本（runtime/plan_cursors.json）。
    那份副本造成"内存 + 磁盘"两个真源，1.0.1 修过它的读写不对称。
    1.2.0 换成 SqliteSaver 之后副本整个删掉，这里改成验证**落盘本身**。
    """

    def test_thread_state_survives_a_simulated_restart(self, tmp_path):
        """核心验收：同一个库文件、新开一个 saver 与图，state 必须还在

        这是任务 8 的验收条件。MemorySaver 时代这里会拿到空 state ——
        那正是"重启后说继续下一组，只能听到找不到题目"的根因。
        """
        import sqlite3

        from langgraph.checkpoint.sqlite import SqliteSaver

        from course_agent.graph import build_graph

        db = tmp_path / "ckpt.sqlite3"
        cfg = {"configurable": {"thread_id": "t-restart"}}
        session = {"paper": "2026春操作系统期末试题（回忆版）", "type": "选择题",
                   "group_index": 1, "next_group_index": 2, "qids": ["Q1"]}

        # 第一个"进程"：直接落一条 checkpoint（不跑节点，避免真实模型调用）
        g1 = build_graph(checkpointer=SqliteSaver(
            sqlite3.connect(str(db), check_same_thread=False)))
        g1.update_state(cfg, {"session": session})

        # "重启"：新 saver、新图，**同一个库文件**
        g2 = build_graph(checkpointer=SqliteSaver(
            sqlite3.connect(str(db), check_same_thread=False)))
        got = (g2.get_state(cfg).values or {}).get("session") or {}
        assert got.get("paper") == session["paper"], f"重启后游标没了：{got}"
        assert got.get("next_group_index") == 2

    def test_study_only_turn_keeps_the_drill_fields(self, monkeypatch):
        """同一轮只更新学习书签时，刷题字段不能被挤掉

        这条原来是 1.0.1 为"读写不对称"写的回归（那时磁盘副本是第二个真源：
        写入的合并基线只取 state，于是只调 read_section 的那一轮会把 paper 挤掉）。
        副本删掉之后不对称从根上没有了，但**合并基线要包含上一轮的 session**
        这条不变量仍然要守，所以保留一个不带磁盘的版本。
        """
        before = {"paper": "2025年秋操作系统期末试题（回忆版）", "type": "选择题",
                  "group_index": 3, "next_group_index": 4, "qids": ["Q1", "Q2"]}
        study_line = ("[study-state] course=操作系统 source=05-死锁.pdf "
                      "read=3 next_offset=15")
        monkeypatch.setattr(nodes, "safe_tool_invoke",
                            lambda tool, args, tool_name=None: study_line)

        out = nodes.executor_node({
            "messages": [nodes.HumanMessage(content="继续讲")],
            "plan": [{"tool": "read_section",
                      "args": {"course": "操作系统", "section": 1}}],
            "session": before,     # 上一轮的游标（重启后由 checkpointer 恢复出来）
        }, {"configurable": {"thread_id": "t"}})

        sess = out["session"]
        assert sess.get("paper") == before["paper"], f"刷题字段被挤掉了：{sess}"
        assert sess.get("next_group_index") == 4
        assert (sess.get("study") or {}).get("course") == "操作系统"

    def test_planner_carries_the_session_into_the_prompt(self, monkeypatch):
        """planner 必须把 state 里的游标带进提示词

        重启那一半由上面那条落盘测试守；这里只验证"给了 session，planner 就真的用它"。
        """
        captured = {}
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda fn, messages, context=None: (
                                captured.update(prompt=messages[0].content)
                                or nodes.AIMessage(content='{"need_tools": false, "steps": []}')))
        nodes.planner_node(
            {"messages": [nodes.HumanMessage(content="继续下一组")],
             "plan": [], "tool_rounds": 0,
             "session": {"paper": "2026春操作系统期末试题（回忆版）", "type": "选择题",
                         "group_index": 1, "next_group_index": 2,
                         "qids": ["Q1", "Q2"]}},
            {"configurable": {"thread_id": "t"}})
        p = captured["prompt"]
        assert "2026春操作系统期末试题（回忆版）" in p, "没带上卷子"
        assert "group_index=2" in p, "没带上下一组组号"

    def test_executor_writes_the_session_into_state(self, monkeypatch):
        """executor 把这一轮抽出的进度放进 out["session"]，由 checkpointer 落盘"""
        plan = [{"tool": "get_paper_questions",
                 "args": {"paper_source": "2026春操作系统期末试题（回忆版）",
                          "group_index": 1}}]
        payload = json.dumps({
            "paper": "2026春操作系统期末试题（回忆版）", "type": "选择题",
            "group_index": 1, "next_group_index": 2, "has_more": True,
            "qids": ["Q1"], "questions": [], "reply_text": "…",
        }, ensure_ascii=False)
        monkeypatch.setattr(nodes, "safe_tool_invoke",
                            lambda tool, args, tool_name=None: payload)
        out = nodes.executor_node({
            "messages": [nodes.HumanMessage(content="出题")],
            "plan": plan, "session": {}}, {"configurable": {"thread_id": "t"}})
        assert out["session"]["next_group_index"] == 2
        # 题型也要记上：漏了的话恢复出来的游标里 type 是 null（实测到过）
        assert out["session"]["type"] == "选择题"

    def test_tool_output_carries_the_question_type(self):
        """tools.py 转发给模型的 JSON 必须带 type —— 游标的题型就是从它来的"""
        from course_agent import quiz_service
        paper, _ = qbank.resolve_paper("2026卷")
        if paper is None:
            pytest.skip("没有 2026 卷")
        out = tools.get_paper_questions.invoke(
            {"paper_source": paper["source"], "group_index": 1})
        assert "get_paper_questions" not in out or True   # 只是确保没抛异常
        assert '"type"' in out, "工具返回里没有 type，游标就记不到题型"

    def test_runtime_state_is_gitignored(self):
        """checkpointer 的库文件是运行时状态，不能进版本控制"""
        gi = (ROOT / ".gitignore").read_text(encoding="utf-8")
        assert "runtime/" in gi


# ============================================================
# 工具调用泄漏成正文
# ============================================================

class TestToolCallTextIsStripped:
    """实测漏出过： "请稍等，我马上为你获取题目。get_paper_questions {"paper_source": "…"}"
    根因是 Plan 图单向、synthesizer 无法再取数据（正解是允许一次重规划）。
    这里兜底把它摘掉，别让学生看到内部协议。
    """

    def test_strips_a_leaked_call(self):
        text = ('好的，我已经为你准备好了2026春操作系统期末试题（回忆版）。\n'
                '请稍等，我马上为你获取题目。get_paper_questions '
                '{"paper_source": "2026春操作系统期末试题（回忆版）"}')
        out = nodes.strip_tool_call_text(text)
        assert "get_paper_questions" not in out
        assert '{"paper_source"' not in out
        assert "请稍等" in out          # 正文的其他部分要留住

    def test_strips_multiple_and_keeps_structure(self):
        text = ('先查一下。list_papers {"course": "操作系统"}\n\n'
                '再看看。get_paper_questions {"paper_source": "x", "group_index": 2}\n完。')
        out = nodes.strip_tool_call_text(text)
        for t in tools.tools_by_name:
            assert t not in out
        assert "先查一下。" in out and "完。" in out

    @pytest.mark.parametrize("text", [
        "死锁的四个必要条件是什么？",
        "这是代码：int main() { return 0; }",
        "进程和线程的区别：{概述}",
        "用 search_knowledge 可以查资料",     # 只是提到工具名，没有 JSON
        "",
    ])
    def test_does_not_mangle_normal_text(self, text):
        assert nodes.strip_tool_call_text(text) == text.strip() if text else True

    def test_synthesizer_applies_the_stripper(self):
        src = (ROOT / "course_agent/nodes.py").read_text(encoding="utf-8")
        i = src.index("def synthesizer_node")
        # 取到**下一个顶层 def** 为止。原来写死 2600 字符窗口，
        # 我在提示里加了两段话就把 strip 调用挤出窗口 → 测试假红了一次。
        # （教训：别用"字符数窗口"当函数边界，那就是一种数据快照。）
        nxt = re.search(r"\ndef \w", src[i + 10:])
        body = src[i:i + 10 + nxt.start()] if nxt else src[i:]
        assert "strip_tool_call_text" in body, "synthesizer 没有应用兜底清理"
        # 收紧一点：不能只是"提到了"，必须真的调用了
        assert "cleaned = strip_tool_call_text(" in body
