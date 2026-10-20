"""回答收尾清理必须**两条路都做**，计划缺 `need_tools` 不许当成"不用工具"

## 这一版修的两个真问题（都是 1.7.2 的评测/离线打桩抓到的）

1. **ReAct 那条路的回答什么都不清**。工具调用原文的清理、编造图片链接的清理、
   学习收尾标记的补全，原来只写在 `synthesizer_node` 里（plan 模式那条路）。
   于是 ReAct 模式下 `explain_question {"question_index": 1}` 这种**工具调用原文
   会直接进学生气泡**（评测用例 `react_mode_streams` 抓到过一次）。
   修法：提成共用函数 `nodes.clean_answer_text`，两条路都调。

2. **planner 回 `{}`（或漏掉 `need_tools`）时，计划为空 → 路由直接进 synthesizer
   → 一个工具都不调就凭记忆作答**。prompt 里写死了"必须返回 need_tools"，
   所以"缺失"是不合规，不是"不用工具"；只有**显式** `need_tools: false` 才是。

## 为什么"两条路都调"要用源码断言守

内容级判据只能证明 `clean_answer_text` 本身对；证明不了**哪条路在调它** ——
而当初的 bug 恰恰是"函数是对的，只是 ReAct 那条路没调"。要走真集成就得打真模型，
于是这里用 AST 断言"这两个节点函数体里都出现 `clean_answer_text`"。
它与内容级判据互补：一个守"清理逻辑对"，一个守"两条路都接了线"。
"""
import ast
import inspect
import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from course_agent import nodes


# ============================================================
# ① 清理逻辑本身
# ============================================================

class TestCleanAnswerText:
    def test_tool_call_text_is_stripped(self):
        t = '好的，我来讲解。\nexplain_question\n{"question_index": 1}\n下面是解析：'
        out = nodes.clean_answer_text(t)
        assert "explain_question" not in out, out
        assert "我来讲解" in out and "下面是解析" in out, "正文不能被误删"

    def test_same_line_tool_call_is_stripped(self):
        out = nodes.clean_answer_text('稍等。grade_answers {"qids": ["a"]} 结果如下。')
        assert "grade_answers" not in out

    def test_fabricated_image_is_stripped(self):
        t = ("小测：\n![题](https://example.com/process_thread_question.png)\n请作答。")
        out = nodes.clean_answer_text(t)
        assert "example.com" not in out
        assert "请作答" in out

    def test_real_question_image_is_kept(self):
        t = "1. 进程状态转换  ![题目](/qb_images/2025%E5%B9%B4..._Q1_p1.png)"
        assert nodes.clean_answer_text(t) == t

    def test_plain_text_is_untouched(self):
        t = "正确答案是 C，因为就绪态不能由阻塞态转化而来。"
        assert nodes.clean_answer_text(t) == t

    def test_empty_answer_does_not_crash(self):
        assert nodes.clean_answer_text("") == ""
        assert nodes.clean_answer_text(None) == ""

    def test_study_session_gets_the_closing_marker(self):
        state = {"session": {"study": {"course": "操作系统"}}}
        out = nodes.clean_answer_text("## 进程的引入\n……讲解……", state)
        assert "本节结束" in out, "学习模式下收尾标记必须由代码保证"

    def test_no_closing_marker_outside_study_mode(self):
        out = nodes.clean_answer_text("……讲解……", {"session": {}})
        assert "本节结束" not in out

    def test_skip_closing_when_the_student_is_asking_for_answers(self):
        """学生要答案的那一轮不是"讲完一节"，不许补收尾标记"""
        state = {"session": {"study": {"course": "操作系统"}}}
        out = nodes.clean_answer_text("1. A 2. B 3. C", state, skip_closing=True)
        assert "本节结束" not in out
        assert out == "1. A 2. B 3. C"

    def test_keyword_argument_style_call_is_stripped(self):
        """2026-10-18 真会话里抓到的第三种泄漏形态（关键字参数）

        学生气泡里出现了：`请稍等片刻……read_section(course="操作系统",
        section="02-1-进程与线程之进程")` —— 前两条正则只认 JSON，漏了它。
        """
        t = ('请稍等片刻，我马上为您准备好课程内容。'
             'read_section(course="操作系统", section="02-1-进程与线程之进程")')
        out = nodes.clean_answer_text(t)
        assert "read_section" not in out, out
        assert "请稍等片刻" in out, "正文不能被误删"

    def test_a_real_function_name_in_the_lesson_is_kept(self):
        """空括号不许当工具调用 —— `pthread_cancel()` 是讲义正文"""
        t = "线程可以用 pthread_cancel() 取消，也可以用 pthread_exit() 退出。"
        assert nodes.clean_answer_text(t) == t


# ============================================================
# ①b 原始协议帧（模型把工具调用协议当正文写出来）
# ============================================================

# 这段是 eval 报告 1.7.3 里**真实抓到**的正文（只截掉了后面的正文）
REAL_LEAK = (
    '好的，我们来复习一下死锁的四个必要条件。首先，我会调用一下知识库来获取更详细的信息。\n'
    '\n'
    '\n'
    '{"course": "操作系统", "query": "死锁的四个必要条件"}\n'
    '```{"index":0,"finish_reason":"tool_calls","delta":{"role":"assistant",'
    '"content":null,"reasoning_content":null,"tool_calls":[{"id":"call_1","index":0,'
    '"type":"function","function":{"name":"search_knowledge","arguments":'
    '"{\\"course\\": \\"操作系统\\"}"}}]}}'
)


class TestProtocolFramesAreStripped:
    def test_the_real_leak_is_cleaned(self):
        out = nodes.strip_protocol_frames(REAL_LEAK)
        assert "finish_reason" not in out, out
        assert '"delta"' not in out, out
        assert '"tool_calls"' not in out, out
        assert "我会调用一下知识库" in out, "学生看的那句话不能被误删"

    def test_it_also_works_through_clean_answer_text(self):
        out = nodes.clean_answer_text(REAL_LEAK)
        assert "finish_reason" not in out
        assert "四个必要条件" in out

    def test_normal_json_in_the_answer_is_kept(self):
        """正常正文里出现 JSON（讲解代码/配置）不许被误删 —— 判据是协议字段，不是花括号"""
        for text in ('这个配置长这样：{"course": "操作系统", "top_k": 3} 就够了。',
                     '返回结构：{"total": 5, "correct": 4}',
                     '看这段代码：\n```python\n{"a": 1}\n```\n结束。'):
            assert nodes.strip_protocol_frames(text) == text.strip(), text

    def test_a_single_protocol_field_is_not_enough(self):
        """判据是"≥2 个协议字段"：只有一个字段的 JSON **不许**被当协议帧摘掉

        模型讲课时完全可能举一个单字段的例子（`{"finish_reason": "stop"}`）——
        但那不是协议帧。这条与"整个帧必须摘掉"配对，才能把阈值钉在 2 上
        （变异验证时抓到的：只测"脏的要摘"，阈值降到 1 也照样绿）。
        """
        for t in ('举例来说，返回里会有 {"finish_reason": "stop"} 这一项。',
                  '参数里带上 {"tool_calls": null} 表示没有调用。'):
            assert nodes.strip_protocol_frames(t) == t, t

    def test_unbalanced_brace_does_not_eat_the_rest(self):
        """配不成对的花括号不许把后面整段吞掉（模型还在流时会出现）"""
        t = '看这个：{"course": "操作系统"，后面还有正文没有闭合'
        assert nodes.strip_protocol_frames(t) == t

    def test_brace_inside_a_string_does_not_break_pairing(self):
        """字符串里的 } 不能让配对错位（否则会吞掉正常文本）"""
        t = '示例：{"note": "这里有个 } 符号", "n": 1} 后面照常。'
        assert nodes.strip_protocol_frames(t) == t

    def test_prompt_forbids_writing_the_protocol_as_text(self):
        """兜底之外还要在提示里明确禁止（实测模型会"打算调用"而不调用）

        判据：**同一条要求里**既要点名三个协议字段、又要有禁止词 ——
        这才是这条禁令的意图。不钉具体措辞（这个项目在 1.7.1 栽过"守字词"）。
        """
        prompt = nodes.REACT_PROMPT
        ban = [ln for ln in prompt.splitlines()
               if all(f in ln for f in ("finish_reason", "delta", "tool_calls"))
               and ("不许" in ln or "绝不要" in ln or "不要" in ln)]
        assert ban, ("提示里没有「不许把这些协议字段写进正文」这条要求 —— "
                     "模型会照着自己编的协议往下写")
        assert "直接调用" in prompt, "没告诉模型「想调就直接调」"


# ============================================================
# ② 两条路都必须接上清理（源码断言 —— 见模块 docstring 的理由）
# ============================================================

def _calls(func, name: str) -> bool:
    """函数体里有没有出现对 `name` 的调用（AST，比字符串包含可靠）"""
    tree = ast.parse(_clean_source(func))
    return any(
        isinstance(node, ast.Call)
        and (getattr(node.func, "id", None) == name
             or getattr(node.func, "attr", None) == name)
        for node in ast.walk(tree)
    )


def _clean_source(func) -> str:
    """函数源码（嵌套函数带缩进，`ast.parse` 前必须 dedent）"""
    import textwrap

    return textwrap.dedent(inspect.getsource(func))


class TestBothAnswerPathsCleanTheirOutput:
    @pytest.mark.parametrize("node_name", ["synthesizer_node", "llm_node"])
    def test_node_calls_the_shared_cleaner(self, node_name):
        node = getattr(nodes, node_name)
        assert _calls(node, "clean_answer_text"), (
            f"{node_name} 没有调用 clean_answer_text —— 这条路的回答会带着"
            f"工具调用原文/编造图片链接直接进学生气泡（这正是 1.7.3 修的问题）")

    def test_the_probe_itself_can_fail(self):
        """反向验证：AST 断言不是"永远为真"（否则上面两条毫无意义）"""

        def innocent(state):
            return {"messages": []}

        assert not _calls(innocent, "clean_answer_text")
        assert _calls(nodes.clean_answer_text, "sanitize_images")

    def test_react_answer_is_actually_cleaned_end_to_end(self, monkeypatch):
        """再补一条真跑的：llm_node 返回的回答必须已经被清理（打桩掉模型）"""
        leaked = '我来讲解。\nexplain_question\n{"question_index": 1}\n下面说明：'
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda *a, **k: AIMessage(content=leaked))
        out = nodes.llm_node({"messages": [HumanMessage(content="讲讲第 1 题")],
                              "tool_rounds": 0, "session": {}})
        answer = out["messages"][-1].content
        assert "explain_question" not in answer, answer
        assert "下面说明" in answer


# ============================================================
# ③ 计划里没有工具、又没明说"不用工具" → 不许直接作答
# ============================================================

class TestEmptyPlanWithoutNeedToolsStillRetrieves:
    def _plan(self, monkeypatch, raw, user="讲讲死锁的四个必要条件", session=None):
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda *a, **k: AIMessage(content=raw))
        return nodes.planner_node({
            "messages": [HumanMessage(content=user)],
            "plan": [], "tool_rounds": 0, "session": session or {}})

    def test_empty_object_falls_back_to_retrieval(self, monkeypatch):
        """实测的坏形态：模型回 `{}` → 计划为空 → 一个工具都不调就凭记忆答"""
        out = self._plan(monkeypatch, "{}")
        plan = out["plan"]
        assert [s["tool"] for s in plan] == ["search_knowledge"], plan
        assert plan[0]["args"]["query"] == "讲讲死锁的四个必要条件"

    def test_missing_need_tools_is_not_a_reason_to_skip_tools(self, monkeypatch):
        """`{"steps": []}` 同样没有明说不用工具 → 也要检索"""
        out = self._plan(monkeypatch, '{"steps": []}')
        assert [s["tool"] for s in out["plan"]] == ["search_knowledge"]

    def test_explicit_need_tools_false_still_skips_tools(self, monkeypatch):
        """**显式**说不用工具（闲聊）时不许硬塞检索 —— 这是合法路径，别改坏"""
        out = self._plan(monkeypatch, '{"need_tools": false, "steps": []}',
                         user="你好呀")
        assert out["plan"] == []

    def test_quiz_answer_round_is_still_left_alone(self, monkeypatch):
        """学生交答案那一轮由守卫 2 处理成"不调工具"，不能被回退计划顶掉"""
        out = self._plan(monkeypatch, "{}", user="1.B 2.D 3.A",
                         session={"study": {"course": "操作系统"}})
        assert out["plan"] == [], out["plan"]

    def test_route_follows_the_plan(self, monkeypatch):
        """把"计划为空 → 直接进 synthesizer（不检索）"这条后果钉住"""
        empty = self._plan(monkeypatch, "{}", user="你好呀")
        explicit = self._plan(monkeypatch, '{"need_tools": false, "steps": []}',
                              user="你好呀")
        assert nodes.route_after_plan({"plan": empty["plan"]}) == "executor"
        assert nodes.route_after_plan({"plan": explicit["plan"]}) == "synthesizer"

    def test_good_plan_is_untouched(self, monkeypatch):
        raw = json.dumps({"need_tools": True, "steps": [
            {"tool": "read_section", "args": {"course": "操作系统"}}]},
            ensure_ascii=False)
        out = self._plan(monkeypatch, raw)
        assert [s["tool"] for s in out["plan"]] == ["read_section"]
