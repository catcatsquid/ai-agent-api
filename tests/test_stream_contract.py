"""正文流的契约：把 sse_content_event 套在**真实 LangGraph 流**上端到端验证

为什么需要它：单测只能证明过滤函数的四层判据各自成立，证明不了
"这套判据套在真实流上，答案能过、污染能挡"。而真实流里有三件**实测确认**的事：

  ① messages 模式会把图内**每一次** LLM 调用都推出来 —— 包括节点内部跑的
     对话摘要（compress_history 就是这样）。它带着**同一个** langgraph_node，
     所以按节点做白名单挡不住；给那次调用加 stream=False 也没用
     （只是从 token 流变成一条整消息，消息照样在流里）。
  ② 它还会把节点**返回**的完整 AIMessage 再推一遍 —— synthesizer 返回的正是
     一条完整 AIMessage，放行就会让答案在气泡里出现**两遍**。
  ③ 不写答案的节点也会产出消息（planner 的 JSON 计划），绝不能进正文。

这个测试用假模型搭出「planner + synthesizer（内部含一次打了 tag 的摘要调用）」
的最小图，把每个 chunk 逐个喂给 sse_content_event，断言用户最终看到的正文里
只有答案、没有摘要、没有计划、且答案只出现一次。
"""
import json
from typing import Annotated, TypedDict

from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END, START, StateGraph, add_messages

from course_agent.memory import INTERNAL_LLM_TAG
from course_agent.server import PLAN_ANSWER_NODES, sse_content_event

ANSWER_TEXT = "本组 3 题：Q1、Q2、Q3（题面见图片）"
SUMMARY_TEXT = "【内部摘要】学生之前问过死锁"
PLAN_TEXT = '{"need_tools": true, "steps": [{"tool": "get_paper_questions"}]}'


class _State(TypedDict):
    messages: Annotated[list, add_messages]


def _fake(content):
    """每次调用新建 —— GenericFakeChatModel 的迭代器是一次性的"""
    return GenericFakeChatModel(messages=iter([AIMessage(content=content)]))


def _planner(state):
    out = _fake(PLAN_TEXT).invoke([HumanMessage(content="规划")])
    return {"messages": [AIMessage(content="计划：" + str(out.content))]}


def _synthesizer(state):
    # 模拟 compress_history：节点**内部**的一次 LLM 调用，打了内部 tag
    _fake(SUMMARY_TEXT).with_config(tags=[INTERNAL_LLM_TAG]).invoke(
        [HumanMessage(content="把老对话总结一下")])
    answer = _fake(ANSWER_TEXT).invoke([HumanMessage(content="回答")])
    return {"messages": [AIMessage(content=str(answer.content))]}


def _build_graph():
    g = StateGraph(_State)
    g.add_node("planner", _planner)
    g.add_node("synthesizer", _synthesizer)
    g.add_edge(START, "planner")
    g.add_edge("planner", "synthesizer")
    g.add_edge("synthesizer", END)
    return g.compile()


def _collect_answer_text(answer_nodes=PLAN_ANSWER_NODES) -> str:
    """把整条流喂给过滤函数，拼出"用户会看到什么" """
    app = _build_graph()
    seen = []
    for msg, meta in app.stream({"messages": [HumanMessage(content="出几道题")]},
                                stream_mode="messages"):
        line = sse_content_event(msg, meta, answer_nodes)
        if line:
            seen.append(json.loads(line[6:])["content"])
    return "".join(seen)


class TestAnswerStreamContract:
    def test_only_the_answer_reaches_the_user(self):
        text = _collect_answer_text()
        assert ANSWER_TEXT in text, f"答案没流出来：{text!r}"
        assert "内部摘要" not in text, "节点**内部**的 LLM 调用（摘要）流进了正文"
        assert "need_tools" not in text, "planner 的 JSON 计划流进了正文"

    def test_the_answer_is_not_duplicated(self):
        """节点返回的完整 AIMessage 会被再推一遍 —— 不能放行，否则答案出现两遍"""
        text = _collect_answer_text()
        n = text.count(ANSWER_TEXT)
        assert n == 1, f"答案出现了 {n} 遍：{text!r}"

    def test_without_the_source_whitelist_the_plan_leaks(self):
        """反向验证：不按图区分来源时，planner 的计划就会流出去

        这条是"来源白名单真的在起作用"的证据 —— 没有它，上面两条可能是碰巧过的。
        """
        text = _collect_answer_text(answer_nodes=None)
        assert "need_tools" in text, (
            "关掉来源白名单后计划内容竟然没进来 —— 说明这条反向验证失效了，"
            "上面那两条「答案干净」的断言也就失去了意义")
