"""另一条会话加进来的两条守卫（2026-10-18）：补上单测，好一起提交

- **守卫 12「零步兜底」**：planner 说 `need_tools=false` 时不让它凭记忆答课程题 ——
  用一次 embedding 探测定性，命中就强制 `search_knowledge`。
  真实动机（真机录 DEMO 踩到）：问「死锁的四个必要条件是什么」竟然一个工具都没调，
  答案是对的、思维过程面板却是空的 —— 对 RAG 产品这是最难发现的失败形态。
- **`_is_quiz_menu_pick`**：学生只说「我想刷题」（菜单式短句）时，先列试卷目录，
  别把"刷题"读成"选第 3 条"直接跳到第 3 组。
"""
import pytest
from langchain_core.messages import AIMessage, HumanMessage

from course_agent import nodes


def _planner(monkeypatch, text, session=None, sim=None):
    """跑一遍 planner：模型总说 need_tools=false，检索探测按 sim 返回"""
    monkeypatch.setattr(nodes, "safe_llm_invoke",
                        lambda *a, **k: AIMessage(content='{"need_tools": false, "steps": []}'))

    class FakeKB:
        def search(self, query, top_k=1):
            return [] if sim is None else [{"sim": sim, "score": sim, "source": "x"}]

    import rag_tool
    monkeypatch.setattr(rag_tool, "get_knowledge_base", lambda: FakeKB())
    return nodes.planner_node({"messages": [HumanMessage(content=text)],
                               "plan": [], "tool_rounds": 0,
                               "session": session or {}})


class TestZeroStepFallback:
    def test_course_question_is_forced_to_retrieve(self, monkeypatch):
        out = _planner(monkeypatch, "死锁的四个必要条件是什么", sim=0.9)
        assert [s["tool"] for s in out["plan"]] == ["search_knowledge"], out["plan"]
        assert out["plan"][0]["args"]["query"] == "死锁的四个必要条件是什么"

    def test_offtopic_stays_zero_step(self, monkeypatch):
        """探测不到（聊天、库外问题）→ 照旧零步直答，别硬塞一次检索"""
        out = _planner(monkeypatch, "今天天气怎么样", sim=0.2)
        assert out["plan"] == [], out["plan"]

    def test_no_hits_at_all_stays_zero_step(self, monkeypatch):
        out = _planner(monkeypatch, "帮我写一首诗", sim=None)
        assert out["plan"] == [], out["plan"]

    def test_intentional_empty_is_not_pushed_back_to_retrieval(self, monkeypatch):
        """开场轮（守卫 4）刻意不调工具 —— 即使知识库命中也不许被零步兜底推翻"""
        out = _planner(monkeypatch, "我想学习", sim=0.99)
        assert out["plan"] == [], out["plan"]


class TestQuizMenuPick:
    @pytest.mark.parametrize("text", ["我想刷题", "刷题", "做题吧", "练题"])
    def test_menu_style_pick_lists_papers(self, text, monkeypatch):
        monkeypatch.setattr(nodes, "safe_llm_invoke",
                            lambda *a, **k: AIMessage(
                                content='{"need_tools": false, "steps": []}'))
        out = nodes.planner_node({"messages": [HumanMessage(content=text)],
                                  "plan": [], "tool_rounds": 0, "session": {}})
        assert [s["tool"] for s in out["plan"]] == ["list_papers"], out["plan"]

    @pytest.mark.parametrize("text", [
        "刷 2026卷", "继续刷", "再做第 2 组", "重做错题", "讲讲这道题", "刷题但是为什么选 B",
    ])
    def test_specified_target_is_not_a_menu_pick(self, text):
        """点了卷子/组号/继续/错题/讲题 —— 都不是"菜单式选模式"，不许被改成列目录"""
        assert nodes._is_quiz_menu_pick(text) is False, text
