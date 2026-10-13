"""服务层安全与协议修复的回归测试

覆盖：
  · /kb/add、/kb/add-directory 的路径包含性校验（任意文件入库链）
  · /kb/browse 仍然只允许 data/
  · 工具产出不再被当作正文流式播给前端（泄题通道）
  · 进度事件（"思维过程"面板）只给语义标签，不回显工具返回内容
"""
import json

import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage, AIMessageChunk, ToolMessage

from course_agent import quiz_service
from course_agent.memory import INTERNAL_LLM_TAG
from course_agent.server import resolve_within_data, sse_content_event, DATA_ROOT, app
from fastapi.testclient import TestClient

client = TestClient(app, raise_server_exceptions=False)


class TestResolveWithinData:
    """路径必须落在 data/ 之下"""

    def test_accepts_relative_path_inside_data(self):
        p = resolve_within_data("data/question_bank/images", must_exist=False)
        assert str(p).startswith(str(DATA_ROOT))

    def test_accepts_absolute_path_inside_data(self):
        inside = DATA_ROOT / "question_bank"
        assert resolve_within_data(str(inside), must_exist=False) == inside.resolve()

    @pytest.mark.parametrize("bad", [
        "README.md",
        ".env",
        "../../etc/passwd",
        "data/../README.md",
        "C:/Windows/win.ini",
        "course_agent/server.py",
    ])
    def test_rejects_paths_outside_data(self, bad):
        with pytest.raises(ValueError):
            resolve_within_data(bad, must_exist=False)

    def test_rejects_empty(self):
        with pytest.raises(ValueError):
            resolve_within_data("")

    def test_must_exist_flag(self):
        with pytest.raises(ValueError):
            resolve_within_data("data/__definitely_not_here__", must_exist=True)
        # 不校验存在性时应当通过
        resolve_within_data("data/__definitely_not_here__", must_exist=False)


class TestKbPathValidation:
    """带合法凭据时，越界路径也必须被 403 拒绝

    攻击链（修复前）：
      ① /auth/login 不校验密码 → 任意匿名者拿到 JWT
      ② /kb/add?file_path=<任意文件> → 被 _classify_doc 兜底成 "notes" 收进库
      ③ /kb/search（当时无鉴权）→ 返回 chunk 全文 = 文件内容外泄
    """

    @pytest.fixture(autouse=True)
    def _auth(self):
        with patch("course_agent.auth.API_KEY", "test-key"):
            yield {"X-API-Key": "test-key"}
    @pytest.mark.parametrize("path", ["README.md", "../../etc/passwd", "C:/Windows/win.ini"])
    def test_kb_add_rejects_outside_data(self, _auth, path):
        resp = client.post("/kb/add", params={"file_path": path}, headers=_auth)
        assert resp.status_code == 200
        body = resp.json()
        assert body["code"] == 403, f"{path} 竟然没被拒绝: {body}"
        assert "data/" in body["msg"]

    def test_kb_add_accepts_inside_data_but_missing(self, _auth):
        """越界是 403，越界之外的"路径不存在"是 404 —— 两者不能混成一个码"""
        resp = client.post("/kb/add",
                           params={"file_path": "data/__not_here__.md"}, headers=_auth)
        assert resp.json()["code"] == 404

    @pytest.mark.parametrize("path", ["C:/", "../..", "course_agent"])
    def test_kb_add_directory_rejects_outside_data(self, _auth, path):
        resp = client.post("/kb/add-directory", params={"dir_path": path}, headers=_auth)
        assert resp.json()["code"] == 403, f"{path} 竟然没被拒绝"

    def test_kb_add_directory_accepts_inside_data(self, _auth):
        resp = client.post("/kb/add-directory",
                           params={"dir_path": "data/__not_a_dir__"}, headers=_auth)
        # 通过路径校验后因为目录不存在而返回 404（而不是越界 403）
        assert resp.json()["code"] == 404

    def test_kb_browse_still_limited_to_data(self, _auth):
        resp = client.get("/kb/browse", params={"path": "../../course_agent"}, headers=_auth)
        assert resp.json()["code"] == 403

    def test_kb_add_requires_auth_when_enforcement_is_on(self):
        """开启强制鉴权时，没有凭据不能在越界校验之前就泄露任何信息"""
        with patch("course_agent.middleware.AUTH_ENFORCE", True), \
             patch("course_agent.auth.AUTH_ENFORCE", True):
            resp = client.post("/kb/add", params={"file_path": "README.md"})
        assert resp.status_code == 401


class TestPathValidationSurvivesAuthOff:
    """关掉鉴权后，路径校验必须依然拦住越界文件

    这是"本地默认免登录"这个决定能成立的前提：真正危险的那条链
    （任意文件入库 → /kb/search 外泄）靠的是处理函数内部的 resolve_within_data，
    与鉴权解耦。如果哪天有人把校验挪到依赖里、或依赖鉴权兜底，这些用例会红。
    """

    @pytest.fixture(autouse=True)
    def _no_auth(self):
        with patch("course_agent.middleware.AUTH_ENFORCE", False), \
             patch("course_agent.auth.AUTH_ENFORCE", False):
            yield

    @pytest.mark.parametrize("path", [
        "README.md", ".env", "../../etc/passwd", "C:/Windows/win.ini",
        "course_agent/server.py", "data/../README.md",
    ])
    def test_add_still_rejects_outside_data_without_auth(self, path):
        resp = client.post("/kb/add", params={"file_path": path})
        assert resp.status_code == 200
        assert resp.json()["code"] == 403, f"{path} 在无鉴权模式下竟然通过了: {resp.json()}"

    @pytest.mark.parametrize("path", ["C:/", "../..", "course_agent"])
    def test_add_directory_still_rejects_outside_data_without_auth(self, path):
        resp = client.post("/kb/add-directory", params={"dir_path": path})
        assert resp.json()["code"] == 403, f"{path} 在无鉴权模式下竟然通过了"

    def test_browse_still_limited_without_auth(self):
        resp = client.get("/kb/browse", params={"path": "../../course_agent"})
        assert resp.json()["code"] == 403

    def test_endpoints_are_reachable_without_credentials(self):
        """反过来也要确认：无鉴权模式下接口确实可用（不是被 401 挡住的假通过）"""
        resp = client.get("/kb/browse", params={"path": "data"})
        assert resp.status_code == 200
        assert resp.json()["code"] == 0


class TestSseContentEvent:
    """正文事件（content）的四层判据

    契约在 1.1.0 变过：**判据从"类型黑名单"改成"来源白名单 + 只放行 LLM 分片"**。
    旧契约是"ToolMessage / 带 tool_calls / 空 content 之外一律放行"——默认放行，
    而且挡不住两件实测确认的事（见 tests/test_stream_contract.py）：
      · 节点**内部**的 LLM 调用（对话摘要）会带着同一个节点名流出来；
      · 节点**返回**的完整 AIMessage 也会被推一遍，放行就让答案在气泡里出现两遍。

    这里只测过滤函数的四个层次；"这套判据套在真实 LangGraph 流上是否有效"
    由 tests/test_stream_contract.py 用假模型图端到端验证。
    """

    def _chunk(self, content, **meta):
        return AIMessageChunk(content=content), meta

    def test_chunk_from_the_answer_node_passes(self):
        msg, meta = self._chunk("这是答案", langgraph_node="synthesizer")
        line = sse_content_event(msg, meta, {"synthesizer"})
        assert line is not None
        assert line.startswith("data: ")
        assert '"type": "content"' in line
        assert "这是答案" in line

    @pytest.mark.parametrize("node", ["planner", "executor", "tools", "web_search_fallback"])
    def test_chunk_from_a_non_answer_node_is_dropped(self, node):
        """① 来源白名单：计划 JSON / 工具轮都不该进正文"""
        msg, meta = self._chunk('{"steps": []}', langgraph_node=node)
        assert sse_content_event(msg, meta, {"synthesizer"}) is None

    def test_unknown_source_fails_closed(self):
        """① 的默认值是"拦住"—— 图里新增节点不会被自动播出去"""
        msg, meta = self._chunk("将来新加的东西", langgraph_node="some_new_node")
        assert sse_content_event(msg, meta, {"synthesizer"}) is None

    def test_internal_tagged_call_is_dropped(self):
        """② 内部工具型调用（对话摘要）即便来自允许的节点也不播"""
        msg, meta = self._chunk("【摘要】用户问过死锁", langgraph_node="synthesizer",
                                tags=[INTERNAL_LLM_TAG])
        assert sse_content_event(msg, meta, {"synthesizer"}) is None

    def test_full_message_from_the_node_is_dropped(self):
        """③ 节点返回的完整 AIMessage 不放行 —— 否则答案会被发两遍"""
        msg = AIMessage(content="这是答案")
        assert sse_content_event(msg, {"langgraph_node": "synthesizer"},
                                 {"synthesizer"}) is None

    def test_tool_message_is_dropped(self):
        msg = ToolMessage(content='{"answer": "D", "reference_points": ["因为..."]}',
                          tool_call_id="c1", name="grade_answers")
        assert sse_content_event(msg, {"langgraph_node": "synthesizer"},
                                 {"synthesizer"}) is None

    def test_empty_content_is_dropped(self):
        """④ ReAct 的工具决策轮通常 content 为空"""
        msg, meta = self._chunk("", langgraph_node="llm")
        assert sse_content_event(msg, meta, {"llm"}) is None

    def test_react_graph_uses_its_own_answer_node(self):
        """同一个函数要按图区分：llm 在 react 图里是答案节点、在 plan 图里不是"""
        msg = AIMessageChunk(content="答案")
        assert sse_content_event(msg, {"langgraph_node": "llm"}, {"llm"}) is not None
        assert sse_content_event(msg, {"langgraph_node": "llm"}, {"synthesizer"}) is None

    def test_without_answer_nodes_only_the_later_layers_apply(self):
        """不传 answer_nodes 时跳过①，②③④ 仍然生效（给不按图区分的调用方）"""
        assert sse_content_event(AIMessageChunk(content="x")) is not None
        assert sse_content_event(AIMessageChunk(content=""),
                                 {"langgraph_node": "anything"}) is None
        assert sse_content_event(AIMessageChunk(content="x"),
                                 {"tags": [INTERNAL_LLM_TAG]}) is None


class TestProgressSummaryIsALabelNotThePayload:
    """"思维过程"面板的进度摘要只能是**语义标签**，不能回显工具返回内容

    三条产品约束同时要求这一点：
      1. 课件是从 PDF 抽出来的，文字很乱，不适合直接给学生看；
      2. 出题阶段的题面与答案要藏到学生答完再核对；
      3. 批改 / 讲题的结果里带参考答案。

    原来两个 BOT 端点都是 `content[:120]` 原样截断 —— 上面三类都会出现在面板里。
    手写助教侧本来就是按工具名给标签，现在两处统一走 quiz_service.tool_summary。
    """

    _SECRETS = ("题干原文", "参考答案是D", "题面原文")

    @pytest.mark.parametrize("tool,payload", [
        ("explain_question",
         {"questions": [{"text": "题干原文", "answer": "参考答案是D"}]}),
        ("grade_answers",
         {"total": 1, "accuracy": "0/1", "correct_answer": "参考答案是D"}),
        ("get_paper_questions",
         {"paper": "2025OS", "progress": "第1组",
          "questions": [{"text": "题面原文", "answer": "D"}]}),
    ])
    def test_label_never_echoes_the_payload(self, tool, payload):
        label = quiz_service.tool_summary(tool, json.dumps(payload, ensure_ascii=False))
        for secret in self._SECRETS:
            assert secret not in label, f"{tool} 的摘要回显了内容：{label!r}"

    def test_accepts_both_dict_and_json_string(self):
        """助教侧传 dict，BOT 侧传 ToolMessage.content（JSON 字符串）—— 两种都要支持"""
        assert quiz_service.tool_summary("list_papers", {"total_papers": 3}) == "3 份试卷"
        assert quiz_service.tool_summary("list_papers", '{"total_papers": 3}') == "3 份试卷"

    def test_plain_text_result_degrades_to_a_fixed_label(self):
        """拿不到结构化计数时给固定标签，也**不许**截断原文"""
        text = "[05-死锁.pdf · 相关度=0.9]\n死锁的四个必要条件……"
        label = quiz_service.tool_summary("search_knowledge", text)
        assert "死锁的四个必要条件" not in label
        assert label == "检索完成"

    def test_unknown_tool_degrades_to_a_plain_label(self):
        """不认识的工具退化成"完成"——不能为了信息量退回成回显内容"""
        assert quiz_service.tool_summary("some_new_tool", {"secret": "别播我"}) == "完成"

    def test_progress_events_go_through_the_shared_label(self):
        """接线守卫：进度摘要只允许走 quiz_service.tool_summary

        这条原来枚举了 `content[:120]` / `content[:100]` 两个**字面量** —— 那是同一类
        弱写法：换成 `content[:150]` 就漏了。（1.1.0 那次守卫审计里我自己审出来的，
        就在这一条上，这次一并修掉。）

        第一版改成 `src.count("quiz_service.tool_summary") == 2` 之后**当场就翻了车**：
        我在 `_node_progress_events` 的**注释**里提了一句它，计数变成 3 —— 正是我
        刚审完的那个"注释能把计数撑起来"的毛病。所以这里改用 **AST 只数真实调用**，
        并顺带把"内容截断"也改成 AST 判据（注释同样骗不过）。
        """
        import ast
        import inspect

        from course_agent import server

        tree = ast.parse(inspect.getsource(server))

        calls = [n for n in ast.walk(tree)
                 if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute)
                 and n.func.attr == "tool_summary"]
        assert len(calls) == 2, (
            f"tool_summary 应只有两处调用（/qa/ask 与 _node_progress_events），实际 {len(calls)} 处")

        # 不许对 content 做切片（原样截断工具返回内容的老写法）
        slices = [n for n in ast.walk(tree)
                  if isinstance(n, ast.Subscript)
                  and isinstance(n.value, ast.Attribute)
                  and n.value.attr == "content"]
        assert not slices, "又出现了「原样截断工具返回内容」的写法（对 content 取切片）"

        # 唯一的进度事件产出点（原来 /qa/stream 与 /chat/stream 各写一份）
        assert sum(1 for n in ast.walk(tree)
                   if isinstance(n, ast.FunctionDef)
                   and n.name == "_node_progress_events") == 1
        decorators = [ast.unparse(d) for n in ast.walk(tree)
                      if isinstance(n, ast.FunctionDef) for d in n.decorator_list]
        assert not any("/chat/stream" in d for d in decorators), \
            "重复的端点是删掉过的，不该回来"
