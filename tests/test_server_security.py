"""服务层安全与协议修复的回归测试

覆盖：
  · /kb/add、/kb/add-directory 的路径包含性校验（任意文件入库链）
  · /kb/browse 仍然只允许 data/
  · 工具产出不再被当作正文流式播给前端（泄题通道）
"""
import pytest
from unittest.mock import patch
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

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
    """工具产出不得进入正文事件（原实现会把 ToolMessage 当正文播出去）"""

    def test_tool_message_is_dropped(self):
        msg = ToolMessage(content='{"answer": "D", "reference_points": ["因为..."]}',
                          tool_call_id="c1", name="grade_answers")
        assert sse_content_event(msg) is None

    def test_message_with_type_tool_is_dropped(self):
        class FakeToolMsg:
            type = "tool"
            content = "检索原文"
        assert sse_content_event(FakeToolMsg()) is None

    def test_ai_message_with_tool_calls_is_dropped(self):
        msg = AIMessage(content="", tool_calls=[
            {"id": "c1", "name": "search_knowledge", "args": {"query": "x"}}
        ])
        assert sse_content_event(msg) is None

    def test_plain_ai_message_becomes_content_event(self):
        line = sse_content_event(AIMessage(content="这是最终答案"))
        assert line is not None
        assert line.startswith("data: ")
        assert '"type": "content"' in line
        assert "这是最终答案" in line

    def test_human_message_also_passes(self):
        """用户消息本来就该显示，不该被过滤（区别于工具消息）"""
        assert sse_content_event(HumanMessage(content="你好")) is not None

    def test_empty_content_is_skipped(self):
        assert sse_content_event(AIMessage(content="")) is None
