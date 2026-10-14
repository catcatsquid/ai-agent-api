"""端点清单守卫：服务端**实际暴露的路由**必须与登记的一致

## 为什么需要它

1.1.0 合并流式端点时，我用脚本把 `server.py` 里从 `/qa/stream` 到"知识库管理页面"
横幅之间的**整段**替换掉了 —— 本意只是删掉重复的 `/chat/stream`，结果把紧挨着它的
`POST /tutor/stream` **也一起删了**（server.py 里 tutor 相关行 5 → 2）。
当时我核对的只有"`/chat/stream` 还剩几处"，**没有问"是不是少了别的"**。

它四个版本都没被发现，因为：

- 界面上早就撤掉了助教入口 —— 没有任何用户路径会 404；
- 助教的测试全是直接调 `tutor_agent` 的函数，**从不经过 HTTP**；
- 于是 README 的接口表和"代码与接口全部保留"那句话，从 1.1.0 起就是**假的**。

这类事故的通用形态是「**我确认了我预期的变化，却没确认没有别的变化**」——
所以这里把清单钉死：少一个、多一个都要显式改这份清单，改动必须是有意识的。
"""
from fastapi.testclient import TestClient

from course_agent import server
from course_agent.server import app

# 有意保留的端点清单。改动这里必须是有意识的：要么新增功能，要么明确废弃。
EXPECTED_ROUTES = {
    ("GET", "/"),
    ("GET", "/health"),
    ("GET", "/chat"),
    ("GET", "/kb"),
    ("POST", "/auth/login"),
    ("POST", "/qa/ask"),
    ("POST", "/qa/stream"),
    # 手写助教：冻结的对照实现，界面上没有入口，但**接口保留**（1.1.0 误删过）
    ("POST", "/tutor/stream"),
    ("POST", "/chat/export"),
    ("GET", "/chat/conversations"),
    ("GET", "/chat/conversations/{thread_id}"),
    ("DELETE", "/chat/conversations/{thread_id}"),
    ("GET", "/kb/stats"),
    ("GET", "/kb/documents"),
    ("DELETE", "/kb/documents/{source}"),
    ("POST", "/kb/search"),
    ("POST", "/kb/add"),
    ("POST", "/kb/add-directory"),
    ("GET", "/kb/browse"),
    ("GET", "/qb/list"),
    ("POST", "/qb/search"),
    ("GET", "/qb/topics"),
}

# FastAPI 自带的，不算我们的产品面
_FRAMEWORK_PATHS = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}


def _actual_routes() -> set:
    out = set()
    for route in app.routes:
        path = getattr(route, "path", None)
        methods = getattr(route, "methods", None)
        if not path or not methods or path in _FRAMEWORK_PATHS:
            continue
        for method in methods:
            out.add((method, path))
    return out


class TestRouteSurface:
    def test_no_route_disappeared(self):
        missing = EXPECTED_ROUTES - _actual_routes()
        assert not missing, (
            "有端点不见了：\n  "
            + "\n  ".join(f"{m} {p}" for m, p in sorted(missing))
            + "\n  —— 有意删除的话请同时更新本文件的清单；"
              "误删（1.1.0 就这样丢过 POST /tutor/stream）就把它加回来。")

    def test_no_route_appeared_without_being_registered(self):
        extra = _actual_routes() - EXPECTED_ROUTES
        assert not extra, (
            "多了没登记的端点：\n  "
            + "\n  ".join(f"{m} {p}" for m, p in sorted(extra))
            + "\n  —— 新端点请登记进 EXPECTED_ROUTES（顺便想一下它该不该要求鉴权）。")


class TestTutorStreamActuallyWorks:
    """光"路由在清单里"不够 —— 它得真的能跑

    1.1.0 的教训：端点消失时没有任何测试会红。这里用一个打桩的 `process_message`
    真发一次请求，确认 SSE 链路是通的（不产生任何网络调用）。
    """

    def test_streams_tutor_events(self, monkeypatch):
        monkeypatch.setattr(
            server.tutor_agent, "process_message",
            lambda session_id, text: [
                {"type": "start", "session_id": "tutor-test-1", "query": text},
                {"type": "done", "answer": "好的", "session_id": "tutor-test-1"},
            ])
        client = TestClient(app)
        with client.stream("POST", "/tutor/stream", json={"query": "你好"}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert '"type": "start"' in body, body
        assert "tutor-test-1" in body, body
        assert '"answer"' in body, body

    def test_errors_are_reported_as_an_sse_event_not_a_500(self, monkeypatch):
        """助教内部抛异常时，应当作为 error 事件流出（和 /qa/stream 一致）"""
        def boom(session_id, text):
            raise RuntimeError("模型炸了")

        monkeypatch.setattr(server.tutor_agent, "process_message", boom)
        client = TestClient(app)
        with client.stream("POST", "/tutor/stream", json={"query": "你好"}) as resp:
            assert resp.status_code == 200
            body = "".join(resp.iter_text())
        assert '"type": "error"' in body, body
        assert "模型炸了" in body, body
