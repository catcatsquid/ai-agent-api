"""Week9-2: 中间件测试

测试限流、请求日志、全局异常处理。
"""

import time
import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from fastapi import FastAPI
from pydantic import BaseModel

from course_agent.middleware import (
    setup_middlewares,
    RateLimitMiddleware,
    RequestLogMiddleware,
)


# 用一个最小的 App 来测试中间件
@pytest.fixture
def app():
    app = FastAPI()
    setup_middlewares(app)

    @app.get("/")
    def health_endpoint():
        return {"code": 0, "msg": "ok"}

    @app.get("/ok")
    def ok_endpoint():
        return {"code": 0, "msg": "ok"}

    @app.get("/error")
    def error_endpoint():
        raise ValueError("故意报错")

    return app


@pytest.fixture
def client(app):
    return TestClient(app, raise_server_exceptions=False)


# ============================================================
# 请求日志中间件测试
# ============================================================

class TestRequestLogMiddleware:
    """请求日志中间件测试"""

    def test_response_has_x_response_time_header(self, client):
        """响应头里应该有 X-Response-Time"""
        response = client.get("/ok")
        assert response.status_code == 200
        assert "x-response-time" in response.headers
        # 应该是个数字（毫秒）
        rt = response.headers["x-response-time"]
        assert rt.replace(".", "").isdigit() or rt.isdigit()

    def test_normal_request_200(self, client):
        """正常请求返回 200"""
        response = client.get("/ok")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0


# ============================================================
# 限流中间件测试
# ============================================================

class TestRateLimitMiddleware:
    """限流中间件测试"""

    def test_no_rate_limit_when_zero(self, client):
        """RATE_LIMIT_PER_MINUTE=0 时不限制"""
        with patch("course_agent.middleware.RATE_LIMIT_PER_MINUTE", 0):
            for _ in range(100):
                response = client.get("/ok")
                assert response.status_code == 200

    def test_rate_limit_blocks_after_limit(self, app):
        """超过限流阈值后返回 429"""
        with patch("course_agent.middleware.RATE_LIMIT_PER_MINUTE", 3):
            client = TestClient(app)

            # 前 3 次应该通过
            for i in range(3):
                response = client.get("/ok")
                assert response.status_code == 200, f"第 {i+1} 次应该通过"

            # 第 4 次应该被拦截
            response = client.get("/ok")
            assert response.status_code == 429
            data = response.json()
            assert data["code"] == 429
            assert "频繁" in data["msg"] or "limit" in data["msg"].lower()

    def test_rate_limit_per_ip(self, app):
        """不同 IP 的限流计数器应该独立"""
        with patch("course_agent.middleware.RATE_LIMIT_PER_MINUTE", 2):
            # 用 TestClient 不方便模拟不同 IP，直接测内部方法
            rl = RateLimitMiddleware(app)

            # IP1 用了 2 次额度
            assert rl._check_rate_limit("1.1.1.1") is True
            assert rl._check_rate_limit("1.1.1.1") is True
            assert rl._check_rate_limit("1.1.1.1") is False  # 第3次被拦

            # IP2 还有额度
            assert rl._check_rate_limit("2.2.2.2") is True
            assert rl._check_rate_limit("2.2.2.2") is True
            assert rl._check_rate_limit("2.2.2.2") is False

    def test_health_check_not_rate_limited(self, app):
        """健康检查接口不应该被限流"""
        with patch("course_agent.middleware.RATE_LIMIT_PER_MINUTE", 1):
            client = TestClient(app)
            # 健康检查 (/) 不限流
            for _ in range(5):
                response = client.get("/")
                assert response.status_code == 200

            # 但普通接口还是会被限流
            response = client.get("/ok")
            assert response.status_code == 200  # 第1次通过
            response = client.get("/ok")
            assert response.status_code == 429  # 第2次被拦


# ============================================================
# 全局异常处理测试
# ============================================================

class TestGlobalExceptionHandler:
    """全局异常处理器测试"""

    def test_500_error_returns_json(self, client):
        """服务器异常应该返回统一格式的 JSON，而不是 HTML"""
        response = client.get("/error")
        assert response.status_code == 500

        data = response.json()
        assert "code" in data
        assert data["code"] == 500
        assert "msg" in data
        # 不应该是 HTML
        assert "<html" not in response.text.lower()

    def test_404_returns_json(self, client):
        """404 应该返回统一格式的 JSON"""
        response = client.get("/nonexistent/path")
        assert response.status_code == 404

        data = response.json()
        assert data["code"] == 404
        assert "不存在" in data["msg"] or "not found" in data["msg"].lower()

    def test_405_returns_json(self, client):
        """405 方法不允许应该返回统一格式的 JSON"""
        response = client.post("/ok")  # /ok 只支持 GET
        assert response.status_code == 405

        data = response.json()
        assert data["code"] == 405
        assert "不允许" in data["msg"] or "not allowed" in data["msg"].lower()


# ============================================================
# CORS 测试
# ============================================================

class TestCORS:
    """CORS 跨域测试"""

    def test_cors_headers_present(self, client):
        """响应头里应该有 CORS 相关字段"""
        response = client.get("/ok", headers={"Origin": "http://example.com"})
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers
        # 当 allow_credentials=True 时，CORS 规范要求返回具体 origin 而不是 *
        assert response.headers["access-control-allow-origin"] == "http://example.com"

    def test_cors_preflight(self, client):
        """OPTIONS 预检请求应该返回 200"""
        response = client.options(
            "/ok",
            headers={
                "Origin": "http://example.com",
                "Access-Control-Request-Method": "GET",
            },
        )
        assert response.status_code == 200
        assert "access-control-allow-origin" in response.headers


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
