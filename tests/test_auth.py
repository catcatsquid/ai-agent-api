"""Week9-1: API 认证测试

测试 API Key 和 JWT 两种认证方式。
"""

import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

from course_agent.server import app
from course_agent.auth import create_jwt, decode_jwt, api_key_auth

client = TestClient(app)


# ============================================================
# JWT 单元测试
# ============================================================

class TestJWT:
    """JWT 功能测试"""

    def test_create_and_decode_jwt(self):
        """生成 JWT 后能正确解码"""
        with patch("course_agent.auth.JWT_SECRET", "test-secret-123"), \
             patch("course_agent.auth.JWT_ALGORITHM", "HS256"):
            token = create_jwt("user123", extra={"role": "admin"})
            assert isinstance(token, str)
            assert len(token) > 20  # JWT 有固定的三段结构，肯定大于20字符

            payload = decode_jwt(token)
            assert payload["sub"] == "user123"
            assert payload["role"] == "admin"
            assert "exp" in payload
            assert "iat" in payload
            assert "jti" in payload

    def test_jwt_expired(self):
        """过期的 JWT 应该报错"""
        import jwt as pyjwt

        with patch("course_agent.auth.JWT_SECRET", "test-secret-123"):
            # 手动造一个过期的 token
            import time
            expired_payload = {
                "sub": "user1",
                "exp": int(time.time()) - 3600,  # 已经过期1小时
            }
            expired_token = pyjwt.encode(expired_payload, "test-secret-123", algorithm="HS256")

            with pytest.raises(Exception) as exc_info:
                decode_jwt(expired_token)
            assert "过期" in str(exc_info.value) or "expired" in str(exc_info.value).lower()

    def test_jwt_invalid(self):
        """无效的 JWT 应该报错"""
        with patch("course_agent.auth.JWT_SECRET", "test-secret-123"):
            with pytest.raises(Exception) as exc_info:
                decode_jwt("fake-token-12345")
            assert "无效" in str(exc_info.value) or "invalid" in str(exc_info.value).lower()


# ============================================================
# API Key 测试
# ============================================================

class TestAPIKey:
    """API Key 认证测试"""

    def test_valid_api_key(self):
        """正确的 API Key 应该通过"""
        with patch("course_agent.auth.API_KEY", "my-secret-key"):
            result = api_key_auth("my-secret-key")
            assert result == "my-secret-key"

    def test_invalid_api_key(self):
        """错误的 API Key 应该返回 401"""
        with patch("course_agent.auth.API_KEY", "my-secret-key"):
            with pytest.raises(Exception) as exc_info:
                api_key_auth("wrong-key")
            assert "401" in str(exc_info.value) or "401" in str(getattr(exc_info.value, 'status_code', ''))

    def test_missing_api_key(self):
        """缺失 API Key 应该返回 401"""
        with patch("course_agent.auth.API_KEY", "my-secret-key"):
            with pytest.raises(Exception) as exc_info:
                api_key_auth(None)
            assert "401" in str(exc_info.value) or "401" in str(getattr(exc_info.value, 'status_code', ''))

    def test_no_api_key_configured(self):
        """未配置 API_KEY 时跳过认证"""
        with patch("course_agent.auth.API_KEY", ""):
            result = api_key_auth(None)
            assert result == "dev-mode"


# ============================================================
# 接口集成测试
# ============================================================

class TestAuthEndpoints:
    """认证接口测试"""

    def test_health_no_auth(self):
        """健康检查接口不需要认证

        注意：这里必须打 /health，不能打 "/" —— "/" 已经是 302 跳 /chat 的页面入口，
        探活拿到 307/HTML 会误判。原测试就是因为用了 "/" 而长期失败。
        """
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert data["status"] == "healthy"
        assert data["question_bank_size"] > 0

    def test_root_redirects_to_chat(self):
        """根路径跳转到聊天页"""
        response = client.get("/", follow_redirects=False)
        assert response.status_code in (302, 307)
        assert response.headers["location"] == "/chat"

    @patch("course_agent.auth.JWT_SECRET", "test-secret")
    @patch("course_agent.auth.API_KEY", "")
    @patch("course_agent.auth.ADMIN_PASSWORD", "test-pwd")
    @patch("config.ADMIN_PASSWORD", "test-pwd")
    def test_login_endpoint(self):
        """凭正确密码登录，返回 JWT"""
        response = client.post("/auth/login",
                               json={"username": "testuser", "password": "test-pwd"})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    @patch("course_agent.auth.JWT_SECRET", "test-secret")
    @patch("course_agent.auth.ADMIN_PASSWORD", "test-pwd")
    @patch("config.ADMIN_PASSWORD", "test-pwd")
    def test_login_wrong_password_is_rejected(self):
        """密码错误必须拒绝签发令牌

        原来 /auth/login 从不读取 password：任意用户名 + 任意密码都能拿到 token，
        于是"需要鉴权"的写接口（/kb/add 等）对匿名者等于不设防。
        """
        response = client.post("/auth/login",
                               json={"username": "testuser", "password": "WRONG"})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 401
        assert not data.get("token")

    @patch("course_agent.auth.JWT_SECRET", "test-secret")
    @patch("course_agent.auth.ADMIN_PASSWORD", "")
    @patch("config.ADMIN_PASSWORD", "")
    def test_login_fails_closed_without_admin_password(self):
        """未配置 ADMIN_PASSWORD 时不能无条件签发（fail closed）"""
        response = client.post("/auth/login",
                               json={"username": "testuser", "password": "anything"})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 503
        assert not data.get("token")

    @patch("course_agent.auth.JWT_SECRET", "")
    @patch("course_agent.auth.JWT_ENABLED", False)
    @patch("course_agent.auth.ADMIN_PASSWORD", "test-pwd")
    @patch("config.ADMIN_PASSWORD", "test-pwd")
    def test_login_unavailable_without_jwt_secret(self):
        """没配 JWT_SECRET 时签发的 token 也不会被接受，所以直接拒绝"""
        response = client.post("/auth/login",
                               json={"username": "testuser", "password": "test-pwd"})
        assert response.status_code == 200
        assert response.json()["code"] == 503

    @patch("course_agent.auth.JWT_SECRET", "test-secret")
    @patch("course_agent.auth.ADMIN_PASSWORD", "test-pwd")
    @patch("config.ADMIN_PASSWORD", "test-pwd")
    def test_login_empty_username(self):
        """用户名为空时返回错误"""
        response = client.post("/auth/login",
                               json={"username": "", "password": "test-pwd"})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 400


class TestDefaultDenyGuard:
    """默认拒绝的鉴权守卫：没在白名单里的接口都要凭据

    原来鉴权是逐个路由手写 Depends(any_auth) 的，`/qa/ask` 加了、
    第 6 周写的 `/qa/stream` 忘了加 —— 而前端实际调的就是 `/qa/stream`。
    于是"加了鉴权"挡住的恰好是没人用的接口。

    注意：AUTH_ENFORCE 默认是 false（本地免登录），所以这里显式打开，
    只测守卫本身的行为。
    """

    @pytest.fixture(autouse=True)
    def _enforce(self):
        with patch("course_agent.middleware.AUTH_ENFORCE", True), \
             patch("course_agent.auth.AUTH_ENFORCE", True):
            yield

    @pytest.mark.parametrize("method,path", [
        ("GET", "/kb/stats"),
        ("GET", "/kb/documents"),
        ("POST", "/kb/search"),
        ("GET", "/kb/browse"),
        ("GET", "/qb/list"),
        ("POST", "/qb/search"),
        ("GET", "/qb/topics"),
        ("GET", "/chat/conversations"),
        ("DELETE", "/chat/conversations/t1"),
        # 手写助教接口（界面上没有入口，但接口保留）—— 它会真的花掉 LLM 额度
        ("POST", "/tutor/stream"),
    ])
    def test_sensitive_endpoints_require_auth(self, method, path):
        """这些接口原本全部匿名可用（能列出语料、能检索全文）
        /chat/conversations 还会吐出**每个会话的第一句提问**，同样不该匿名可读。"""
        resp = client.request(method, path, json={"query": "x"} if method == "POST" else None)
        assert resp.status_code == 401, f"{method} {path} 竟然不需要鉴权"
        assert resp.json()["code"] == 401

    @pytest.mark.parametrize("path", ["/health", "/chat", "/kb", "/", "/docs", "/openapi.json"])
    def test_public_paths_stay_open(self, path):
        """页面本体与探活必须匿名可访问，否则页面自己都打不开"""
        resp = client.get(path, follow_redirects=False)
        assert resp.status_code != 401, f"{path} 不该要求鉴权"

    def test_login_is_public(self):
        """/auth/login 不能要求鉴权，否则永远拿不到令牌"""
        resp = client.post("/auth/login", json={"username": "u", "password": "p"})
        assert resp.status_code != 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
