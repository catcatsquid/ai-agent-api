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
        """健康检查接口不需要认证"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0

    @patch("course_agent.auth.JWT_SECRET", "test-secret")
    @patch("course_agent.auth.API_KEY", "")
    def test_login_endpoint(self):
        """登录接口返回 JWT"""
        response = client.post("/auth/login", json={"username": "testuser"})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 0
        assert "token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] > 0

    def test_login_empty_username(self):
        """用户名为空时返回错误"""
        response = client.post("/auth/login", json={"username": ""})
        assert response.status_code == 200
        data = response.json()
        assert data["code"] == 400


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
