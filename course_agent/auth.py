"""API 认证模块

提供两种认证方式:
  - API Key: 简单的固定密钥认证，适合服务间调用
  - JWT: 令牌认证，适合用户登录场景，支持过期时间和用户信息

使用方式:
  from course_agent.auth import api_key_auth, jwt_auth
  # 在路由函数中作为依赖注入
  @app.get("/protected")
  def protected(api_key: str = Depends(api_key_auth)):
      return {"msg": "ok"}
"""

import time
import secrets
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials

from config import API_KEY, JWT_SECRET, JWT_ALGORITHM, JWT_EXPIRE_MINUTES
from course_agent.logger import logger


# ============================================================
# API Key 认证
# ============================================================

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def api_key_auth(api_key: Optional[str] = Depends(_api_key_header)) -> str:
    """API Key 认证依赖

    从请求头 X-API-Key 读取密钥，与配置的 API_KEY 比对。

    Returns:
        验证通过的 API Key 字符串

    Raises:
        HTTPException: 401 - 未提供或密钥错误
    """
    if not API_KEY:
        # 未配置 API_KEY 时跳过认证（开发模式）
        logger.warning("API_KEY 未配置，API Key 认证已跳过")
        return "dev-mode"

    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 API Key，请在请求头 X-API-Key 中提供",
        )

    # 使用 secrets.compare_digest 防时序攻击
    if not secrets.compare_digest(api_key, API_KEY):
        logger.warning("API Key 认证失败")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API Key 无效",
        )

    return api_key


# ============================================================
# JWT 认证
# ============================================================

_bearer_scheme = HTTPBearer(auto_error=False)


def create_jwt(user_id: str, extra: dict = None) -> str:
    """生成 JWT 令牌

    Args:
        user_id: 用户标识
        extra: 额外的载荷数据

    Returns:
        JWT 令牌字符串
    """
    now = int(time.time())
    payload = {
        "sub": user_id,                    # subject: 用户标识
        "iat": now,                        # issued at: 签发时间
        "exp": now + JWT_EXPIRE_MINUTES * 60,  # expiration: 过期时间
        "jti": secrets.token_hex(8),       # JWT ID: 唯一标识，防重放
    }
    if extra:
        payload.update(extra)

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    logger.debug("生成 JWT: user_id=%s, 有效期=%d分钟", user_id, JWT_EXPIRE_MINUTES)
    return token


def decode_jwt(token: str) -> dict:
    """解析并验证 JWT 令牌

    Args:
        token: JWT 令牌字符串

    Returns:
        解码后的 payload 字典

    Raises:
        HTTPException: 401 - 令牌无效或过期
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌已过期，请重新获取",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="令牌无效",
        )


def jwt_auth(credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme)) -> dict:
    """JWT 认证依赖

    从 Authorization: Bearer <token> 请求头读取令牌并验证。

    Returns:
        解码后的 payload 字典（包含 user_id 等信息）

    Raises:
        HTTPException: 401 - 未提供令牌或令牌无效
    """
    if not JWT_SECRET:
        # 未配置 JWT_SECRET 时跳过认证（开发模式）
        logger.warning("JWT_SECRET 未配置，JWT 认证已跳过")
        return {"sub": "dev-user", "dev": True}

    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="缺少 Authorization Bearer 令牌",
        )

    return decode_jwt(credentials.credentials)


# ============================================================
# 通用：任意一种认证通过即可
# ============================================================

def any_auth(
    api_key: Optional[str] = Depends(_api_key_header),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer_scheme),
) -> dict:
    """混合认证：API Key 或 JWT 任一通过即可

    优先尝试 API Key，再尝试 JWT。都不通过则返回 401。

    Returns:
        {"type": "api_key"|"jwt", ...} 认证信息字典
    """
    # 开发模式：都没配置时跳过
    if not API_KEY and not JWT_SECRET:
        logger.warning("API_KEY 和 JWT_SECRET 都未配置，认证已跳过")
        return {"type": "dev", "user": "dev-user"}

    # 尝试 API Key
    if api_key and API_KEY:
        if secrets.compare_digest(api_key, API_KEY):
            return {"type": "api_key", "user": "api-user"}

    # 尝试 JWT
    if credentials and credentials.scheme.lower() == "bearer" and JWT_SECRET:
        try:
            payload = decode_jwt(credentials.credentials)
            return {"type": "jwt", "user": payload.get("sub", ""), "payload": payload}
        except HTTPException:
            pass

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="认证失败，请提供有效的 X-API-Key 或 Authorization Bearer 令牌",
    )
