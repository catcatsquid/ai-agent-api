"""全局配置 - 统一管理环境变量

所有模块通过 `from config import xxx` 获取配置，不直接调用 os.getenv。
.env 文件由 load_dotenv() 加载，缺失必填项时启动即报错。
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(name: str) -> str:
    """读取必填环境变量，缺失时抛出明确错误"""
    val = os.getenv(name)
    if not val:
        raise EnvironmentError(f"环境变量 {name} 未设置，请在 .env 文件中配置")
    return val


def _optional(name: str, default: str = "") -> str:
    """读取可选环境变量，缺失时返回默认值"""
    return os.getenv(name, default)


# ============================================================
# 智谱 API
# ============================================================

ZHIPU_API_KEY = _require("ZHIPU_API_KEY")
ZHIPU_CHAT_URL = _optional("ZHIPU_CHAT_URL", "https://open.bigmodel.cn/api/paas/v4/chat/completions")
ZHIPU_EMBEDDING_URL = _optional("ZHIPU_EMBEDDING_URL", "https://open.bigmodel.cn/api/paas/v4/embeddings")
ZHIPU_MODEL = _optional("ZHIPU_MODEL", "glm-4-flash")

# ============================================================
# DeepSeek API
# ============================================================

DEEPSEEK_API_KEY = _optional("DEEPSEEK_API_KEY", "")
DEEPSEEK_CHAT_URL = _optional("DEEPSEEK_CHAT_URL", "https://api.deepseek.com/chat/completions")

# ============================================================
# Agent 参数
# ============================================================

LLM_TEMPERATURE = float(_optional("LLM_TEMPERATURE", "0.7"))
MAX_TOOL_STEPS = int(_optional("MAX_TOOL_STEPS", "10"))
CHUNK_SIZE = int(_optional("CHUNK_SIZE", "300"))
CHUNK_OVERLAP = int(_optional("CHUNK_OVERLAP", "50"))
EMBEDDING_BATCH_SIZE = int(_optional("EMBEDDING_BATCH_SIZE", "20"))

# ============================================================
# 日志
# ============================================================

LOG_LEVEL = _optional("LOG_LEVEL", "INFO").upper()
LOG_FILE = _optional("LOG_FILE", "")

# ============================================================
# API 认证
# ============================================================

# API Key（为空则不开启认证，开发模式）
API_KEY = _optional("API_KEY", "")

# JWT 配置（JWT_SECRET 为空则不开启 JWT 认证）
JWT_SECRET = _optional("JWT_SECRET", "")
JWT_ALGORITHM = _optional("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(_optional("JWT_EXPIRE_MINUTES", "60"))

# ============================================================
# 限流
# ============================================================

# 每分钟每个 IP 最多请求数（0 表示不限制）
RATE_LIMIT_PER_MINUTE = int(_optional("RATE_LIMIT_PER_MINUTE", "0"))
