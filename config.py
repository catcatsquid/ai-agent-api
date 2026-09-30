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

LLM_TEMPERATURE = float(_optional("LLM_TEMPERATURE", "0.9"))
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
# 注意: /auth/login 会签发 token, 若 JWT_SECRET 为空则自动生成进程级随机密钥,
#       重启后旧 token 失效。生产环境务必在 .env 中配置固定强密钥。
import secrets as _secrets
JWT_SECRET = _optional("JWT_SECRET", "") or _secrets.token_hex(32)
JWT_ALGORITHM = _optional("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(_optional("JWT_EXPIRE_MINUTES", "60"))

# ============================================================
# 限流
# ============================================================

# 每分钟每个 IP 最多请求数（0 表示不限制）
RATE_LIMIT_PER_MINUTE = int(_optional("RATE_LIMIT_PER_MINUTE", "0"))

# ============================================================
# 对话记忆管理
# ============================================================

# 最大消息条数（滑动窗口：超过就裁剪老消息）
MAX_MESSAGES = int(_optional("MAX_MESSAGES", "20"))

# 启用对话摘要压缩（老消息太多时用 LLM 总结成摘要）
ENABLE_SUMMARY = _optional("ENABLE_SUMMARY", "true").lower() == "true"

# 超过多少条消息后触发摘要（需要 ENABLE_SUMMARY=true）
SUMMARY_THRESHOLD = int(_optional("SUMMARY_THRESHOLD", "30"))

# 摘要后保留最近几轮对话（不被摘要掉）
SUMMARY_KEEP_LAST_N = int(_optional("SUMMARY_KEEP_LAST_N", "10"))
