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
# 应用版本
# ============================================================

# 唯一真源：与 git tag / GitHub Release / CHANGELOG.md 三者对齐。
#
# 原来版本号只硬编码在 course_agent/server.py 的 FastAPI(version="3.0") 里，
# 周围没有一句说明，也没有对应的 tag 或释出记录 —— 一个自称 3.0 的 API
# 却查不到任何版本历史。现在收到这里，发版只改一处。
APP_VERSION = _optional("APP_VERSION", "1.4.1")


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

# 管理密码：/auth/login 用它换取 JWT。
#
# 原来 /auth/login 完全不看 password —— 任意用户名 + 任意密码都能换到 token，
# 于是"需要鉴权"的写接口（/kb/add 等）对匿名者等于不设防：
# 先换一个 JWT，再拿它去打任意路径入库。
# 现在：未配置 ADMIN_PASSWORD 时登录返回 503（fail closed），不再无条件签发。
ADMIN_PASSWORD = _optional("ADMIN_PASSWORD", "")

# 是否强制鉴权。
#
# 默认 **false** —— 这是本地学习项目的真实用法：服务只绑 127.0.0.1，
# 外面访问不到，日常每次输密码的摩擦换不来实际安全增益。
#
# 关掉它并不等于"裸奔"：真正危险的那条链（/kb/add 任意文件入库 → /kb/search 外泄）
# 靠的是**处理函数内部的路径校验** `resolve_within_data()`，与鉴权完全解耦，
# 关掉鉴权后仍然会 403 拒绝 data/ 之外的文件（tests/test_server_security.py 有守卫）。
#
# 对外部署前请设 AUTH_ENFORCE=true，并配置 ADMIN_PASSWORD（登录用）或 API_KEY（服务间调用）。
AUTH_ENFORCE = _optional("AUTH_ENFORCE", "false").lower() == "true"

# JWT 配置
# 说明: 这里保留"未配置"和"已配置"的区别，供 auth.py 判断是否开启 JWT 认证。
#       原来写成 `_optional("JWT_SECRET", "") or _secrets.token_hex(32)`，
#       空值时自动生成随机密钥，导致 auth.py 里 `if not JWT_SECRET: 跳过认证`
#       永远是 False —— 文档说"为空则不开启"，实际是所有鉴权接口 401。
#       现在: 显式配置才开启 JWT；未配置时由 auth.py 走"仅 API Key"模式。
JWT_SECRET = _optional("JWT_SECRET", "")
JWT_SECRET_IS_CONFIGURED = bool(JWT_SECRET)
JWT_ALGORITHM = _optional("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(_optional("JWT_EXPIRE_MINUTES", "60"))

if not JWT_SECRET:
    # 开发环境仍可签发/校验 token（用进程级随机密钥），但会明确告知
    import secrets as _secrets

    JWT_SECRET = _secrets.token_hex(32)
    JWT_SECRET_IS_EPHEMERAL = True
else:
    JWT_SECRET_IS_EPHEMERAL = False

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

# ============================================================
# 检索相关度阈值
# ============================================================

# 判定"知识库是否有相关内容"的绝对余弦相似度阈值。
# 用途: rag_tool.search_knowledge 决定是否补充搜课本 / 是否提示调用 web_search。
#
# 这个值必须作用在**绝对**相似度上，不能作用在混合检索的归一化融合分上 ——
# 融合分是按最大值归一化的，排名第一的结果恒 ≥ 0.6，拿它比 0.3 永远不成立。
#
# 取值依据（debug/check_threshold.py，域内/域外各 12 条查询实测）:
#   换算成余弦相似度后  域内最低 0.665，域外最高 0.524
#   0.55 可把 24 条样本 100% 分开
RELEVANCE_THRESHOLD = float(_optional("RELEVANCE_THRESHOLD", "0.55"))
