"""中间件模块

包含:
  - RequestLogMiddleware: 请求日志（记录 URL、方法、状态码、耗时、IP）
  - RateLimitMiddleware:  限流（按 IP 统计每分钟请求数）
  - 全局异常处理器: 统一捕获异常，返回友好的 JSON 错误响应

设计原则:
  - 日志和限流都做成可选，环境变量控制开关
  - 不影响正常业务逻辑，出问题宁可降级也不搞崩服务
"""

import time
import json
import uuid
from collections import defaultdict
from threading import Lock

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from config import RATE_LIMIT_PER_MINUTE, AUTH_ENFORCE, API_KEY, ADMIN_PASSWORD
from course_agent.logger import logger, set_trace_id, clear_trace_id
from course_agent.auth import check_credentials


# 噪声路径：探活、文档、图标。**两处都要用同一份**：
#   · 请求日志跳过它们（太频繁）
#   · 限流放过它们（否则探活会被 429 判死）
#
# 原来这两处各自硬编码了一份，而且**都漏了 /health**（放的是没意义的 "/"）：
# 真实健康检查端点会被限流拦成 429，而我自己的测试恰好断言的是 "/" ——
# 测试绿了、线上探活会挂。这是"测试测了错误的对象"的典型案例。
# 现在合成一个常量，谁也不许再各写一份。
QUIET_PATHS = frozenset({
    "/", "/health", "/favicon.ico",
    "/docs", "/redoc", "/openapi.json",
})


# ============================================================
# 1. 请求日志中间件
# ============================================================

class RequestLogMiddleware(BaseHTTPMiddleware):
    """请求日志中间件

    记录每个请求的:
      - HTTP 方法、URL 路径
      - 客户端 IP
      - 响应状态码
      - 处理耗时（毫秒）

    格式示例:
      INFO | 127.0.0.1 | POST /qa/ask | 200 | 1234ms
    """

    async def dispatch(self, request: Request, call_next):
        # 生成 trace_id（取前8位，够短也够唯一）
        trace_id = uuid.uuid4().hex[:8]
        set_trace_id(trace_id)

        try:
            # 记录开始时间
            start_time = time.time()
            client_ip = request.client.host if request.client else "unknown"
            method = request.method
            path = request.url.path

            # 跳过噪声路径的日志（探活/文档/图标太频繁）
            skip_log = path in QUIET_PATHS

            try:
                # 交给下一层（路由函数）处理
                response = await call_next(request)
            except Exception as e:
                # 异常会被全局异常处理器捕获，这里只记录
                duration = (time.time() - start_time) * 1000
                if not skip_log:
                    logger.error("请求异常 | %s | %s %s | %.0fms | %s",
                                 client_ip, method, path, duration,
                                 type(e).__name__)
                raise

            # 计算耗时
            duration_ms = (time.time() - start_time) * 1000

            # 打日志
            if not skip_log:
                status_code = response.status_code
                if status_code >= 500:
                    logger.error("请求完成 | %s | %s %s | %d | %.0fms",
                                 client_ip, method, path, status_code, duration_ms)
                elif status_code >= 400:
                    logger.warning("请求完成 | %s | %s %s | %d | %.0fms",
                                   client_ip, method, path, status_code, duration_ms)
                else:
                    logger.info("请求完成 | %s | %s %s | %d | %.0fms",
                                client_ip, method, path, status_code, duration_ms)

            # 给响应加自定义头
            response.headers["X-Response-Time"] = f"{duration_ms:.0f}"
            response.headers["X-Trace-Id"] = trace_id
            return response
        finally:
            # 请求结束，清理 trace_id
            clear_trace_id()


# ============================================================
# 2. 限流中间件
# ============================================================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """IP 级别的限流中间件

    使用"滑动窗口"思路：记录每个 IP 在当前分钟内的请求次数，
    超过阈值就返回 429 Too Many Requests。

    简单版实现：内存存储，进程内有效。生产环境建议用 Redis。

    配置:
      RATE_LIMIT_PER_MINUTE=0  -> 关闭限流
      RATE_LIMIT_PER_MINUTE=60 -> 每分钟 60 次
    """

    def __init__(self, app):
        super().__init__(app)
        # {ip: {"count": 次数, "minute": 分钟时间戳}}
        self._counters = defaultdict(lambda: {"count": 0, "minute": 0})
        self._lock = Lock()  # 线程安全

    def _get_current_minute(self) -> int:
        """获取当前分钟的时间戳（精度到分钟）"""
        return int(time.time() // 60)

    def _check_rate_limit(self, ip: str) -> bool:
        """检查是否超过限流

        Returns:
            True  = 允许通过
            False = 超过限制
        """
        if RATE_LIMIT_PER_MINUTE <= 0:
            return True  # 限流关闭

        current_minute = self._get_current_minute()

        with self._lock:
            # 顺手清理过期的 IP 条目：原来每个新 IP 都会永久留下一条记录，
            # 计数器只增不减，长期运行内存单调增长
            if len(self._counters) > 1024:
                stale = [k for k, v in self._counters.items() if v["minute"] != current_minute]
                for k in stale:
                    self._counters.pop(k, None)

            counter = self._counters[ip]

            # 新的一分钟，重置计数器
            if counter["minute"] != current_minute:
                counter["count"] = 0
                counter["minute"] = current_minute

            if counter["count"] >= RATE_LIMIT_PER_MINUTE:
                return False  # 超过限制

            counter["count"] += 1
            return True

    async def dispatch(self, request: Request, call_next):
        # 限流关闭时直接放行
        if RATE_LIMIT_PER_MINUTE <= 0:
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path

        # 探活与文档页不限流（/health 必须在里面，否则健康检查会被 429 判死）
        if path in QUIET_PATHS:
            return await call_next(request)

        if not self._check_rate_limit(client_ip):
            logger.warning("限流拦截 | %s | %s %s | 超过每分钟%d次限制",
                           client_ip, request.method, path, RATE_LIMIT_PER_MINUTE)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "code": 429,
                    "msg": f"请求过于频繁，请稍后再试（每分钟最多 {RATE_LIMIT_PER_MINUTE} 次）",
                },
            )

        return await call_next(request)


# ============================================================
# 3. 全局异常处理器
# ============================================================

def register_exception_handlers(app):
    """注册全局异常处理器

    统一捕获各种异常，返回格式一致的 JSON 响应，
    而不是 FastAPI 默认的 HTML 错误页。
    """

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """兜底异常处理器：捕获所有未处理的异常"""
        logger.error("未捕获异常 | %s | %s %s | %s: %s",
                     request.client.host if request.client else "unknown",
                     request.method,
                     request.url.path,
                     type(exc).__name__,
                     str(exc)[:200])

        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "code": 500,
                "msg": f"服务器内部错误: {type(exc).__name__}",
                "detail": str(exc)[:100],
            },
        )

    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        """404 处理"""
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={
                "code": 404,
                "msg": f"接口不存在: {request.method} {request.url.path}",
            },
        )

    @app.exception_handler(405)
    async def method_not_allowed_handler(request: Request, exc):
        """405 处理"""
        return JSONResponse(
            status_code=status.HTTP_405_METHOD_NOT_ALLOWED,
            content={
                "code": 405,
                "msg": f"方法不允许: {request.method} {request.url.path}",
            },
        )


# ============================================================
# 4. 鉴权守卫中间件（默认拒绝）
# ============================================================

# 公开路径：页面本体、健康检查、接口文档、登录端点。
# 这些必须匿名可访问，否则页面自己都打不开、也就没法带凭证去登录。
PUBLIC_PATHS = {
    "/", "/chat", "/kb", "/health",
    "/docs", "/redoc", "/openapi.json", "/favicon.ico",
    "/auth/login",
}

# 图片与静态资源一律公开：<img src> 天生带不了 Authorization 头，
# 若把 /qb_images 也拦下来，出题时题目图片会全部裂图。
# （题目图片本身不是机密；真正要挡的是"任意文件入库 + 检索外泄"那条链。）
PUBLIC_PREFIXES = ("/static", "/qb_images")


class AuthGuardMiddleware(BaseHTTPMiddleware):
    """默认拒绝的鉴权守卫

    原来鉴权是**逐个路由手写** `Depends(any_auth)` 的：`/qa/ask` 加了、
    第 6 周写的 `/qa/stream` 忘了加 —— 而前端实际调的就是 `/qa/stream`。
    于是"加了鉴权"挡住的恰好是没人用的接口，而会烧 LLM 额度的流式接口、
    会触发 embedding 的 `/kb/search`、能列出全部语料的 `/kb/documents`
    以及能任意文件入库的 `/kb/add` 全都可以匿名访问。

    这里改成**默认拒绝 + 显式白名单**：除 PUBLIC_PATHS/PUBLIC_PREFIXES 之外，
    所有请求都必须带有效凭据。新增路由会自动受保护，不会再"忘加"。
    """

    async def dispatch(self, request: Request, call_next):
        if not AUTH_ENFORCE:
            return await call_next(request)

        # CORS 预检不携带凭据，必须放行（本中间件在 CORS 之外层）
        if request.method == "OPTIONS":
            return await call_next(request)

        path = request.url.path
        if path in PUBLIC_PATHS or path.startswith(PUBLIC_PREFIXES):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "") or ""
        bearer = auth_header[7:].strip() if auth_header.lower().startswith("bearer ") else None
        info = check_credentials(request.headers.get("x-api-key"), bearer)

        if info is None:
            logger.warning("鉴权拦截: %s %s", request.method, path)
            return JSONResponse(
                status_code=status.HTTP_401_UNAUTHORIZED,
                content={
                    "code": 401,
                    "msg": "认证失败，请提供有效的 X-API-Key 或 Authorization Bearer 令牌",
                },
            )

        request.state.auth = info
        return await call_next(request)


# ============================================================
# 5. CORS 中间件配置
# ============================================================

def setup_cors(app):
    """配置 CORS 跨域

    开发环境允许所有来源（方便前端调试），
    生产环境应该限制为具体的域名。
    """
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],        # 允许的源（生产环境改成具体域名）
        allow_credentials=True,    # 允许携带 cookie
        allow_methods=["*"],       # 允许所有 HTTP 方法
        allow_headers=["*"],       # 允许所有请求头
        expose_headers=["X-Response-Time"],  # 暴露自定义响应头
    )


# ============================================================
# 6. 一键注册所有中间件
# ============================================================

def setup_middlewares(app):
    """一键注册所有中间件

    注意：中间件的顺序很重要。Starlette 的 add_middleware 是 insert(0)，
    也就是**最后注册的在最外层**；所以要得到"请求日志 → 限流 → 鉴权 → CORS → 路由"
    这个顺序，注册顺序必须是 CORS → 鉴权 → 限流 → 请求日志。

    当前顺序:
      请求 → 请求日志 → 限流 → 鉴权 → CORS → 路由
      响应 ← 请求日志 ← 限流 ← 鉴权 ← CORS ← 路由
    """
    # CORS 应该在最内层（最先处理请求，最后处理响应）
    setup_cors(app)

    # 鉴权守卫：默认拒绝，白名单放行
    app.add_middleware(AuthGuardMiddleware)

    # 限流中间件
    app.add_middleware(RateLimitMiddleware)

    # 请求日志在最外层（最先捕获请求，最后记录响应）
    app.add_middleware(RequestLogMiddleware)

    # 注册全局异常处理器
    register_exception_handlers(app)

    rate_limit_info = f"{RATE_LIMIT_PER_MINUTE}/min" if RATE_LIMIT_PER_MINUTE > 0 else "关闭"
    if AUTH_ENFORCE:
        auth_info = "开启(默认拒绝)"
        if not API_KEY and not ADMIN_PASSWORD:
            logger.warning(
                "AUTH_ENFORCE=true 但 API_KEY 与 ADMIN_PASSWORD 都未配置："
                "所有需鉴权的接口都会 401，且无法登录。"
                "请在 .env 里设置 ADMIN_PASSWORD（登录用）或 API_KEY（服务间调用用）；"
                "临时本地调试可设 AUTH_ENFORCE=false"
            )
    else:
        auth_info = "关闭(本地模式)"
        from config import KB_IMPORT_ALLOW_OUTSIDE

        if KB_IMPORT_ALLOW_OUTSIDE:
            logger.info(
                "AUTH_ENFORCE=false：接口无需凭据即可调用。"
                "路径校验仍然生效，但**只限制写入位置**：导入的落点一律在 data/ 之内；"
                "而 /kb/add 允许把 data/ 之外的文件**复制进** data/ 再入库"
                "（KB_IMPORT_ALLOW_OUTSIDE=true）"
            )
            logger.warning(
                "⚠️ 「无鉴权」+「允许从 data/ 之外导入」同时开着："
                "任何能访问本端口的人都可以把机器上可读的文件灌进知识库、"
                "再用 /kb/search 读出来。仅限本机自用；对外部署请设 AUTH_ENFORCE=true，"
                "或设 KB_IMPORT_ALLOW_OUTSIDE=false"
            )
        else:
            logger.info(
                "AUTH_ENFORCE=false：接口无需凭据即可调用。"
                "文件路径校验生效：data/ 之外的文件会被拒绝"
                "（KB_IMPORT_ALLOW_OUTSIDE=false）；对外部署前请设 AUTH_ENFORCE=true"
            )
    logger.info("中间件已加载: 请求日志 | 限流(%s) | 鉴权(%s) | CORS | 全局异常处理",
                rate_limit_info, auth_info)
