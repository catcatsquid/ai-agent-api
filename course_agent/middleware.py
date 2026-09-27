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
from collections import defaultdict
from threading import Lock

from fastapi import Request, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.cors import CORSMiddleware

from config import RATE_LIMIT_PER_MINUTE
from course_agent.logger import logger


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
        # 记录开始时间
        start_time = time.time()
        client_ip = request.client.host if request.client else "unknown"
        method = request.method
        path = request.url.path

        # 跳过健康检查的日志（太频繁了）
        skip_log = path in ["/", "/favicon.ico", "/docs", "/openapi.json"]

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

        # 给响应加一个 X-Response-Time 头（方便调试）
        response.headers["X-Response-Time"] = f"{duration_ms:.0f}"
        return response


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

        # 健康检查和文档页面不限流
        if path in ["/", "/favicon.ico", "/docs", "/openapi.json"]:
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
# 4. CORS 中间件配置
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
# 5. 一键注册所有中间件
# ============================================================

def setup_middlewares(app):
    """一键注册所有中间件

    注意：中间件的顺序很重要！
    先注册的在外层（请求时最后经过，响应时最先经过）

    当前顺序:
      请求 → 请求日志 → 限流 → CORS → 路由
      响应 ← 请求日志 ← 限流 ← CORS ← 路由
    """
    # CORS 应该在最内层（最先处理请求，最后处理响应）
    setup_cors(app)

    # 限流中间件
    app.add_middleware(RateLimitMiddleware)

    # 请求日志在最外层（最先捕获请求，最后记录响应）
    app.add_middleware(RequestLogMiddleware)

    # 注册全局异常处理器
    register_exception_handlers(app)

    rate_limit_info = f"{RATE_LIMIT_PER_MINUTE}/min" if RATE_LIMIT_PER_MINUTE > 0 else "关闭"
    logger.info("中间件已加载: 请求日志 | 限流(%s) | CORS | 全局异常处理", rate_limit_info)
