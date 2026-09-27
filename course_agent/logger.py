"""结构化日志系统

替代 print，提供分级日志输出：
  DEBUG   -> 详细调试信息（变量值、中间状态）
  INFO    -> 正常流程节点（加载完成、入库完成）
  WARNING -> 可恢复异常（跳过文件、限流重试）
  ERROR   -> 不可恢复异常（API失败、LLM错误）

配置方式（环境变量）：
  LOG_LEVEL=DEBUG   设置日志级别（默认 INFO）
  LOG_FILE=logs/agent.log  同时写入文件（默认仅控制台）

Trace ID 支持:
  用 set_trace_id() 设置当前请求的 trace_id，之后的日志会自动带上。
  用在中间件里，每个请求进来设置一次，请求结束清理。
"""

import os
import logging
import contextvars
from pathlib import Path

from config import LOG_LEVEL, LOG_FILE

# 上下文变量：存当前请求的 trace_id
# contextvars 是线程安全、协程安全的，每个请求有自己的值
_trace_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("trace_id", default="")


def set_trace_id(trace_id: str) -> None:
    """设置当前请求的 trace_id，之后日志会自动带上"""
    _trace_id_var.set(trace_id)


def get_trace_id() -> str:
    """获取当前 trace_id，没有就返回空字符串"""
    return _trace_id_var.get()


def clear_trace_id() -> None:
    """清理 trace_id（请求结束时调用）"""
    _trace_id_var.set("")


_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[35m",  # magenta
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """带颜色 + trace_id 的日志 Formatter"""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        levelname = f"{color}{record.levelname:<8}{_RESET}"
        module = record.name.split(".")[-1] if "." in record.name else record.name

        # 有 trace_id 就带上，没有就不显示
        trace_id = get_trace_id()
        trace_part = f"[{trace_id[:8]}] " if trace_id else ""

        return (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} "
            f"| {levelname} "
            f"| {trace_part}"
            f"{module:<16} "
            f"| {record.getMessage()}"
        )


class PlainFormatter(logging.Formatter):
    """不带颜色的 Formatter（写文件用），也带 trace_id"""

    def format(self, record: logging.LogRecord) -> str:
        trace_id = get_trace_id()
        trace_part = f"[{trace_id[:8]}] " if trace_id else ""
        return (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} "
            f"| {record.levelname:<8s} "
            f"| {trace_part}"
            f"{record.name:<16s} "
            f"| {record.getMessage()}"
        )


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("agent")
    logger.setLevel(LOG_LEVEL)
    logger.propagate = False

    if logger.handlers:
        return logger

    console = logging.StreamHandler()
    console.setLevel(LOG_LEVEL)
    console.setFormatter(ColorFormatter())
    logger.addHandler(console)

    if LOG_FILE:
        Path(LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(LOG_LEVEL)
        file_handler.setFormatter(
            PlainFormatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


logger = _build_logger()
