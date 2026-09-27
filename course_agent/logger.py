"""结构化日志系统

替代 print，提供分级日志输出：
  DEBUG   -> 详细调试信息（变量值、中间状态）
  INFO    -> 正常流程节点（加载完成、入库完成）
  WARNING -> 可恢复异常（跳过文件、限流重试）
  ERROR   -> 不可恢复异常（API失败、LLM错误）

配置方式（环境变量）：
  LOG_LEVEL=DEBUG   设置日志级别（默认 INFO）
  LOG_FILE=logs/agent.log  同时写入文件（默认仅控制台）
"""

import os
import logging
from pathlib import Path

from config import LOG_LEVEL, LOG_FILE

_LEVEL_COLORS = {
    logging.DEBUG: "\033[36m",     # cyan
    logging.INFO: "\033[32m",       # green
    logging.WARNING: "\033[33m",   # yellow
    logging.ERROR: "\033[31m",     # red
    logging.CRITICAL: "\033[35m",  # magenta
}
_RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """带颜色区分日志级别的 Formatter"""

    def format(self, record: logging.LogRecord) -> str:
        color = _LEVEL_COLORS.get(record.levelno, "")
        levelname = f"{color}{record.levelname:<8}{_RESET}"
        module = record.name.split(".")[-1] if "." in record.name else record.name
        return (
            f"{self.formatTime(record, '%Y-%m-%d %H:%M:%S')} "
            f"| {levelname} "
            f"| {module:<16} "
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
            logging.Formatter(
                "%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(file_handler)

    return logger


logger = _build_logger()
