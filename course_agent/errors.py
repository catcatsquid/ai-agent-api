"""全局错误处理与重试机制

错误分类:
  - APIError: API 调用失败（超时、网络错误、HTTP 非 200）
  - RateLimitError: 429 限流，需要等待后重试
  - LLMError: LLM 调用失败
  - EmbeddingError: Embedding 调用失败
  - ToolError: 工具执行失败

重试策略:
  - 429 限流: 指数退避（3s → 6s → 12s），最多 3 次
  - 超时/网络错误: 固定间隔 2s，最多 2 次
  - 其他错误: 不重试，直接抛出
"""

import time
import functools
import requests
from typing import Callable, TypeVar

from course_agent.logger import logger

T = TypeVar("T")


class APIError(Exception):
    """API 调用通用错误"""
    pass


class RateLimitError(APIError):
    """429 限流错误"""
    pass


class LLMError(Exception):
    """LLM 调用失败"""
    pass


class EmbeddingError(Exception):
    """Embedding 调用失败"""
    pass


class ToolError(Exception):
    """工具执行失败"""
    pass


def retry_on_rate_limit(max_retries: int = 3, base_wait: float = 3.0):
    """装饰器：429 限流时指数退避重试

    Args:
        max_retries: 最大重试次数
        base_wait: 第一次等待秒数，后续翻倍（3 → 6 → 12）
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error = None
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (RateLimitError, requests.exceptions.HTTPError) as e:
                    last_error = e
                    if "429" in str(e) and attempt < max_retries - 1:
                        wait = base_wait * (2 ** attempt)
                        logger.warning("限流重试 %s 第%d次失败，等待%.1fs", func.__name__, attempt + 1, wait)
                        time.sleep(wait)
                    else:
                        raise
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        wait = 2.0
                        logger.warning("网络重试 %s 超时/断连，等待%.1fs", func.__name__, wait)
                        time.sleep(wait)
                    else:
                        raise
            raise last_error
        return wrapper
    return decorator


def safe_llm_invoke(llm, messages, context: str = ""):
    """安全的 LLM 调用，捕获超时和限流

    Args:
        llm: LangChain LLM 实例
        messages: 消息列表
        context: 调用上下文（用于错误日志），如 "planner" / "synthesizer"

    Returns:
        LLM 响应对象

    Raises:
        LLMError: 调用失败时抛出
    """
    try:
        return llm.invoke(messages)
    except Exception as e:
        err_msg = f"LLM 调用失败 (context={context}): {type(e).__name__}: {str(e)[:200]}"
        logger.error(err_msg)
        raise LLMError(err_msg) from e


def safe_tool_invoke(tool, args: dict, tool_name: str = ""):
    """安全的工具调用，捕获执行异常

    Args:
        tool: LangChain 工具对象
        args: 参数字典
        tool_name: 工具名（用于错误日志）

    Returns:
        工具执行结果字符串（失败时返回错误提示而非抛异常，不中断 Agent 流程）

    设计决策: 工具失败时不抛异常，而是返回错误信息字符串。
    因为 Agent 可能调多个工具，一个失败不应中断其他工具的执行。
    """
    try:
        return tool.invoke(args)
    except Exception as e:
        err_msg = f"[工具错误] {tool_name or tool.name}: {type(e).__name__}: {str(e)[:150]}"
        logger.error(err_msg)
        return err_msg
