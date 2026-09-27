"""Week7-1 错误处理单元测试"""

import pytest
import requests
from unittest.mock import MagicMock

from course_agent.errors import (
    APIError, RateLimitError, LLMError, EmbeddingError, ToolError,
    retry_on_rate_limit, safe_llm_invoke, safe_tool_invoke,
)


class TestErrorClasses:
    """异常类定义"""

    def test_api_error_is_exception(self):
        assert issubclass(APIError, Exception)

    def test_rate_limit_is_api_error(self):
        assert issubclass(RateLimitError, APIError)

    def test_llm_error_is_exception(self):
        assert issubclass(LLMError, Exception)

    def test_embedding_error_is_exception(self):
        assert issubclass(EmbeddingError, Exception)

    def test_tool_error_is_exception(self):
        assert issubclass(ToolError, Exception)


class TestRetryOnRateLimit:
    """retry_on_rate_limit 装饰器"""

    def test_retry_succeeds_on_429(self):
        call_count = {"n": 0}

        @retry_on_rate_limit(max_retries=3, base_wait=0.01)
        def mock_api():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise requests.exceptions.HTTPError("429 Too Many Requests")
            return "ok"

        assert mock_api() == "ok"
        assert call_count["n"] == 3

    def test_retry_exhausted_raises(self):
        @retry_on_rate_limit(max_retries=2, base_wait=0.01)
        def always_429():
            raise requests.exceptions.HTTPError("429 Too Many Requests")

        with pytest.raises(requests.exceptions.HTTPError):
            always_429()

    def test_retry_on_timeout(self):
        call_count = {"n": 0}

        @retry_on_rate_limit(max_retries=3, base_wait=0.01)
        def mock_timeout():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise requests.exceptions.Timeout("timed out")
            return "recovered"

        assert mock_timeout() == "recovered"
        assert call_count["n"] == 2

    def test_no_retry_on_other_errors(self):
        call_count = {"n": 0}

        @retry_on_rate_limit(max_retries=3, base_wait=0.01)
        def raises_value_error():
            call_count["n"] += 1
            raise ValueError("not a 429 or timeout")

        with pytest.raises(ValueError):
            raises_value_error()
        assert call_count["n"] == 1


class TestSafeLLMInvoke:
    """safe_llm_invoke"""

    def test_success(self):
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = "response"
        result = safe_llm_invoke(mock_llm, [], context="test")
        assert result == "response"

    def test_failure_raises_llm_error(self):
        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = TimeoutError("LLM timeout")
        with pytest.raises(LLMError) as exc_info:
            safe_llm_invoke(mock_llm, [], context="planner")
        assert "planner" in str(exc_info.value)
        assert "LLM timeout" in str(exc_info.value)


class TestSafeToolInvoke:
    """safe_tool_invoke"""

    def test_success(self):
        mock_tool = MagicMock()
        mock_tool.invoke.return_value = "result"
        result = safe_tool_invoke(mock_tool, {"query": "test"}, tool_name="search_knowledge")
        assert result == "result"

    def test_failure_returns_error_string(self):
        """工具失败时不抛异常，返回错误信息字符串"""
        mock_tool = MagicMock()
        mock_tool.invoke.side_effect = ValueError("crashed")
        result = safe_tool_invoke(mock_tool, {"query": "test"}, tool_name="search_knowledge")
        assert isinstance(result, str)
        assert "[工具错误]" in result
        assert "search_knowledge" in result
        assert "crashed" in result
