"""Week7-3 配置管理单元测试"""

import os
import importlib

import pytest


class TestRequire:
    """_require 函数：必填环境变量"""

    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("TEST_REQUIRED_VAR", "hello")
        from config import _require
        assert _require("TEST_REQUIRED_VAR") == "hello"

    def test_raises_when_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_MISSING_VAR", raising=False)
        from config import _require
        with pytest.raises(EnvironmentError, match="TEST_MISSING_VAR"):
            _require("TEST_MISSING_VAR")

    def test_raises_when_empty(self, monkeypatch):
        monkeypatch.setenv("TEST_EMPTY_VAR", "")
        from config import _require
        with pytest.raises(EnvironmentError):
            _require("TEST_EMPTY_VAR")


class TestOptional:
    """_optional 函数：可选环境变量"""

    def test_returns_value_when_set(self, monkeypatch):
        monkeypatch.setenv("TEST_OPT_VAR", "value")
        from config import _optional
        assert _optional("TEST_OPT_VAR") == "value"

    def test_returns_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("TEST_MISSING_OPT", raising=False)
        from config import _optional
        assert _optional("TEST_MISSING_OPT", "fallback") == "fallback"


class TestConfigValues:
    """配置值类型和默认值"""

    def test_zhipu_api_key_exists(self):
        from config import ZHIPU_API_KEY
        assert ZHIPU_API_KEY is not None
        assert len(ZHIPU_API_KEY) > 0

    def test_zhipu_model_default(self):
        from config import ZHIPU_MODEL
        assert ZHIPU_MODEL == "glm-4-flash"

    def test_llm_temperature_is_float(self):
        from config import LLM_TEMPERATURE
        assert isinstance(LLM_TEMPERATURE, float)
        assert 0.0 <= LLM_TEMPERATURE <= 2.0

    def test_chunk_size_is_int(self):
        from config import CHUNK_SIZE
        assert isinstance(CHUNK_SIZE, int)
        assert CHUNK_SIZE > 0

    def test_max_tool_steps_is_int(self):
        from config import MAX_TOOL_STEPS
        assert isinstance(MAX_TOOL_STEPS, int)
        assert MAX_TOOL_STEPS > 0

    def test_log_level_uppercase(self):
        from config import LOG_LEVEL
        assert LOG_LEVEL == LOG_LEVEL.upper()
        assert LOG_LEVEL in ("DEBUG", "INFO", "WARNING", "ERROR")
