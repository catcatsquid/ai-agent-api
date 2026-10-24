"""模型调用的重试边界

## 为什么需要这个文件

重试原来只写在 `chat_with_tools` 里（非流式那条）；流式那条
（`_iter_stream_events`，也就是 `/qa/stream` 真正走的）**一次都不重试**。
而智谱偶发把请求挂死到 120s 超时是实测过的（`eval/` 的 runner 因此单独加过
传输层重试）—— 于是用户主路径上一次抖动就等于"学生收到一句道歉，而不是答案"。

现在重试统一收在 `llm._post_chat`。这个文件守的三件事：

  1. **首 token 之前会重试**（流式也要重试，否则主路径没有保护）；
  2. **只重试"会自己好"的失败** —— 401 不许重试 3 次白等；
  3. **已经吐过正文之后不许重试** —— 重发会让前端把前半段正文再收一遍。
     这条是最重要的：它把"重试"和"降级"的边界钉死在第一个 chunk 上。
"""
import json

import pytest
import requests
from langchain_core.messages import HumanMessage

from course_agent import llm as llm_mod
from course_agent.llm import (
    SimpleLLM, _iter_stream_events, _is_retryable, _status_of, chat_with_tools,
)

ANSWER = "死锁的四个必要条件是互斥、持有并等待、不可抢占、循环等待。"


# ============================================================
# HTTP 替身
# ============================================================

def _sse(delta: dict) -> str:
    return "data: " + json.dumps({"choices": [{"delta": delta}]}, ensure_ascii=False)


def _lines(text: str) -> list:
    return [_sse({"content": ch}) for ch in text]


class _OkResponse:
    """200 响应：iter_lines 回 **bytes**（生产代码会 decode）"""

    status_code = 200

    def __init__(self, lines=None, payload=None):
        self._lines = lines or []
        self._payload = payload
        self.closed = False

    def raise_for_status(self):
        return None

    def iter_lines(self):
        for line in self._lines:
            yield line.encode("utf-8")

    def close(self):
        self.closed = True

    def json(self):
        return self._payload


class _ErrorResponse:
    """错误响应：raise_for_status 抛带 response 的 HTTPError（与 requests 同形）"""

    def __init__(self, status):
        self.status_code = status
        self.closed = False

    def raise_for_status(self):
        raise requests.HTTPError(
            f"{self.status_code} Client Error: for url: {llm_mod.ZHIPU_CHAT_URL}",
            response=self,
        )

    def close(self):
        self.closed = True


class _ResponseDiesMidStream:
    """200 已经开始吐正文，然后连接断了 —— 这一种**不可重试**"""

    status_code = 200

    def raise_for_status(self):
        return None

    def iter_lines(self):
        yield _sse({"content": "死锁的"}).encode("utf-8")
        raise requests.exceptions.ConnectionError("connection reset by peer")

    def close(self):
        return None


class _BadJsonResponse:
    """200，但响应体不是合法 JSON —— 模拟响应体被截断（网络中断 / 代理断流）"""

    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        raise ValueError("Expecting value: line 1 column 1 (char 0)")

    def close(self):
        return None


def _patch_post(monkeypatch, responses, calls, record=None):
    """按顺序吐出 responses（最后一个会一直复用）；calls 记调用次数与请求体"""
    seq = list(responses)

    def fake_post(url, headers=None, json=None, stream=False, timeout=None):
        calls.append({"url": url, "stream": stream, "payload": json})
        if record is not None:
            record.append(json)
        return seq.pop(0) if len(seq) > 1 else seq[0]

    monkeypatch.setattr(llm_mod.requests, "post", fake_post)
    # 别让测试真的等 3s/6s
    monkeypatch.setattr(llm_mod, "_RATE_LIMIT_WAIT", 0)
    monkeypatch.setattr(llm_mod, "_NETWORK_WAIT", 0)


def _drain(events) -> str:
    return "".join(ev[1] for ev in events if ev[0] == "content")


class _HTTPErrorWith(requests.HTTPError):
    """带 response.status_code 的 HTTPError（requests 的真实形态）"""

    def __init__(self, status):
        super().__init__(f"{status} Client Error")
        self.response = type("R", (), {"status_code": status})()


# ============================================================
# ① 判定函数：什么该重试、什么不该
# ============================================================

class TestRetryable:
    def test_status_code_is_read_from_response(self):
        assert _status_of(_HTTPErrorWith(429)) == 429
        assert _status_of(_HTTPErrorWith(401)) == 401

    def test_only_429_and_5xx_are_retryable(self):
        for code in (429, 500, 502, 503, 504):
            assert _is_retryable(_HTTPErrorWith(code)), f"{code} 应该重试"
        for code in (400, 401, 403, 404):
            assert not _is_retryable(_HTTPErrorWith(code)), f"{code} 不该重试"

    def test_timeout_and_connection_error_are_retryable(self):
        assert _is_retryable(requests.exceptions.Timeout("read timed out"))
        assert _is_retryable(requests.exceptions.ConnectionError("reset"))
        # 解析类错误不算传输失败：服务端返回了坏数据，重发同样坏
        assert not _is_retryable(ValueError("bad json"))


# ============================================================
# ② 首 token 之前：流式也要重试
# ============================================================

class TestStreamRetriesBeforeFirstToken:
    def test_one_429_then_success_still_streams_the_whole_answer(self, monkeypatch):
        calls = []
        _patch_post(monkeypatch, [_ErrorResponse(429), _OkResponse(_lines(ANSWER))], calls)

        text = _drain(_iter_stream_events([{"role": "user", "content": "讲讲死锁"}]))

        assert text == ANSWER, "429 之后应该重试并拿到完整正文"
        assert len(calls) == 2
        assert calls[0]["stream"] is True, "重试必须还是流式请求"

    def test_the_retry_does_not_touch_the_request_body(self, monkeypatch):
        """重试只能重发同一个请求体：planner/synthesizer 不许被塞进工具定义"""
        bodies = []
        calls = []
        _patch_post(monkeypatch, [_ErrorResponse(429), _OkResponse(_lines(ANSWER))],
                    calls, record=bodies)

        _drain(_iter_stream_events([{"role": "user", "content": "讲讲死锁"}]))

        assert len(bodies) == 2
        assert bodies[0] == bodies[1]
        assert "tools" not in bodies[0], "不绑工具时请求体里不能出现 tools"

    def test_429_attempts_are_exhausted_then_raised(self, monkeypatch):
        calls = []
        _patch_post(monkeypatch, [_ErrorResponse(429)], calls)

        with pytest.raises(requests.HTTPError):
            _drain(_iter_stream_events([{"role": "user", "content": "x"}]))

        assert len(calls) == llm_mod._CHAT_ATTEMPTS == 3, "429 应该重试到上限再抛"

    def test_failed_response_is_closed_before_retrying(self, monkeypatch):
        """重试前要放掉上一条连接，否则流式请求的 socket 会挂着"""
        first = _ErrorResponse(429)
        calls = []
        _patch_post(monkeypatch, [first, _OkResponse(_lines(ANSWER))], calls)

        _drain(_iter_stream_events([{"role": "user", "content": "x"}]))

        assert first.closed is True

    def test_simple_llm_stream_goes_through_the_same_retry(self, monkeypatch):
        """经 SimpleLLM._stream 再验一次：LangGraph 走的正是这条路"""
        calls = []
        _patch_post(monkeypatch, [_ErrorResponse(429), _OkResponse(_lines(ANSWER))], calls)

        model = SimpleLLM()
        chunks = list(model._stream([HumanMessage(content="讲讲死锁")]))

        text = "".join(c.message.content for c in chunks)
        assert text == ANSWER
        assert len(calls) == 2


# ============================================================
# ③ 永久性失败：不许重试
# ============================================================

class TestPermanentFailuresAreNotRetried:
    def test_401_fails_on_the_first_attempt(self, monkeypatch):
        calls = []
        _patch_post(monkeypatch, [_ErrorResponse(401)], calls)

        with pytest.raises(requests.HTTPError):
            _drain(_iter_stream_events([{"role": "user", "content": "x"}]))

        assert len(calls) == 1, "401 是密钥配错，重试三次只是白等 4 秒"

    def test_400_fails_on_the_first_attempt(self, monkeypatch):
        calls = []
        _patch_post(monkeypatch, [_ErrorResponse(400)], calls)

        with pytest.raises(requests.HTTPError):
            _drain(_iter_stream_events([{"role": "user", "content": "x"}]))

        assert len(calls) == 1


# ============================================================
# ④ 首 token 之后：绝不许重发（这条是"重试 vs 降级"的分界）
# ============================================================

class TestNoRetryAfterFirstChunk:
    def test_mid_stream_failure_is_raised_not_retried(self, monkeypatch):
        calls = []
        _patch_post(monkeypatch, [_ResponseDiesMidStream()], calls)

        seen = []
        with pytest.raises(requests.exceptions.ConnectionError):
            for event in _iter_stream_events([{"role": "user", "content": "x"}]):
                seen.append(event)

        assert "死锁的" in "".join(ev[1] for ev in seen), "断流前已经吐出的正文不该被吞"
        assert len(calls) == 1, "已经开始 yield 就不许重发 —— 否则前端会收到两遍正文"


# ============================================================
# ⑤ 非流式路径共用同一套
# ============================================================

class TestNonStreamPathSharesTheBoundary:
    def test_chat_with_tools_retries_5xx_and_keeps_tool_calls(self, monkeypatch):
        calls = []
        payload = {"choices": [{"message": {
            "content": "",
            "tool_calls": [{"id": "call_1", "type": "function",
                            "function": {"name": "search_knowledge",
                                         "arguments": "{\"query\": \"死锁\"}"}}],
        }}]}
        _patch_post(monkeypatch, [_ErrorResponse(503), _OkResponse([], payload=payload)], calls)

        result = chat_with_tools([{"role": "user", "content": "x"}],
                                 tools=[{"type": "function",
                                         "function": {"name": "search_knowledge"}}])

        assert result["tool_calls"][0]["id"] == "call_1"
        assert len(calls) == 2

    def test_chat_with_tools_does_not_retry_401(self, monkeypatch):
        calls = []
        _patch_post(monkeypatch, [_ErrorResponse(401)], calls)

        with pytest.raises(requests.HTTPError):
            chat_with_tools([{"role": "user", "content": "x"}])

        assert len(calls) == 1


# ============================================================
# ⑥ 200 但响应体坏了：再试一次（多为响应体被截断）
# ============================================================

class TestBrokenBodyIsRetriedOnce:
    def test_broken_body_is_retried_once(self, monkeypatch):
        calls = []
        ok = {"choices": [{"message": {"content": "答复", "tool_calls": None}}]}
        _patch_post(monkeypatch, [_BadJsonResponse(), _OkResponse([], payload=ok)], calls)

        result = chat_with_tools([{"role": "user", "content": "x"}])

        assert result["content"] == "答复"
        assert len(calls) == 2, "200 但 body 坏了（多半是被截断）应该再试一次"

    def test_broken_body_gives_up_after_one_retry(self, monkeypatch):
        calls = []
        _patch_post(monkeypatch, [_BadJsonResponse()], calls)

        with pytest.raises(ValueError):
            chat_with_tools([{"role": "user", "content": "x"}])

        assert len(calls) == 2, "只补一次：坏数据重发三次还是坏"
