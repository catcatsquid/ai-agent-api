"""LLM 配置 - 智谱 GLM-4-Flash（直接 HTTP 调用，不依赖 zhipuai SDK）

提供:
- llm: **真正的 LangChain ChatModel**（BaseChatModel 子类，给 nodes.py 用）
- stream_chat / chat: 直接流式/非流式调用
- chat_with_tools: 支持 function calling 的非流式调用

Week 4/5/7 的 Agent 都用这个。

## 重试只有一处（`_post_chat`）

所有对智谱 chat 端点的请求都经过 `_post_chat`：429/5xx/超时/断连重试，
401/400 立即抛；**流式只在第一个 chunk 之前可重试**（这条边界由生成器的
惰性执行天然保证）。别再在调用方另包一层循环 —— 见下方"统一调用边界"。
回归测试见 `tests/test_llm_retry.py`。

## 为什么 llm 必须是 BaseChatModel（1.7.2 修的真问题）

原来是"看起来像 ChatModel"的普通类：只有 `invoke`/`bind_tools`，外加一个
**假的** `astream`（拿非流式结果逐字切片充数）。后果是 `/qa/stream` 的
`content` 事件在生产里**从未触发过** —— LangGraph 的 `stream_mode="messages"`
只对**实现了 `_stream` 的 Runnable** 吐 `AIMessageChunk`；普通类走的是
"节点把完整 AIMessage 返回再推一遍"那条路，被 `sse_content_event` 的
"只放行 chunk"判据正确挡掉（这是对的，放行就会重复两遍）。

于是学生等约 35 秒，然后整段答案一次性出现 —— 而 674 条测试全绿，
因为流式相关用例用的是 `GenericFakeChatModel`（真 LangChain 模型），
从没跑过生产 LLM。回归测试见 `tests/test_llm_streaming.py`。
"""

import json
import re
import time
from typing import Iterator

import requests
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL, ZHIPU_MODEL, LLM_TEMPERATURE
from course_agent.logger import logger

_headers = {
    "Authorization": f"Bearer {ZHIPU_API_KEY}",
    "Content-Type": "application/json",
}


# ============================================================
# 统一调用边界：重试只发生在"建连 + 状态码"阶段
# ============================================================
#
# ## 为什么要把重试收进**一个**函数
#
# 重试原来只写在 `chat_with_tools` 里（非流式那条路）；流式那条
# （`_iter_stream_events`，也就是 `/qa/stream` 真正走的）**一次都不重试**。
# 而智谱偶发把请求挂死到 120s 超时是实测过的（eval runner 因此单独加过传输层重试）——
# 于是用户主路径上一次抖动就等于"学生收到一句道歉，而不是答案"。
#
# ## 重试的**安全边界**：第一个 chunk 之前
#
# `_iter_stream_events` 是生成器：`requests.post` 直到第一次 next() 才执行。
# 所以把重试放在 `_post_chat` 里，天然只覆盖"还没吐过正文"的阶段。
# **一旦开始 yield 就不再重试** —— 重发会让前端把前半段正文再收一遍。
# 首 token 之后断流只能由节点降级处理，不归这一层管（见 nodes 的 LLMError 兜底）。
#
# ## 只重试"会自己好"的失败
#
# 429 / 5xx / 超时 / 断连 → 重试；401（密钥错）、400（请求体错）→ 立即抛。
# 原来 `chat_with_tools` 的 `except Exception` 连 401 都会重试 3 次，
# 密钥配错时要白发三个请求、白等 4 秒才报错。
_RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

_CHAT_ATTEMPTS = 3        # 含首次
_RATE_LIMIT_WAIT = 3.0    # 429: 3s → 6s（与 errors.retry_on_rate_limit 同口径）
_NETWORK_WAIT = 2.0       # 超时/断连：固定 2s


def _status_of(exc: Exception):
    """尽力取 HTTP 状态码；取不到返回 None（= 不按状态码判断是否重试）"""
    resp = getattr(exc, "response", None)
    code = getattr(resp, "status_code", None)
    if isinstance(code, int):
        return code
    # 没有 response 的异常（测试替身、requests 的少数路径）退回消息里的状态码
    m = re.search(r"\b(429|500|502|503|504)\b", str(exc))
    return int(m.group(1)) if m else None


def _is_retryable(exc: Exception) -> bool:
    """这个异常重发一次有没有可能成功"""
    if isinstance(exc, (requests.exceptions.Timeout,
                        requests.exceptions.ConnectionError)):
        return True
    if isinstance(exc, requests.exceptions.HTTPError):
        return _status_of(exc) in _RETRYABLE_STATUS
    return False


def _post_chat(payload: dict, *, stream: bool = False, timeout: int = 120):
    """POST 到智谱 chat 端点，返回**已通过 raise_for_status** 的 response

    只在这一层重试（`_CHAT_ATTEMPTS` 次），调用方不需要也不应该再包一层循环。
    同步函数里的 `time.sleep` 会占住调用线程（LangGraph 的工作线程），
    最坏 3s + 6s —— 这是有意的：限流退避本来就不该并发再打过去。
    """
    for attempt in range(_CHAT_ATTEMPTS):
        resp = None
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=_headers, json=payload,
                                 stream=stream, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if resp is not None:
                close = getattr(resp, "close", None)
                if close is not None:
                    try:
                        close()
                    except Exception:   # noqa: BLE001 —— 关连接失败不影响重试判断
                        pass
            if not _is_retryable(e) or attempt == _CHAT_ATTEMPTS - 1:
                raise
            wait = (_RATE_LIMIT_WAIT * (2 ** attempt)
                    if _status_of(e) == 429 else _NETWORK_WAIT)
            logger.warning("模型调用失败(%s: %s)，%.0fs 后重试（%d/%d）",
                           type(e).__name__, str(e)[:80], wait,
                           attempt + 1, _CHAT_ATTEMPTS - 1)
            time.sleep(wait)
    # 循环只会以上面的 return 或 raise 结束；这一行只为显式闭合控制流
    raise RuntimeError("unreachable")   # pragma: no cover


# 200 但 body 坏了时再试一次。为什么不并进 `_post_chat`：那一层只管"建连 + 状态码"，
# 解析是调用方的事；而这一类失败值得重试，因为 200 + 非 JSON 最常见的原因是
# **响应体被截断**（网络中断、代理断流），属于瞬时故障，而且它失败得很快，
# 不会像读超时那样占满 120s。只补一次：坏数据重发三次还是坏，截断重发一次基本就对了。
#
# 历史：旧的 `chat_with_tools` 用 `except Exception` 把解析错误一起重试了；
# 收编到 `_post_chat` 时漏掉这一类，等于把"体被截断"从可重试变成了直接失败。
_PARSE_ATTEMPTS = 2


def _post_json(payload: dict) -> dict:
    """发请求并解析 JSON body；200 但响应体不是合法 JSON 时再试一次"""
    last = None
    for attempt in range(_PARSE_ATTEMPTS):
        resp = _post_chat(payload)
        try:
            return resp.json()
        except ValueError as e:     # json.JSONDecodeError 是 ValueError 的子类
            last = e
            if attempt < _PARSE_ATTEMPTS - 1:
                logger.warning("模型响应不是合法 JSON（%s），重试一次", str(e)[:80])
    raise last


def chat(messages, temperature=None, max_tokens=None):
    """非流式调用智谱 API，返回完整文本"""
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    return _post_json(payload)["choices"][0]["message"]["content"]


def stream_chat(messages, temperature=None):
    """流式调用智谱 API，yield content 字符串"""
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
    }
    resp = _post_chat(payload, stream=True)
    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                chunk = json.loads(line[6:])
                content = chunk["choices"][0].get("delta", {}).get("content", "")
                if content:
                    yield content
            except (json.JSONDecodeError, KeyError, IndexError):
                continue


def chat_with_tools(messages, tools=None, temperature=None, max_tokens=2000,
                    tool_choice="auto"):
    """调用智谱 API，支持 function calling。非流式。

    重试分两段：建连/状态码在 `_post_chat`（429/5xx/超时最多 3 次，401/400 立即抛）；
    "200 但响应体坏了"在 `_post_json`（再试一次）。
    这里**不再自带循环**：以前那层 `except Exception` 会把 401 也重试 3 次。

    Returns:
        dict with keys:
          - content: str (可能为空)
          - tool_calls: list of {id, name, arguments} (可能为 None)
    """
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
        "max_tokens": max_tokens,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    # 建连/状态码的重试在 _post_chat；"200 但 body 坏了"的重试在 _post_json（各管一段）
    msg = _post_json(payload)["choices"][0]["message"]
    return {
        "content": msg.get("content", ""),
        "tool_calls": msg.get("tool_calls"),
    }


# ============================================================
# 兼容 LangChain ChatModel 接口的封装
# （给 nodes.py 用，graph.py/nodes.py 无需改动）
# ============================================================

def _iter_stream_events(api_messages, tools=None, temperature=None,
                        tool_choice="auto"):
    """流式调智谱 API，yield 两种事件：

      ("content", str)                     —— 正文增量（token）
      ("tool_call", index, id, name, args) —— 工具调用的**增量**（可能被拆成多片）

    tools 为空时**不发** `tools` 字段：不绑工具的那几次调用（planner/synthesizer）
    必须与改前逐字节相同，绝不能因为改流式而多带上工具定义。
    """
    payload = {
        "model": ZHIPU_MODEL,
        "messages": api_messages,
        "stream": True,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
    }
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    # 只重试"建连 + 状态码"阶段：本函数是生成器，post 到第一次 next() 才执行，
    # 所以这里天然满足"还没吐过正文才能重试"。开吐之后断流不重试（见文件头说明）。
    resp = _post_chat(payload, stream=True)

    for line in resp.iter_lines():
        if not line:
            continue
        line = line.decode("utf-8")
        if not line.startswith("data: ") or line == "data: [DONE]":
            continue
        try:
            chunk = json.loads(line[6:])
            delta = chunk["choices"][0].get("delta", {}) or {}
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

        content = delta.get("content")
        if content:
            yield ("content", content)
        for tc in (delta.get("tool_calls") or []):
            fn = tc.get("function") or {}
            yield ("tool_call", tc.get("index", 0), tc.get("id"),
                   fn.get("name"), fn.get("arguments"))


class SimpleLLM(BaseChatModel):
    """真正的 LangChain ChatModel（`BaseChatModel` 子类），直连智谱 HTTP

    接口与 ChatZhipuAI 兼容，nodes.py 无需改动。流式与非流式**共用同一套
    HTTP 端点**，区别只在 `stream: true`：

      · `_generate` → 非流式（`chat_with_tools`），一次性拿完整 AIMessage
      · `_stream`   → 流式，逐片 yield `ChatGenerationChunk`

    `stream_mode="messages"` 只认 `_stream`，所以 `/qa/stream` 的逐字 `content`
    事件靠的是后者（见模块顶部"为什么 llm 必须是 BaseChatModel"）。
    `stream()` 不会被节点直接调用，所以 `_generate` 现在只在"`_stream` 不可用
    （NotImplementedError）"或别的代码直接 `generate()` 时才会走到 —— 保留它
    是为了与非流式调用方保持行为一致。
    """

    def __init__(self, tools=None, tool_choice="auto", **kwargs):
        super().__init__(**kwargs)
        # "auto" 让模型自己决定；"required" 强制这一轮必须发起工具调用
        # （ReAct 的"嘴上说要调却没调"补一轮时用）
        self._tool_choice = tool_choice
        # 已转成智谱 function 格式的工具定义；None = 这次调用不带工具
        self._tools = [_tool_to_zhipu_format(t) for t in tools] if tools else None

    @property
    def _llm_type(self) -> str:
        return "zhipu-glm-http"

    def bind_tools(self, tools, **kwargs):
        """绑定工具，返回**新实例**（`BaseChatModel.bind_tools` 是抽象方法，直接 raise）

        为什么必须 copy 而不是就地改写：`llm` 是全局单例。就地塞进工具定义后，
        planner / synthesizer 这些本该发纯文本请求的节点也会带上全部工具 schema，
        模型于是可能返回 tool_calls 而 content 为空 —— planner 静默退化到兜底计划
        （这是修过的真实缺陷，见 1.5.x 的 CHANGELOG）。

        不用 `self.bind(tools=…)`（RunnableBinding + 调用时注入 kwargs）的原因：
        `_generate`/`_stream` 拿到的 messages 是 LangChain 归一化后的**新对象**，
        而现有实现与测试都是按"直接看 messages"写的；换成注入式会同时改掉
        非流式与流式两条路的消息形态，风险远大于收益。这里只多一个 _tools 字段。
        """
        import copy

        bound = copy.copy(self)
        bound._tools = [_tool_to_zhipu_format(t) for t in tools] if tools else None
        return bound

    def with_tool_choice(self, choice: str):
        """返回一个"强制/放开工具调用"的新实例（不改动这个对象）

        ReAct 用它在"文字里说要调、实际没调"时补一轮 `tool_choice="required"`。
        """
        import copy

        bound = copy.copy(self)
        bound._tool_choice = choice
        return bound

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        """非流式调用：一次性拿完整结果"""
        api_messages = _lc_messages_to_zhipu(messages)
        result = chat_with_tools(api_messages, tools=self._tools,
                                 tool_choice=self._tool_choice)
        return ChatResult(generations=[
            ChatGeneration(message=_zhipu_response_to_ai_message(result))])

    def _stream(self, messages, stop=None, run_manager=None,
                **kwargs) -> Iterator[ChatGenerationChunk]:
        """流式调用，逐片产出

        ## 工具调用也要一起流出去（不能只顾正文）

        ReAct 图靠**工具调用**往前走。`_stream` 若只吐 content，工具调用就丢了，
        ReAct 会静默退化成"什么工具都不调"。所以 `tool_call` 事件转成
        `AIMessageChunk(tool_call_chunks=[...])` —— LangGraph 会把同一 index 的
        分片合并回完整的 `tool_calls`，最终 AIMessage 与**非流式那条路完全一致**。

        工具调用的参数常常被拆成多片（index 相同、arguments 各是一段字符串），
        这里原样转出去，由 LangChain 的合并逻辑拼接 —— 自己拼容易漏掉 id/name
        只在第一片出现这回事（`tests/test_llm_streaming.py` 里有一条专门守它）。
        """
        api_messages = _lc_messages_to_zhipu(messages)
        tools = self._tools
        # temperature 从 kwargs 透传（bind_tools(..., temperature=…) 这类用法）
        temperature = kwargs.get("temperature")

        for event in _iter_stream_events(api_messages, tools=tools,
                                         temperature=temperature,
                                         tool_choice=self._tool_choice):
            if event[0] == "content":
                yield ChatGenerationChunk(message=AIMessageChunk(content=event[1]))
            else:
                _, index, tc_id, name, args = event
                yield ChatGenerationChunk(message=AIMessageChunk(
                    content="",
                    tool_call_chunks=[{
                        "name": name,
                        "args": args,
                        "id": tc_id,
                        "index": index,
                        "type": "tool_call_chunk",
                    }],
                ))


def _tool_to_zhipu_format(tool) -> dict:
    """把 LangChain Tool 转成智谱 API 的 function 格式"""
    # 如果已经是 dict，直接返回
    if isinstance(tool, dict):
        return tool
    # LangChain Tool 对象
    name = getattr(tool, "name", str(tool))
    description = getattr(tool, "description", "")
    # 从 args_schema 或直接获取参数 schema
    args_schema = getattr(tool, "args_schema", None)
    if args_schema:
        try:
            parameters = args_schema.model_json_schema()
        except Exception:
            parameters = {"type": "object", "properties": {}, "required": []}
    else:
        # 尝试从 get_input_schema 获取
        try:
            parameters = tool.get_input_schema().schema()
        except Exception:
            parameters = {"type": "object", "properties": {}, "required": []}
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": parameters,
        },
    }


def _lc_messages_to_zhipu(messages) -> list:
    """把 LangChain 消息列表转成智谱 API 原生格式"""
    api_messages = []
    for m in messages:
        role = getattr(m, "type", None)
        if role == "human":
            api_messages.append({"role": "user", "content": m.content})
        elif role == "ai":
            msg = {"role": "assistant", "content": m.content}
            if hasattr(m, "tool_calls") and m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"], ensure_ascii=False)
                                if isinstance(tc["args"], dict) else str(tc["args"]),
                        },
                    }
                    for tc in m.tool_calls
                ]
            api_messages.append(msg)
        elif role == "tool":
            tool_name = getattr(m, "name", "") or "tool"
            api_messages.append({
                "role": "tool",
                "tool_call_id": m.tool_call_id,
                "name": tool_name,
                "content": m.content,
            })
        elif role == "system":
            api_messages.append({"role": "system", "content": m.content})
        else:
            api_messages.append({"role": "user", "content": str(m.content)})
    return api_messages


def _zhipu_response_to_ai_message(result: dict):
    """把智谱 API 响应转成 LangChain AIMessage"""
    ai_kwargs = {"content": result.get("content", "")}
    if result.get("tool_calls"):
        tool_calls = []
        for tc in result["tool_calls"]:
            args = tc["function"]["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            tool_calls.append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "args": args,
            })
        ai_kwargs["tool_calls"] = tool_calls
    return AIMessage(**ai_kwargs)


# 全局实例，供 nodes.py 导入使用。
# 绑定工具走下面那个自制的 `bind_tools`（copy 一份，不改写单例）—— 原因是
# `BaseChatModel.bind_tools` 在 langchain-core 1.6.x 里是**抽象方法**（直接 raise
# NotImplementedError），基类那条路走不通，必须自己实现。
llm = SimpleLLM()
