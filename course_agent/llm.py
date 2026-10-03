"""LLM 配置 - 智谱 GLM-4-Flash（直接 HTTP 调用，不依赖 zhipuai SDK）

提供:
- llm: 兼容 LangChain 的 chat model 接口（给 nodes.py 用）
- stream_chat / chat: 直接流式/非流式调用
- chat_with_tools: 支持 function calling 的非流式调用

Week 4/5/7 的 Agent 都用这个。
"""

import json
import time
import requests

from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL, ZHIPU_MODEL, LLM_TEMPERATURE

_headers = {
    "Authorization": f"Bearer {ZHIPU_API_KEY}",
    "Content-Type": "application/json",
}


def chat(messages, temperature=None, max_tokens=None):
    """非流式调用智谱 API，返回完整文本"""
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
    }
    if max_tokens:
        payload["max_tokens"] = max_tokens
    resp = requests.post(ZHIPU_CHAT_URL, headers=_headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def stream_chat(messages, temperature=None):
    """流式调用智谱 API，yield content 字符串"""
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": temperature if temperature is not None else LLM_TEMPERATURE,
    }
    resp = requests.post(ZHIPU_CHAT_URL, headers=_headers, json=payload, stream=True, timeout=120)
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


def chat_with_tools(messages, tools=None, temperature=None, max_tokens=2000):
    """调用智谱 API，支持 function calling。非流式。

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
        payload["tool_choice"] = "auto"

    for attempt in range(3):
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=_headers, json=payload, timeout=120)
            resp.raise_for_status()
            msg = resp.json()["choices"][0]["message"]
            result = {
                "content": msg.get("content", ""),
                "tool_calls": msg.get("tool_calls"),
            }
            return result
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(3 * (attempt + 1))
                continue
            if attempt < 2:
                time.sleep(2)
                continue
            raise


# ============================================================
# 兼容 LangChain ChatModel 接口的封装
# （给 nodes.py 用，graph.py/nodes.py 无需改动）
# ============================================================

class SimpleLLM:
    """极简的 LangChain 兼容 ChatModel，只实现 invoke 和 bind_tools

    接口与 ChatZhipuAI 兼容，nodes.py 直接替换即可。
    """

    def __init__(self):
        self._tools = None

    def bind_tools(self, tools):
        """绑定工具（LangChain 风格）

        返回**新实例**而不是 self：全局单例 llm 被 bind_tools 就地改写后，
        planner / synthesizer 这些本该发纯文本请求的节点也会带上全部工具定义，
        模型于是可能返回 tool_calls 而 content 为空，planner 静默退化到兜底计划。
        """
        import copy

        bound = copy.copy(self)
        bound._tools = [_tool_to_zhipu_format(t) for t in tools] if tools else None
        return bound

    def invoke(self, messages, config=None):
        """调用 LLM，返回 AIMessage 兼容对象"""
        api_messages = _lc_messages_to_zhipu(messages)
        result = chat_with_tools(api_messages, tools=self._tools)
        return _zhipu_response_to_ai_message(result)

    async def astream(self, messages, config=None):
        """异步流式（用同步模拟，够用就行）"""
        api_messages = _lc_messages_to_zhipu(messages)
        content = chat(api_messages)
        for ch in content:
            yield ch


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
    from langchain_core.messages import AIMessage
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


# 全局实例，供 nodes.py 导入使用
llm = SimpleLLM()
