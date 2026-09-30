"""LLM 配置 - 智谱 GLM-4-Flash

LangChain 的 ChatZhipuAI 用于 Agent 工具调用；
stream_chat / chat 用于学习/复习模式的直接流式输出。
"""

import json
import requests

from langchain_community.chat_models import ChatZhipuAI

from config import ZHIPU_API_KEY, ZHIPU_MODEL, LLM_TEMPERATURE, ZHIPU_CHAT_URL

llm = ChatZhipuAI(
    model=ZHIPU_MODEL,
    api_key=ZHIPU_API_KEY,
    temperature=LLM_TEMPERATURE,
)

_headers = {
    "Authorization": f"Bearer {ZHIPU_API_KEY}",
    "Content-Type": "application/json",
}


def stream_chat(messages, temperature=0.7):
    """流式调用智谱 API，yield content 字符串"""
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "stream": True,
        "temperature": temperature,
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


def chat(messages, temperature=0.7):
    """非流式调用智谱 API，返回完整文本"""
    payload = {
        "model": ZHIPU_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    resp = requests.post(ZHIPU_CHAT_URL, headers=_headers, json=payload, timeout=60)
    return resp.json()["choices"][0]["message"]["content"]
