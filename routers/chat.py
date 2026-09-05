import json
import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from config import API_KEY, API_URL
from schemas import ChatRequest, ChatResponse, success, error

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    payload = {"model": req.model, "messages": messages}
    try:
        resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        return ChatResponse(code=0, msg="ok", reply=reply)
    except requests.exceptions.Timeout:
        return ChatResponse(code=1, msg="请求超时", reply=None)
    except requests.exceptions.HTTPError:
        return ChatResponse(code=1, msg="API调用失败", reply=None)
    except Exception:
        return ChatResponse(code=1, msg="服务器内部错误", reply=None)


@router.post("/chat/stream")
#async:流式接口要用异步.
async def chat_stream(req: ChatRequest):
    #文档类字符串,鼠标悬浮函数上会弹出提示说明;会生成字符串对象,函数开头会保存为文档
    """流式聊天：AI回复一个字一个字地推给前端"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    payload = {
        "model": req.model,
        "messages": messages,
        "stream": True,  # 开启流式输出
    }
    #定义生成器函数
    def generate():
        resp = requests.post(API_URL, headers=headers, json=payload, stream=True, timeout=60)
        for line in resp.iter_lines():
            #跳过空行
            if not line:
                continue
            line_str = line.decode("utf-8")
            # SSE格式: data: {...}
            if line_str.startswith("data: "):
                data_str = line_str[6:]  # 去掉 "data: " 前缀
                if data_str == "[DONE]":
                    break  # 流结束
                try:
                    #从嵌套字典中取出增量数据
                    chunk = json.loads(data_str)
                    #取出内容,可能是一个字,也可能为空
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    return StreamingResponse(generate(), media_type="text/event-stream")
