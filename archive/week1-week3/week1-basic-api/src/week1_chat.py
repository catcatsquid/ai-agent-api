import json
import requests
from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL
from schemas import ChatRequest, APIResponse

router = APIRouter()


@router.post("/chat")
def chat(req: ChatRequest):
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    payload = {"model": req.model, "messages": messages}
    try:
        resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        reply = data["choices"][0]["message"]["content"]
        return APIResponse(data={"reply": reply})
    except requests.exceptions.Timeout:
        return APIResponse(code=1, msg="请求超时")
    except requests.exceptions.HTTPError:
        return APIResponse(code=1, msg="API调用失败")
    except Exception:
        return APIResponse(code=1, msg="服务器内部错误")


@router.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """流式聊天：AI回复一个字一个字地推给前端"""
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if req.system:
        messages.append({"role": "system", "content": req.system})
    messages.append({"role": "user", "content": req.prompt})

    payload = {
        "model": req.model,
        "messages": messages,
        "stream": True,
    }

    def generate():
        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, stream=True, timeout=60)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            yield "请求超时，请稍后重试"
            return
        except requests.exceptions.HTTPError:
            yield f"API调用失败（HTTP {resp.status_code}）"
            return
        except requests.exceptions.RequestException as e:
            yield f"网络错误: {e}"
            return

        for line in resp.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0]["delta"]
                    content = delta.get("content", "")
                    if content:
                        yield content
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

    return StreamingResponse(generate(), media_type="text/event-stream")
