"""简化的AI Agent Web应用"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import json
import uvicorn

app = FastAPI(title="AI Agent Web应用", version="1.0.0")

# 挂载静态文件
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def get_home():
    """重定向到聊天页面"""
    return """
<!DOCTYPE html>
<html>
<head>
    <title>AI Agent Demo</title>
    <meta http-equiv="refresh" content="0; url=/static/simple_chat.html">
</head>
<body>
    <a href="/static/simple_chat.html">点击进入聊天界面</a>
</body>
</html>
"""

@app.post("/api/chat")
async def chat_endpoint():
    """简化聊天API"""
    def generate():
        # 模拟AI回复
        responses = [
            "TCP是面向连接的传输层协议。三次握手过程：1）客户端发送SYN；2）服务器回复SYN-ACK；3）客户端回复ACK。",
            "进程是程序的一次执行，是系统资源分配的基本单位；线程是进程内的执行单元。主要区别：进程有独立内存空间，线程共享进程资源。",
            "死锁的四个必要条件：1）互斥条件；2）请求与保持条件；3）不可剥夺条件；4）循环等待条件。",
            "虚拟内存是一种内存管理技术，通过页表机制将逻辑地址转换为物理地址。它允许程序使用比实际物理内存更大的地址空间。",
            "HTTP是超文本传输协议，基于TCP/IP，是无状态协议。HTTPS是HTTP的安全版本，通过SSL/TLS加密通信。"
        ]

        # 模拟流式响应
        response = responses[0]  # 使用第一个响应
        for char in response:
            yield f"data: {json.dumps({'type': 'content', 'content': char}, ensure_ascii=False)}\n\n"
        yield f"data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")

if __name__ == "__main__":
    print("Starting AI Agent Web App...")
    print("Access at: http://127.0.0.1:8004")
    uvicorn.run("web_app_simple:app", host="0.0.0.0", port=8004, reload=True)