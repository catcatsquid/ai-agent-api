from fastapi import FastAPI
from fastapi.responses import FileResponse
from routers import health, chat

app = FastAPI(title="AI Agent API", version="1.0")

app.include_router(health.router)
app.include_router(chat.router)


@app.get("/test")
def test_page():
    """流式聊天测试页面"""
    return FileResponse("stream_test.html")
