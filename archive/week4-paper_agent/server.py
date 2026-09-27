import os
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from schemas import APIResponse
from paper_agent.graph import graph, DEFAULT_CONFIG

app = FastAPI(title="课程知识库问答 Agent", version="1.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def chat_page():
    return FileResponse(os.path.join(STATIC_DIR, "chat.html"))


@app.post("/qa/ask") #装饰器,注册路由
def ask(req: QuestionRequest): #FastAPI 会自动把请求体 JSON 解析成 QuestionRequest 对象
    """问答接口：用户提问 → LangGraph Agent 自主调用工具 → 返回答案"""
    try:
        result = graph.invoke(
            {"messages": [("user", req.question)]},
            config=DEFAULT_CONFIG  # 限制最多10轮
        )
        # 取最后一条 AI 消息作为答案
        last_msg = result["messages"][-1] #负索引,[0]是第一个,[-1]是最后一个
        answer = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
        #三元表达式,hasattr()判断对象是否有某个属性,如果有就返回last_msg.content,否则返回str(last_msg)

        # 统计工具调用
        tool_calls = []
        for msg in result["messages"]:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tool_calls.append({"tool": tc["name"], "args": tc["args"]})

        return APIResponse(data={
            "answer": answer,
            "tools_used": tool_calls,
        })
    except Exception as e:
        return APIResponse(code=1, msg=f"Agent执行出错: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8003)
