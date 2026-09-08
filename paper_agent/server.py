"""课程知识库问答 Agent API 服务 (端口 8003)

启动:
    python -m uvicorn paper_agent.server:app --reload --port 8003

接口文档: http://127.0.0.1:8003/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from paper_agent.graph import graph

app = FastAPI(title="Course QA Agent API", version="1.0")


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    code: int = 0
    msg: str = "ok"
    answer: str = None
    tool_calls: list = []


@app.get("/")
def health():
    return {"code": 0, "msg": "ok", "data": "课程知识库问答 Agent 正在运行"}


@app.post("/qa/ask", response_model=QueryResponse)
def qa_ask(req: QueryRequest):
    """课程知识库问答接口

    用户提问 → LangGraph 图自动执行 ReAct 循环 → 返回答案
    """
    result = graph.invoke({"messages": [HumanMessage(content=req.query)]})

    # 提取最终答案（最后一条不带 tool_calls 的 AI 消息）
    messages = result["messages"]
    answer = messages[-1].content if messages else ""

    # 提取工具调用记录
    tool_calls = []
    for msg in messages:
        if msg.type == "ai" and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})

    return QueryResponse(
        code=0,
        msg="ok",
        answer=answer,
        tool_calls=tool_calls,
    )
