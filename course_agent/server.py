"""课程知识库问答 Agent API 服务 (端口 8003)

Week 5 升级：Plan-and-Execute + 多轮对话持久化
Week 6 升级：SSE 流式接口
Week 9 升级：API 认证 + 中间件
Week 10 升级：知识库管理接口（增量导入 + 检索测试）

启动:
    python -m uvicorn course_agent.server:app --reload --port 8003

接口文档: http://127.0.0.1:8003/docs

认证方式（配置 API_KEY 或 JWT_SECRET 后开启）:
  - API Key: 请求头 X-API-Key: <your-key>
  - JWT:     请求头 Authorization: Bearer <token>
"""

import json
import uuid
from pathlib import Path as FilePath
from fastapi import FastAPI, Depends, UploadFile, File, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
from course_agent.graph import graph, react_graph
from course_agent.auth import any_auth, create_jwt, api_key_auth
from course_agent.logger import logger
from course_agent.middleware import setup_middlewares
from rag_tool import get_knowledge_base, KnowledgeBase, ChunkConfig

app = FastAPI(title="Course QA Agent API", version="3.0")

# 注册所有中间件（请求日志、限流、CORS、全局异常处理）
setup_middlewares(app)

# 静态文件服务（知识库管理页面）
_static_dir = FilePath(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# 题库图片服务
_qb_image_dir = FilePath("data") / "question_bank" / "images"
if _qb_image_dir.exists():
    app.mount("/static/qb_images", StaticFiles(directory=str(_qb_image_dir)), name="qb_images")


class QueryRequest(BaseModel):
    query: str
    mode: str = "plan"
    thread_id: str = None  # 传入则续接历史会话，不传则新会话


class QueryResponse(BaseModel):
    code: int = 0
    msg: str = "ok"
    answer: str = None
    tool_calls: list = []
    plan: list = []
    thread_id: str = None
    turn_count: int = 0


class LoginRequest(BaseModel):
    username: str
    password: str = None  # 演示用，实际项目要校验密码


class LoginResponse(BaseModel):
    code: int = 0
    msg: str = "ok"
    token: str = None
    token_type: str = "bearer"
    expires_in: int = 0


@app.get("/")
def health():
    return {"code": 0, "msg": "ok", "data": "课程知识库问答 Agent v3.0 正在运行", "pages": {"chat": "/chat", "kb": "/kb", "docs": "/docs"}}


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """登录获取 JWT 令牌（演示用，实际项目需对接用户系统）

    演示版：任意用户名都能登录，返回有效期 60 分钟的 JWT
    """
    from config import JWT_EXPIRE_MINUTES

    if not req.username:
        return LoginResponse(code=400, msg="用户名不能为空")

    token = create_jwt(user_id=req.username, extra={"username": req.username})
    logger.info("用户登录: %s", req.username)
    return LoginResponse(
        code=0,
        msg="登录成功",
        token=token,
        expires_in=JWT_EXPIRE_MINUTES * 60,
    )


@app.post("/qa/ask", response_model=QueryResponse)
def qa_ask(req: QueryRequest, auth: dict = Depends(any_auth)):
    """课程知识库问答接口

    mode="plan": Plan-and-Execute 模式（默认）
    mode="react": ReAct 模式
    thread_id: 会话ID，传入相同值可续接多轮对话
    """
    tid = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    try:
        if req.mode == "react":
            result = react_graph.invoke(
                {"messages": [HumanMessage(content=req.query)], "plan": []},
                config=config,
            )
        else:
            result = graph.invoke(
                {"messages": [HumanMessage(content=req.query)], "plan": []},
                config=config,
            )
    except Exception as e:
        return QueryResponse(
            code=500,
            msg=f"Agent 执行失败: {str(e)[:200]}",
            answer=f"抱歉，处理您的问题时出错: {str(e)[:100]}",
            thread_id=tid,
        )

    messages = result["messages"]
    answer = messages[-1].content if messages else ""

    tool_calls = []
    for msg in messages:
        if isinstance(msg, ToolMessage):
            tool_calls.append({"tool": "executed", "content_preview": msg.content[:100]})
        elif isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({"tool": tc["name"], "args": tc["args"]})

    plan = result.get("plan", [])

    human_count = sum(1 for m in messages if isinstance(m, HumanMessage))

    return QueryResponse(
        code=0,
        msg="ok",
        answer=answer,
        tool_calls=tool_calls,
        plan=plan,
        thread_id=tid,
        turn_count=human_count,
    )


# ============================================================
# SSE 流式接口（Week 6 新增）
# ============================================================

def _sse(data: dict) -> str:
    """把字典格式化成一条 SSE 事件"""
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


@app.post("/qa/stream")
def qa_stream(req: QueryRequest, auth: dict = Depends(any_auth)):
    """SSE 流式问答接口

    返回 text/event-stream，逐步推送：
      1. start   — 会话开始，返回 thread_id
      2. node    — 每个节点执行完毕，推送节点名和关键数据
      3. done    — 全部完成，推送最终回答
      4. error   — 出错时推送错误信息
    """
    tid = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    active_graph = react_graph if req.mode == "react" else graph
    initial_state = {"messages": [HumanMessage(content=req.query)], "plan": []}

    def event_stream():
        yield _sse({"type": "start", "thread_id": tid, "query": req.query})

        try:
            # 双模式流：updates 捕获节点级事件，messages 捕获 LLM 逐 token 输出
            for mode, chunk in active_graph.stream(
                initial_state, config=config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node_name, node_output in chunk.items():
                        event = {"type": "node", "node": node_name}

                        if node_name == "planner":
                            plan = node_output.get("plan", [])
                            # 兼容两种格式：数组 或 {need_tools, steps}
                            if isinstance(plan, dict):
                                steps = plan.get("steps", [])
                                need_tools = plan.get("need_tools", False)
                            else:
                                steps = plan if isinstance(plan, list) else []
                                need_tools = len(steps) > 0
                            event["plan"] = steps
                            event["need_tools"] = need_tools
                            event["message"] = f"规划完成，{len(steps)} 个步骤"
                            yield _sse(event)

                        elif node_name == "executor":
                            msgs = node_output.get("messages", [])
                            # 逐个推送每个工具调用的详情
                            for i, msg in enumerate(msgs):
                                content = msg.content if hasattr(msg, "content") else str(msg)
                                # 提取结果摘要（前100字）
                                summary = content[:120].replace("\n", " ")
                                if len(content) > 120:
                                    summary += "..."
                                # 尝试从 plan 里获取工具名和查询词（用索引对应）
                                tool_name = getattr(msg, "name", "") or f"工具{i+1}"
                                detail_event = {
                                    "type": "node",
                                    "node": "executor_step",
                                    "step": i + 1,
                                    "total": len(msgs),
                                    "tool": tool_name,
                                    "summary": summary,
                                    "message": f"[{tool_name}] {summary}",
                                }
                                yield _sse(detail_event)
                            # 最后推送完成事件
                            event["tool_count"] = len(msgs)
                            event["message"] = f"执行完成，{len(msgs)} 个工具调用"
                            yield _sse(event)

                        elif node_name == "synthesizer":
                            event["message"] = "回答生成完毕"
                            yield _sse(event)

                        elif node_name == "llm":
                            event["message"] = "LLM 思考中"
                            yield _sse(event)

                        elif node_name == "tools":
                            msgs = node_output.get("messages", [])
                            # ReAct模式：逐个推送工具结果
                            for i, msg in enumerate(msgs):
                                content = msg.content if hasattr(msg, "content") else str(msg)
                                summary = content[:120].replace("\n", " ")
                                if len(content) > 120:
                                    summary += "..."
                                tool_name = getattr(msg, "name", "") or f"工具{i+1}"
                                detail_event = {
                                    "type": "node",
                                    "node": "tool_step",
                                    "step": i + 1,
                                    "total": len(msgs),
                                    "tool": tool_name,
                                    "summary": summary,
                                    "message": f"[{tool_name}] {summary}",
                                }
                                yield _sse(detail_event)
                            event["tool_count"] = len(msgs)
                            event["message"] = f"工具执行完成，{len(msgs)} 个结果"
                            yield _sse(event)

                        elif node_name == "force_search":
                            event["message"] = "强制检索知识库"
                            yield _sse(event)

                        else:
                            event["message"] = f"节点 {node_name} 执行完毕"
                            yield _sse(event)

                elif mode == "messages":
                    # LLM 逐 token 输出，过滤工具调用消息（防止 tool call JSON 被流式输出到气泡）
                    msg_chunk, metadata = chunk
                    content = ""
                    if hasattr(msg_chunk, "content") and msg_chunk.content:
                        content = msg_chunk.content

                    # 跳过工具调用消息，其 content 可能包含原始 tool call JSON
                    if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                        continue

                    if content:
                        node_name = metadata.get("langgraph_node", "")
                        yield _sse({
                            "type": "token",
                            "content": content,
                            "node": node_name,
                        })

            final_state = active_graph.get_state(config)
            messages = final_state.values.get("messages", [])
            answer = messages[-1].content if messages else ""

            yield _sse({"type": "done", "answer": answer, "thread_id": tid})

        except Exception as e:
            yield _sse({"type": "error", "msg": str(e)[:200], "thread_id": tid})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# 新增：兼容前端的聊天接口
# ============================================================

@app.post("/chat/stream")
async def chat_stream_frontend(req: QueryRequest, auth: dict = Depends(any_auth)):
    """兼容前端聊天接口的SSE流式响应"""
    tid = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    active_graph = react_graph if req.mode == "react" else graph
    initial_state = {"messages": [HumanMessage(content=req.query)], "plan": []}

    def event_stream():
        yield f"data: {json.dumps({'type': 'start', 'thread_id': tid}, ensure_ascii=False)}\n\n"

        try:
            for mode, chunk in active_graph.stream(
                initial_state, config=config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node_name, node_output in chunk.items():
                        event = {"type": "node", "node": node_name}

                        if node_name == "planner":
                            plan = node_output.get("plan", [])
                            if isinstance(plan, dict):
                                steps = plan.get("steps", [])
                            else:
                                steps = plan if isinstance(plan, list) else []
                            event["plan"] = steps
                            event["message"] = f"规划完成，{len(steps)} 个步骤"
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                        elif node_name == "executor":
                            msgs = node_output.get("messages", [])
                            for i, msg in enumerate(msgs):
                                content = msg.content if hasattr(msg, "content") else str(msg)
                                summary = content[:120].replace("\n", " ")
                                if len(content) > 120:
                                    summary += "..."
                                tool_name = getattr(msg, "name", "") or f"工具{i+1}"
                                detail_event = {
                                    "type": "node", "node": "executor_step",
                                    "step": i + 1, "total": len(msgs),
                                    "tool": tool_name, "summary": summary,
                                    "message": f"[{tool_name}] {summary}",
                                }
                                yield f"data: {json.dumps(detail_event, ensure_ascii=False)}\n\n"
                            event["tool_count"] = len(msgs)
                            event["message"] = f"执行完成，{len(msgs)} 个工具调用"
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                        elif node_name == "synthesizer":
                            event["message"] = "回答生成完毕"
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                        elif node_name == "llm":
                            event["message"] = "LLM 思考中"
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                        elif node_name == "tools":
                            msgs = node_output.get("messages", [])
                            for i, msg in enumerate(msgs):
                                content = msg.content if hasattr(msg, "content") else str(msg)
                                summary = content[:120].replace("\n", " ")
                                if len(content) > 120:
                                    summary += "..."
                                tool_name = getattr(msg, "name", "") or f"工具{i+1}"
                                detail_event = {
                                    "type": "node", "node": "tool_step",
                                    "step": i + 1, "total": len(msgs),
                                    "tool": tool_name, "summary": summary,
                                    "message": f"[{tool_name}] {summary}",
                                }
                                yield f"data: {json.dumps(detail_event, ensure_ascii=False)}\n\n"
                            event["tool_count"] = len(msgs)
                            event["message"] = f"工具执行完成，{len(msgs)} 个结果"
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                        elif node_name == "force_search":
                            event["message"] = "强制检索知识库"
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                        else:
                            event["message"] = f"节点 {node_name} 执行完毕"
                            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"

                elif mode == "messages":
                    msg_chunk, metadata = chunk
                    content = ""
                    if hasattr(msg_chunk, "content") and msg_chunk.content:
                        content = msg_chunk.content

                    # 跳过工具调用消息，其 content 可能包含原始 tool call JSON
                    if hasattr(msg_chunk, "tool_calls") and msg_chunk.tool_calls:
                        continue

                    if content:
                        yield f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"

            yield f"data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# 知识库管理页面 + 接口（Week 10 新增）
# ============================================================

@app.get("/kb")
def kb_page():
    """知识库管理页面"""
    html_path = FilePath(__file__).parent / "static" / "kb.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>kb.html not found</h1>", status_code=404)


@app.get("/chat")
def chat_page():
    """聊天问答页面"""
    html_path = FilePath(__file__).parent / "static" / "chat.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>chat.html not found</h1>", status_code=404)
# ============================================================



class SearchTestRequest(BaseModel):
    query: str
    top_k: int = 5
    doc_types: list[str] = None
    chapter: str = None


@app.get("/kb/stats")
def kb_stats(auth: dict = Depends(any_auth)):
    """知识库统计信息"""
    kb = get_knowledge_base()
    return {"code": 0, "msg": "ok", "data": kb.get_stats()}


@app.get("/kb/documents")
def kb_documents(auth: dict = Depends(any_auth)):
    """知识库文档列表"""
    kb = get_knowledge_base()
    return {"code": 0, "msg": "ok", "data": kb.list_documents()}


@app.post("/kb/search")
def kb_search(req: SearchTestRequest, auth: dict = Depends(any_auth)):
    """检索测试 — 返回 chunk + score，方便调参"""
    kb = get_knowledge_base()
    results = kb.search(
        req.query,
        top_k=req.top_k,
        doc_types=req.doc_types,
        chapter=req.chapter,
    )
    return {"code": 0, "msg": "ok", "query": req.query, "count": len(results), "data": results}


@app.post("/kb/add")
def kb_add_document(
    file_path: str = Query(..., description="文件路径（相对项目根目录）"),
    doc_type: str = Query(None, description="文档类型：ppt/notes/textbook，留空自动判断"),
    chapter: str = Query(None, description="章节，留空自动从文件名提取"),
    auth: dict = Depends(any_auth),
):
    """增量导入单个文件"""
    kb = get_knowledge_base()
    try:
        added = kb.add_document(file_path, doc_type=doc_type, chapter=chapter)
        return {"code": 0, "msg": "ok", "added_chunks": added}
    except Exception as e:
        return {"code": 500, "msg": f"导入失败: {str(e)[:200]}"}


@app.post("/kb/add-directory")
def kb_add_directory(
    dir_path: str = Query(..., description="目录路径"),
    skip_textbook: bool = Query(False, description="是否跳过课本"),
    auth: dict = Depends(any_auth),
):
    """增量导入目录下所有新文件"""
    kb = get_knowledge_base()
    try:
        result = kb.add_directory(dir_path, skip_textbook=skip_textbook)
        return {"code": 0, "msg": "ok", "data": result}
    except Exception as e:
        return {"code": 500, "msg": f"导入失败: {str(e)[:200]}"}


@app.delete("/kb/documents/{source}")
def kb_remove_document(source: str, auth: dict = Depends(any_auth)):
    """从知识库删除某文档"""
    kb = get_knowledge_base()
    deleted = kb.remove_document(source)
    if deleted:
        return {"code": 0, "msg": f"已删除: {source}"}
    return {"code": 404, "msg": f"文档不在知识库中: {source}"}


@app.get("/kb/browse")
def kb_browse(path: str = Query("data", description="要浏览的目录路径")):
    """浏览目录结构，返回子目录和文件列表。最高只能到 data 目录。"""
    base = (FilePath(".") / path).resolve()
    data_root = FilePath("data").resolve()
    try:
        base.relative_to(data_root)
    except ValueError:
        return {"code": 403, "msg": "只能浏览 data 目录下的内容"}

    if not base.exists() or not base.is_dir():
        return {"code": 404, "msg": f"目录不存在: {path}"}

    dirs = []
    files = []
    try:
        for entry in sorted(base.iterdir()):
            if entry.is_dir():
                dirs.append({"name": entry.name, "path": str(entry).replace("\\", "/")})
            elif entry.is_file():
                stat = entry.stat()
                files.append({
                    "name": entry.name,
                    "path": str(entry).replace("\\", "/"),
                    "size": stat.st_size,
                    "ext": entry.suffix.lower(),
                })
    except PermissionError:
        return {"code": 403, "msg": "无权限访问该目录"}

    parent = None
    if base != data_root:
        rel_parent = base.parent
        try:
            rel_parent.relative_to(data_root)
            parent = str(rel_parent).replace("\\", "/").replace("./", "")
        except ValueError:
            parent = None
    return {
        "code": 0,
        "msg": "ok",
        "data": {
            "current": path,
            "parent": parent,
            "dirs": dirs,
            "files": files,
        },
    }


# ============================================================
# 题库检索接口
# ============================================================

QUESTION_BANK_DIR = FilePath("data") / "question_bank"


@app.get("/qb/list")
def qb_list():
    """列出所有题库JSON文件"""
    if not QUESTION_BANK_DIR.exists():
        return {"code": 0, "msg": "ok", "data": []}
    json_files = sorted(QUESTION_BANK_DIR.glob("questions_*.json"))
    combined = QUESTION_BANK_DIR / "question_bank.json"
    files = []
    for f in json_files:
        try:
            with open(f, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
            files.append({
                "name": f.stem,
                "filename": f.name,
                "count": len(data),
            })
        except Exception:
            files.append({"name": f.stem, "filename": f.name, "count": 0})
    if combined.exists():
        try:
            with open(combined, "r", encoding="utf-8-sig") as fh:
                data = json.load(fh)
            files.insert(0, {
                "name": "全部题库（合并）",
                "filename": combined.name,
                "count": len(data),
            })
        except Exception:
            pass
    return {"code": 0, "msg": "ok", "data": files}


class QBSearchRequest(BaseModel):
    keyword: str = ""
    q_type: str = ""
    topic: str = ""
    source: str = ""
    filename: str = "question_bank.json"


@app.post("/qb/search")
def qb_search(req: QBSearchRequest):
    """搜索题库题目"""
    if not QUESTION_BANK_DIR.exists():
        return {"code": 0, "msg": "ok", "count": 0, "data": []}

    target = QUESTION_BANK_DIR / req.filename
    if not target.exists():
        if req.filename == "question_bank.json":
            json_files = list(QUESTION_BANK_DIR.glob("questions_*.json"))
            if json_files:
                all_questions = []
                for f in json_files:
                    try:
                        with open(f, "r", encoding="utf-8-sig") as fh:
                            all_questions.extend(json.load(fh))
                    except Exception:
                        pass
                questions = all_questions
            else:
                return {"code": 0, "msg": "ok", "count": 0, "data": []}
        else:
            return {"code": 404, "msg": f"题库文件不存在: {req.filename}"}
    else:
        try:
            with open(target, "r", encoding="utf-8-sig") as fh:
                questions = json.load(fh)
        except Exception as e:
            return {"code": 500, "msg": f"读取题库失败: {str(e)[:100]}"}

    results = []
    kw = req.keyword.lower().strip()
    topic_kw = req.topic.lower().strip()

    for q in questions:
        if req.q_type and q.get("type") != req.q_type:
            continue
        if req.source and req.source not in q.get("source", ""):
            continue
        if topic_kw and topic_kw not in q.get("topic", "").lower():
            continue
        if kw:
            searchable = (
                q.get("text", "").lower()
                + " " + str(q.get("answer", "")).lower()
                + " " + " ".join(str(p) for p in q.get("reference_points", [])).lower()
                + " " + " ".join(
                    str(sq.get("answer", "")) for sq in (q.get("sub_questions") or [])
                ).lower()
                + " " + q.get("topic", "").lower()
            )
            if kw not in searchable:
                continue
        results.append(q)

    return {
        "code": 0,
        "msg": "ok",
        "count": len(results),
        "total": len(questions),
        "data": results,
    }


@app.get("/qb/topics")
def qb_topics():
    """获取所有题目知识点列表（用于筛选项）"""
    if not QUESTION_BANK_DIR.exists():
        return {"code": 0, "msg": "ok", "data": []}
    combined = QUESTION_BANK_DIR / "question_bank.json"
    questions = []
    if combined.exists():
        try:
            with open(combined, "r", encoding="utf-8-sig") as fh:
                questions = json.load(fh)
        except Exception:
            pass
    else:
        for f in QUESTION_BANK_DIR.glob("questions_*.json"):
            try:
                with open(f, "r", encoding="utf-8-sig") as fh:
                    questions.extend(json.load(fh))
            except Exception:
                pass

    topics = sorted(set(q.get("topic", "") for q in questions if q.get("topic")))
    sources = sorted(set(q.get("source", "") for q in questions if q.get("source")))
    return {"code": 0, "msg": "ok", "data": {"topics": topics, "sources": sources}}
