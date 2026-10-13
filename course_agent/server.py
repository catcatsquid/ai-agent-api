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
from datetime import datetime, timezone
from pathlib import Path as FilePath
from fastapi import FastAPI, Depends, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, HTMLResponse, RedirectResponse
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage, AIMessageChunk
from course_agent.graph import (
    graph,
    react_graph,
    plan_checkpointer,
    react_checkpointer,
)
from course_agent.auth import any_auth, create_jwt
from course_agent.logger import logger
from course_agent.memory import INTERNAL_LLM_TAG
from course_agent.middleware import setup_middlewares
from course_agent import tutor_agent, qbank, quiz_service
from config import APP_VERSION
from rag_tool import get_knowledge_base

# 项目根目录：一律基于文件位置解析，不依赖启动时的 CWD
# （原来用 FilePath("data") 相对路径，从别处启动 uvicorn 会静默找不到图片/题库）
PROJECT_ROOT = FilePath(__file__).resolve().parent.parent

# 知识库的文件根目录：所有"按用户传入路径读文件"的接口都必须落在这个目录之下
DATA_ROOT = (PROJECT_ROOT / "data").resolve()


def resolve_within_data(user_path: str, *, must_exist: bool = True) -> FilePath:
    """把用户传入的路径解析到 data/ 之下；越界就抛 ValueError

    为什么需要它：`/kb/add` 与 `/kb/add-directory` 原来把 file_path / dir_path
    **原样**交给 `rag_tool.add_document`，而那条路径只做 `Path(file_path).exists()`，
    没有任何"必须在 data/ 之下"的校验。再加上 `_classify_doc` 的兜底行为
    （路径不在 base 之下时退回按全路径判断，最终 `return "notes"`），
    结果就是：data/ 之外的任意 md/txt/pdf/docx 都会被当成"笔记"收进知识库，
    然后经（当时还无鉴权的）`/kb/search` 把全文读出去。

    `/kb/browse` 一直是这么防的（resolve + relative_to），这里抽成同一个函数给三处复用 ——
    校验放在 **API 边界**而不是 `add_document` 内部，因为 CLI 的 `--add-file`
    确实支持传项目外的绝对路径，那是合理用法，不该被库层一刀切掉。
    """
    if not user_path or not str(user_path).strip():
        raise ValueError("路径不能为空")
    p = FilePath(str(user_path).strip())
    candidate = (p if p.is_absolute() else PROJECT_ROOT / p).resolve()
    try:
        candidate.relative_to(DATA_ROOT)
    except ValueError:
        raise ValueError(f"路径必须位于 data/ 目录之下: {user_path}")
    if must_exist and not candidate.exists():
        raise ValueError(f"路径不存在: {user_path}")
    return candidate


# 只有"写答案的那个节点"有资格把内容推给用户（按图区分：plan 图是 synthesizer，
# react 图是 llm）。用白名单而不是黑名单 —— 未知来源一律不放行（fail closed）。
PLAN_ANSWER_NODES = frozenset({"synthesizer"})
REACT_ANSWER_NODES = frozenset({"llm"})


def sse_content_event(msg_chunk, metadata=None, answer_nodes=None) -> str | None:
    """把一条流式消息转成 SSE content 事件；该跳过的返回 None

    ## 判据是"来源白名单"，不是"类型黑名单"

    原来只判类型（ToolMessage / 带 tool_calls / 空 content）—— 那是个**黑名单**，
    默认值是"放行"：图里新增任何消息类型都会直接被播出去。

    更实际的问题：LangGraph 的 messages 模式会把图内**每一次** LLM 调用都推出来，
    包括 `compress_history` 在 planner/synthesizer **内部**跑的那次摘要调用。
    它是一条普通消息，类型黑名单拦不住。实测（见 tests/test_stream_contract.py）：
      · 摘要 token 带着**同一个** langgraph_node 流出去 → 按节点也挡不住；
      · 给那次调用加 `stream=False` 只把它从 token 流变成一条整消息，消息照样在流里。
      所以 `memory.compress_history` 会给那次调用打 INTERNAL_LLM_TAG，这里再挡一道。

    还有一条容易忽略的：messages 模式除了 LLM 的 token，还会把**节点返回的消息**
    再发一遍。而 `synthesizer_node` 返回的正是一条完整 AIMessage —— 放行它，
    答案会在气泡里出现两遍（token 一遍 + 整条一遍）。token 的类型是
    `AIMessageChunk`、节点返回的是 `AIMessage`，用这个差来区分最省事。

    四层判据，顺序即优先级：
      ① 来源白名单：metadata["langgraph_node"] 必须在允许集合里；
      ② 内部 tag：带 INTERNAL_LLM_TAG 的调用一律不播（如对话摘要）；
      ③ 只放行 AIMessageChunk —— 顺带挡掉节点返回的消息与 ToolMessage；
      ④ content 非空。

    Args:
        msg_chunk: LangGraph messages 模式给出的消息或消息片段
        metadata: 同一条流给出的 metadata（含 langgraph_node / tags）
        answer_nodes: 允许成为正文的节点名集合；None 表示跳过①（给不按图区分的调用方）
    """
    meta = metadata or {}

    # ① 来源白名单 —— 未知来源不放行
    if answer_nodes is not None and meta.get("langgraph_node") not in answer_nodes:
        return None

    # ② 内部工具型调用（如对话摘要）打了 tag，即便来自允许的节点也不播
    if INTERNAL_LLM_TAG in (meta.get("tags") or []):
        return None

    # ③ 只放行"LLM 逐字产出"。节点返回的完整 AIMessage / ToolMessage / HumanMessage
    #    都不是 chunk，一律不放行（否则答案会被发两遍）
    if not isinstance(msg_chunk, AIMessageChunk):
        return None

    # ④ 没有正文可发（ReAct 中间轮的工具决策 chunk 通常 content 为空）
    content = getattr(msg_chunk, "content", "") or ""
    if not content:
        return None
    return f"data: {json.dumps({'type': 'content', 'content': content}, ensure_ascii=False)}\n\n"


app = FastAPI(title="Course QA Agent API", version=APP_VERSION)

# 注册所有中间件（请求日志、限流、CORS、全局异常处理）
setup_middlewares(app)

# 静态文件服务（知识库管理页面）
_static_dir = FilePath(__file__).parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# 题库图片服务（独立路径，避免被 /static 挂载覆盖）
_qb_image_dir = qbank.IMAGES_DIR
if _qb_image_dir.exists():
    app.mount("/qb_images", StaticFiles(directory=str(_qb_image_dir)), name="qb_images")


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


class TutorRequest(BaseModel):
    query: str
    session_id: str = None


class LoginRequest(BaseModel):
    username: str
    password: str = None  # 演示用，实际项目要校验密码


class LoginResponse(BaseModel):
    code: int = 0
    msg: str = "ok"
    token: str = None
    token_type: str = "bearer"
    expires_in: int = 0


@app.get("/health")
def health():
    """健康检查（供 docker-compose / 负载均衡探活）

    注意：不要用 "/" 做健康检查——"/" 现在会 302 跳到 /chat，
    探活返回 307 在部分探针里会被判失败。（原 test_health_no_auth 就是因此挂掉的）
    """
    questions = qbank.load_questions()
    try:
        kb = get_knowledge_base()
        chunks = kb.count()
    except Exception as e:
        logger.warning("健康检查读取知识库失败: %s", e)
        chunks = -1
    return {
        "code": 0,
        "msg": "ok",
        "status": "healthy",
        "knowledge_chunks": chunks,
        "question_bank_size": len(questions),
    }


@app.get("/")
def root():
    return RedirectResponse(url="/chat")


@app.post("/auth/login", response_model=LoginResponse)
def login(req: LoginRequest):
    """登录换取 JWT 令牌

    注意：`req.password` 原来**从未被读取** —— 任意用户名 + 任意密码都能拿到 token，
    等于"需要鉴权"的写接口（/kb/add 等）对匿名者不设防。
    现在必须匹配 .env 里的 ADMIN_PASSWORD，未配置就直接拒绝签发（fail closed）。
    """
    from config import JWT_EXPIRE_MINUTES, ADMIN_PASSWORD
    from course_agent.auth import JWT_ENABLED, verify_admin_password

    if not ADMIN_PASSWORD:
        logger.warning("登录被拒：未配置 ADMIN_PASSWORD")
        return LoginResponse(code=503, msg="未配置 ADMIN_PASSWORD，登录不可用；请在 .env 中设置")
    if not JWT_ENABLED:
        logger.warning("登录被拒：未配置 JWT_SECRET")
        return LoginResponse(code=503, msg="未配置 JWT_SECRET，无法签发令牌")
    if not req.username:
        return LoginResponse(code=400, msg="用户名不能为空")

    if not verify_admin_password(req.password):
        # 不透露是用户名错还是密码错
        logger.warning("登录失败：密码错误 (user=%s)", req.username)
        return LoginResponse(code=401, msg="用户名或密码错误")

    token = create_jwt(user_id=req.username, extra={"username": req.username})
    logger.info("用户登录成功: %s", req.username)
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
            # 只给语义标签，不回显工具返回内容（同 /qa/stream 的进度事件）。
            # 原来这里连工具名都没了（写死 "tool": "executed"），只剩一段原文预览。
            tool_name = getattr(msg, "name", "") or "工具"
            tool_calls.append({
                "tool": tool_name,
                "summary": quiz_service.tool_summary(tool_name, msg.content),
            })
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

def _tool_result_messages(msgs: list) -> list:
    """只挑出**工具结果**消息（渲染步骤用）

    executor / tools 节点的输出里除了 ToolMessage，还有一条带 `tool_calls` 的
    AIMessage —— 它没有 `name` 属性，渲染时会兜底成「工具1」，于是前端多出一个
    **不存在的步骤**，`total` 也多算 1。实测 trace：

        步骤 1/3: 工具1   （调用了工具：list_papers、get_paper_questions）
        步骤 2/3: list_papers
        步骤 3/3: get_paper_questions

    工具其实只有 2 个。所以这里按"有没有 tool_call_id / 是不是 ToolMessage"过滤。
    """
    out = []
    for m in msgs or []:
        if getattr(m, "tool_call_id", None) or type(m).__name__ == "ToolMessage":
            out.append(m)
    return out

def _node_progress_events(node_name: str, node_output: dict) -> list:
    """把一个节点的产出翻译成若干条进度事件

    两种图共用：plan 图有 planner / executor / web_search_fallback / synthesizer，
    react 图有 llm / tools。前端对 executor_step 与 tool_step 的渲染是一样的，
    保留两个名字只是沿用历史（前端两个 case 都在）。
    """
    event = {"type": "node", "node": node_name}
    out = []

    if node_name == "planner":
        plan = node_output.get("plan", [])
        if isinstance(plan, dict):
            steps = plan.get("steps", [])
            need_tools = plan.get("need_tools", False)
        else:
            steps = plan if isinstance(plan, list) else []
            need_tools = len(steps) > 0
        event["plan"] = steps
        event["need_tools"] = need_tools
        event["message"] = f"规划完成，{len(steps)} 个步骤"
        out.append(event)

    elif node_name in ("executor", "tools"):
        step_kind = "executor_step" if node_name == "executor" else "tool_step"
        msgs = _tool_result_messages(node_output.get("messages", []))
        for i, msg in enumerate(msgs):
            tool_name = getattr(msg, "name", "") or f"工具{i+1}"
            # 只给语义标签，**不回显工具返回内容**（见 quiz_service.tool_summary）
            summary = quiz_service.tool_summary(tool_name, msg.content)
            out.append({
                "type": "node", "node": step_kind,
                "step": i + 1, "total": len(msgs),
                "tool": tool_name, "summary": summary,
                "message": f"[{tool_name}] {summary}",
            })
        event["tool_count"] = len(msgs)
        event["message"] = (f"执行完成，{len(msgs)} 个工具调用" if node_name == "executor"
                            else f"工具执行完成，{len(msgs)} 个结果")
        out.append(event)

    elif node_name == "web_search_fallback":
        event["message"] = "知识库结果不足，已补充联网搜索"
        out.append(event)

    elif node_name == "synthesizer":
        event["message"] = "回答生成完毕"
        out.append(event)

    elif node_name == "llm":
        event["message"] = "LLM 思考中"
        out.append(event)

    elif node_name == "force_search":
        event["message"] = "强制检索知识库"
        out.append(event)

    else:
        event["message"] = f"节点 {node_name} 执行完毕"
        out.append(event)

    return out


@app.post("/qa/stream")
def qa_stream(req: QueryRequest):
    """SSE 流式问答（**唯一实现** —— 原来 /chat/stream 是它的逐行复制品，已删）

    返回 text/event-stream，逐步推送：
      1. start   — 会话开始，返回 thread_id
      2. node    — 每条进度（planner / executor_step / tool_step / …）
      3. content — 答案的**逐字**输出，只来自"写答案的那个节点"
      4. done    — 完成，带最终回答（权威值）
      5. error   — 出错

    `content` 与 `done.answer` 的关系：前者是打字机效果，后者是权威结果 ——
    前端在 done 时用 answer 覆盖一次，所以"流式少了一个字/多了一遍"这类问题
    不会留在界面上；而模型不流式时（一个 content 都没有）答案也照样能到。

    注意 `content` 的过滤有四层判据（来源白名单 / 内部 tag / 只放行 chunk / 非空），
    详见 sse_content_event —— 尤其"只放行 AIMessageChunk"这条：messages 模式会把
    节点**返回**的完整 AIMessage 也推一遍，放行就会让答案在气泡里出现两遍。
    """
    tid = req.thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}
    is_react = req.mode == "react"
    active_graph = react_graph if is_react else graph
    answer_nodes = REACT_ANSWER_NODES if is_react else PLAN_ANSWER_NODES
    initial_state = {"messages": [HumanMessage(content=req.query)], "plan": []}

    def event_stream():
        yield _sse({"type": "start", "thread_id": tid, "query": req.query})
        try:
            for mode, chunk in active_graph.stream(
                initial_state, config=config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node_name, node_output in chunk.items():
                        for event in _node_progress_events(node_name, node_output or {}):
                            yield _sse(event)
                elif mode == "messages":
                    msg_chunk, metadata = chunk
                    line = sse_content_event(msg_chunk, metadata, answer_nodes)
                    if line:
                        yield line

            # 权威的最终回答：即使一个 content 都没流出来，前端也能拿到结果
            final_state = active_graph.get_state(config)
            messages = final_state.values.get("messages", [])
            answer = messages[-1].content if messages else ""
            yield _sse({"type": "done", "answer": answer, "thread_id": tid})

        except Exception as e:
            import traceback
            traceback.print_exc()
            yield _sse({"type": "error", "msg": str(e)[:300], "thread_id": tid})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )


# ============================================================
# 会话列表（服务端保存的 thread）
# ============================================================

# 1582-10-15 → 1970-01-01 的秒数：checkpoint_id（UUIDv6）的时间起点
_GREGORIAN_TO_UNIX_SEC = 12219292800


def _checkpoint_time(checkpoint_id: str) -> str | None:
    """把 checkpointer 的 checkpoint_id 解成 ISO 时间

    checkpoint_id 是 **UUIDv6**：前 48 位是 60 位时间戳的**高 48 位**，
    而时间戳是「1582-10-15 起的 100 纳秒数」。所以左移 12 位补回 60 位、
    除以 1e7 得秒，再减掉到 1970 的偏移。

    （实测：三个相隔 2.5 秒写入的 checkpoint，解出来与实际写入时间差 0.0 秒。
      解析失败返回 None —— 排序不依赖它，靠 checkpoint_id 自身的单调性。）
    """
    try:
        high48 = int(checkpoint_id.replace("-", "")[:12], 16)
        return datetime.fromtimestamp(
            (high48 << 12) / 1e7 - _GREGORIAN_TO_UNIX_SEC, timezone.utc).isoformat()
    except (ValueError, AttributeError, OSError, OverflowError):
        return None


def _thread_title(app, thread_id: str) -> str:
    """用这个会话里的第一句用户提问当标题

    取的是**最新** state 里的第一条 HumanMessage。对话被摘要压缩过之后，
    最早那条可能已经不在 state 里了 —— 那时标题会往后挪，属于已知取舍：
    为了拿"最初那一问"去逐条扫 checkpointer 历史，代价不值得。
    """
    try:
        state = app.get_state({"configurable": {"thread_id": thread_id}})
        for msg in ((state.values or {}).get("messages") or []):
            if isinstance(msg, HumanMessage) and (msg.content or "").strip():
                return str(msg.content).strip().replace("\n", " ")[:40]
    except Exception as e:      # 读不出来不该让整个列表挂掉
        logger.debug("会话标题读取失败 %s: %s", thread_id, e)
    return ""


def list_server_conversations(limit: int = 30) -> list:
    """列出落盘的会话（最近的在前）

    为什么这个接口以前不可能有：会话状态原来在 MemorySaver 里（**进程内存**），
    服务一重启就没了，也就没什么可列的。1.2.0 换成落盘的 SqliteSaver 之后，
    服务端才第一次真的"存着"这些会话。

    直接查 checkpointer 的库：`checkpoints` 表按 thread_id 聚合，
    用 `checkpoint_ns = ''` 过滤掉子图（只有顶层才是"一个会话"）。
    """
    rows = []
    for saver, app, mode in ((plan_checkpointer, graph, "plan"),
                             (react_checkpointer, react_graph, "react")):
        try:
            found = saver.conn.execute(
                "SELECT thread_id, MAX(checkpoint_id) AS latest, COUNT(*) AS n "
                "FROM checkpoints WHERE checkpoint_ns = '' "
                "GROUP BY thread_id ORDER BY latest DESC LIMIT ?",
                (limit,)).fetchall()
        except Exception as e:      # 库被并发写锁住时别让接口 500
            logger.warning("会话列表查询失败(%s): %s", mode, e)
            continue
        for thread_id, latest, n in found:
            rows.append({
                "thread_id": thread_id,
                "mode": mode,
                "checkpoints": n,
                "updated_at": _checkpoint_time(latest),
                "title": _thread_title(app, thread_id),
            })
    rows.sort(key=lambda c: (c["updated_at"] or ""), reverse=True)
    return rows[:limit]


@app.get("/chat/conversations")
def chat_conversations(limit: int = Query(30, ge=1, le=200)):
    """列出服务端保存的会话

    前端目前把会话列表存在浏览器 localStorage（`chats`）里，换设备或清缓存就没了。
    这个接口把"服务端存着哪些会话"暴露出来，供前端改用。

    ⚠️ 这是单用户项目，**没有会话归属**：所有凭据看到同一份列表。
    鉴权由 AuthGuardMiddleware 默认拒绝地管着 —— 开启 AUTH_ENFORCE 时
    这个接口和其它接口一样需要凭据。
    """
    return {"code": 0, "msg": "ok", "data": list_server_conversations(limit)}


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
def kb_stats():
    """知识库统计信息"""
    kb = get_knowledge_base()
    return {"code": 0, "msg": "ok", "data": kb.get_stats()}


@app.get("/kb/documents")
def kb_documents():
    """知识库文档列表"""
    kb = get_knowledge_base()
    return {"code": 0, "msg": "ok", "data": kb.list_documents()}


@app.post("/kb/search")
def kb_search(req: SearchTestRequest):
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
    file_path: str = Query(..., description="文件路径（必须位于 data/ 目录下）"),
    doc_type: str = Query(None, description="文档类型：ppt/notes/textbook，留空自动判断"),
    chapter: str = Query(None, description="章节，留空自动从文件名提取"),
    auth: dict = Depends(any_auth),
):
    """增量导入单个文件（只允许 data/ 目录下的文件）"""
    try:
        resolved = resolve_within_data(file_path, must_exist=False)
    except ValueError as e:
        logger.warning("拒绝导入越界路径: %s", file_path)
        return {"code": 403, "msg": str(e)}
    if not resolved.is_file():
        return {"code": 404, "msg": f"文件不存在: {file_path}"}
    kb = get_knowledge_base()
    try:
        added = kb.add_document(str(resolved), doc_type=doc_type, chapter=chapter)
        return {"code": 0, "msg": "ok", "added_chunks": added}
    except Exception as e:
        return {"code": 500, "msg": f"导入失败: {str(e)[:200]}"}


@app.post("/kb/add-directory")
def kb_add_directory(
    dir_path: str = Query(..., description="目录路径（必须位于 data/ 目录下）"),
    skip_textbook: bool = Query(False, description="是否跳过课本"),
    auth: dict = Depends(any_auth),
):
    """增量导入目录下所有新文件（只允许 data/ 目录下的目录）"""
    try:
        resolved = resolve_within_data(dir_path, must_exist=False)
    except ValueError as e:
        logger.warning("拒绝导入越界目录: %s", dir_path)
        return {"code": 403, "msg": str(e)}
    if not resolved.is_dir():
        return {"code": 404, "msg": f"目录不存在: {dir_path}"}
    kb = get_knowledge_base()
    try:
        result = kb.add_directory(str(resolved), skip_textbook=skip_textbook)
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


# ---------------------------------------------------------------- 对话导出
#
# 为什么要有服务端导出：前端原来只做 `Blob` + `<a download>`，
# 然后**无条件**弹"对话已导出"。但下载是异步的，还可能被浏览器/iframe 拦掉，
# 前端根本无从得知结果 —— 实测点完提示成功，下载目录里却一个文件都没有。
#
# 所以改成：**先在服务端落盘**（这一步的成败前端一定拿得到），
# 把真实路径显示给用户；浏览器下载降级为"顺带触发一下"。
EXPORT_ROOT = (PROJECT_ROOT / "exports").resolve()


class ChatExportRequest(BaseModel):
    title: str = ""
    markdown: str = ""
    messages: list[dict] = []


def _safe_export_name(title: str) -> str:
    """把对话标题变成安全的文件名（去掉路径分隔符与 Windows 非法字符）"""
    name = (title or "对话").strip()
    for ch in '\\/:*?"<>|\r\n\t':
        name = name.replace(ch, "_")
    name = name.strip(". ") or "对话"
    return name[:60]


def _render_chat_markdown(title: str, messages: list[dict]) -> str:
    """把 messages 渲染成 Markdown（与前端保持同一份格式）"""
    lines = [f"# {title or '对话'}", "",
             f"> 导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             f"> 消息数: {len(messages)}", "", "---", ""]
    for m in messages:
        role = "👤 用户" if m.get("role") == "user" else "🤖 助手"
        lines += [f"## {role}", "", str(m.get("content") or ""), ""]
    return "\n".join(lines)


@app.post("/chat/export")
def chat_export(req: ChatExportRequest, auth: dict = Depends(any_auth)):
    """把当前对话导出成 Markdown，落在项目的 exports/ 目录下

    返回里带**真实路径**，前端会显示出来 —— 这样"导出到哪了"永远有答案，
    不依赖浏览器的下载行为。
    """
    md = req.markdown
    if not md and req.messages:
        md = _render_chat_markdown(req.title, req.messages)
    if not md.strip():
        return {"code": 400, "msg": "没有可导出的内容"}

    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    # 时间戳只到秒 —— 同一秒内导出两次会算出同一个文件名，
    # 后一次会把前一次**静默覆盖**掉（写测试时实测到了）。
    # 所以这里保证不重名：撞了就往后加序号。
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = f"{_safe_export_name(req.title)}_{stamp}"
    target = EXPORT_ROOT / f"{base}.md"
    n = 2
    while target.exists():
        target = EXPORT_ROOT / f"{base}_{n}.md"
        n += 1

    # 文件名已过滤分隔符，这里再兜一道：必须仍落在 exports/ 之内
    if EXPORT_ROOT != target.resolve().parent:
        logger.warning("拒绝越界的导出文件名: %s", target.name)
        return {"code": 403, "msg": "非法文件名"}
    target = target.resolve()

    try:
        target.write_text(md, encoding="utf-8")
    except OSError as e:
        logger.warning("对话导出写入失败: %s", e)
        return {"code": 500, "msg": f"写入失败: {str(e)[:120]}"}

    rel = target.relative_to(PROJECT_ROOT).as_posix()
    nbytes = len(md.encode("utf-8"))
    logger.info("对话已导出: %s (%d 字节)", rel, nbytes)
    return {"code": 0, "msg": "ok", "path": rel, "abs_path": str(target),
            "filename": target.name, "bytes": nbytes}


@app.get("/kb/browse")
def kb_browse(path: str = Query("data", description="要浏览的目录路径")):
    """浏览目录结构，返回子目录和文件列表。最高只能到 data 目录。"""
    data_root = DATA_ROOT
    try:
        base = resolve_within_data(path, must_exist=False)
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
            parent = str(rel_parent.relative_to(PROJECT_ROOT)).replace("\\", "/")
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

QUESTION_BANK_DIR = qbank.QUESTION_BANK_DIR

# 前端下拉里的"全部题库（合并）"就是这个文件名。
# 注意它**不是**真的"全部"：question_bank.json 只有 125 题，而 questions_*.json
# 去重后合计 191 题（这文件漏了 66 题，因为有两份卷子没有对应的源 PDF）。
# 所以凡是要"全部"的地方都必须走 qbank.load_questions()，不能直接读这个文件。
ALL_PAPERS_SENTINEL = "question_bank.json"


def _load_questions_for_request(filename: str):
    """题库取数的唯一入口

    Returns:
        (questions, error_response) —— error_response 非 None 时直接返回给客户端
    """
    if not filename or filename == ALL_PAPERS_SENTINEL:
        # 走题库访问层，它会合并所有 questions_*.json
        return qbank.load_questions(), None

    target = QUESTION_BANK_DIR / filename
    if not target.exists():
        return None, {"code": 404, "msg": f"题库文件不存在: {filename}"}
    try:
        with open(target, "r", encoding="utf-8-sig") as fh:
            return json.load(fh), None
    except Exception as e:
        return None, {"code": 500, "msg": f"读取题库失败: {str(e)[:100]}"}


@app.get("/qb/list")
def qb_list():
    """列出所有题库JSON文件"""
    if not QUESTION_BANK_DIR.exists():
        return {"code": 0, "msg": "ok", "data": []}
    json_files = sorted(QUESTION_BANK_DIR.glob("questions_*.json"))
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
    # "全部题库（合并）"现在是**虚拟条目**：数据来自 qbank.load_questions()，
    # 不再依赖 question_bank.json 这个文件。
    # 那个文件只装了 125 题（单卷合计 191），是"看着像权威、其实漏 66 题"的陷阱，
    # 已经删掉了。保留这个下拉项只是为了不破坏前端的选择体验。
    files.insert(0, {
        "name": "全部题库（合并）",
        "filename": ALL_PAPERS_SENTINEL,
        "count": len(qbank.load_questions()),
    })
    return {"code": 0, "msg": "ok", "data": files}


class QBSearchRequest(BaseModel):
    keyword: str = ""
    q_type: str = ""
    topic: str = ""
    source: str = ""
    filename: str = ALL_PAPERS_SENTINEL


@app.post("/qb/search")
def qb_search(req: QBSearchRequest):
    """搜索题库题目"""
    if not QUESTION_BANK_DIR.exists():
        return {"code": 0, "msg": "ok", "count": 0, "data": []}

    questions, err = _load_questions_for_request(req.filename)
    if err is not None:
        return err
    if not questions:
        return {"code": 0, "msg": "ok", "count": 0, "data": []}

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
    """获取所有题目知识点列表（用于筛选项）

    走 qbank 而不是直接读 question_bank.json —— 后者只是"某次批量构建"的产物，
    并不包含全部试卷（漏了 2019 计网期末和 2018 OS WuuTang 共 66 题），
    直接读它会让筛选项和助教实际能出的卷子对不上。
    """
    questions = qbank.load_questions()
    topics = sorted({q.get("topic", "") for q in questions if q.get("topic")})
    sources = sorted({q.get("source", "") for q in questions if q.get("source")})
    return {"code": 0, "msg": "ok",
            "data": {"topics": topics, "sources": sources, "total": len(questions)}}
