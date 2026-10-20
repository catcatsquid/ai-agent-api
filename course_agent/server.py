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
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path as FilePath
from fastapi import FastAPI, Depends, File, Query, HTTPException, UploadFile
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
from course_agent import qbank, quiz_service, progress
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


# ---------------------------------------------------------------- 导入的落盘规则
#
# **源不限制、目标必须在 data/ 之下。**
#
# 原来 `resolve_within_data` 连**读取**都要求路径在 data/ 之下 —— 于是"从下载目录
# 导入一个文件"必然失败（用户实测遇到）。而当初立这条的理由是"防越界读取"，
# 可它防错了方向：真正要守的是**写入位置**（别把东西写到 data/ 外面去）。
#
# 另外还有一条技术理由必须复制而不是就地读：`_classify_doc` 是按"路径相对 data/
# 的位置"判类型的（含 ppt → ppt、含 课本 → textbook、含 往年题 → 跳过），
# 外部路径一律落成 notes。所以复制进 data/ 才能落到**正确的分类**上。

DEFAULT_IMPORT_DIR = "导入"


def _safe_filename(name: str) -> str:
    """只取文件名部分，丢掉任何目录成分 —— 防 `../../x` 这类名字"""
    base = FilePath(str(name).replace("\\", "/")).name.strip().replace("\x00", "")
    if base in ("", ".", ".."):
        raise ValueError(f"非法文件名: {name!r}")
    return base


def _import_root() -> FilePath:
    """没指定目标目录时的默认落点：data/导入/（按规则会判成 notes）"""
    return (DATA_ROOT / DEFAULT_IMPORT_DIR).resolve()


def _destination_dir(target_dir: str | None) -> FilePath:
    """把 target_dir 解析成 data/ 之下的目录；空则用默认落点

    **语义只有一种：相对 data/ 的路径**（如 `操作系统/ppt`）。
    为什么不顺手支持"相对项目根"：那样 `data/x` 会被解释成 `data/data/x`，
    然后**静默建出一个怪目录** —— 两种约定混在一起必然出这种事。
    （`/kb/browse` 与 `/kb/add-directory` 用的是"相对项目根"，那是它们的历史约定，
    这里刻意不同，并由 `/kb/import-targets` 直接给出本函数认识的取值。）
    """
    if not target_dir:
        dest = _import_root()
    else:
        raw = FilePath(str(target_dir).strip().replace("\\", "/"))
        if raw.is_absolute():
            raise ValueError(f"目标目录请填相对 data/ 的路径，不要填绝对路径: {target_dir}")
        candidate = (DATA_ROOT / raw).resolve()
        # 越界检查（`..` 会被 resolve 消掉，所以必须比 resolve 之后的结果）
        if candidate != DATA_ROOT and DATA_ROOT not in candidate.parents:
            raise ValueError(f"目标目录必须位于 data/ 之下: {target_dir}")
        dest = candidate
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def _place_into_data(source: FilePath, target_dir: str | None) -> FilePath:
    """把**源文件复制**到 data/ 之下，返回落点

    重名不覆盖：撞了就加序号（和对话导出同样的策略 —— 静默覆盖是最难查的一类问题）。
    复制完再兜一道"必须仍在 data/ 之下"。
    """
    dest_dir = _destination_dir(target_dir)
    name = _safe_filename(source.name)
    target = dest_dir / name
    n = 2
    while target.exists():
        target = dest_dir / f"{FilePath(name).stem}_{n}{FilePath(name).suffix}"
        n += 1
    shutil.copy2(source, target)
    final = target.resolve()
    if final != DATA_ROOT and DATA_ROOT not in final.parents:
        target.unlink(missing_ok=True)
        raise ValueError(f"目标路径越界: {target}")
    return final


# 只有"写答案的那个节点"有资格把内容推给用户（按图区分：plan 图是 synthesizer，
# react 图是 llm）。用白名单而不是黑名单 —— 未知来源一律不放行（fail closed）。
PLAN_ANSWER_NODES = frozenset({"synthesizer"})
REACT_ANSWER_NODES = frozenset({"llm"})

# 「从题库出题」这一轮**不逐字推正文**（2026-10-18 用户实测后拍板）
#
# 现象：出题那轮的正文里，每道题都带一条 `![题目](/qb_images/2024%E5%B9%B4...png)`
# —— 百分号编码的中文文件名，一条就 100+ 字符。逐字推出去时，学生先看到的是
# 一大串半截 URL 和 `![题目]`（渲染出来还是坏图），一条 5 题的答案光 URL 就要
# 输出好几百 token，看着像卡住了。
#
# 所以：这一轮只要**真的调了取题工具**，正文就不逐字推，等 `done.answer` 一次给全
# （前端在 done 时本来就会用 answer 覆盖气泡）。进度事件（node）照旧，学生看得到
# "正在取题/正在出题"，不会一片空白。
#
# 判据取**实际工具轨迹**而不是猜正文内容 —— 猜"这段像不像题"是最脆的判法；
# 而且这样"自拟练习 / 讲题 / 知识问答"都不受影响（它们照常流式）。
# 常量本身放在 quiz_service（那里是题库口径的家），评测侧 import 同一份。
QUIZ_TOOLS = quiz_service.QUIZ_TOOLS


def _quiz_turn_update(node_output) -> bool:
    """这次节点输出里有没有"从题库取题"的动作

    两种图都要认：
      · plan 图：executor 返回一条**声明**了 tool_calls 的 AIMessage
      · react 图：tool_node 返回的 ToolMessage（带 name）
    """
    for msg in (node_output or {}).get("messages") or []:
        if getattr(msg, "name", None) in QUIZ_TOOLS:
            return True
        for tc in getattr(msg, "tool_calls", None) or []:
            if isinstance(tc, dict) and tc.get("name") in QUIZ_TOOLS:
                return True
    return False



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


class LoginRequest(BaseModel):
    username: str
    password: str = None  # 演示用，实际项目要校验密码


class LoginResponse(BaseModel):
    code: int = 0
    msg: str = "ok"
    token: str = None
    token_type: str = "bearer"
    expires_in: int = 0


@app.get("/progress")
def learning_progress():
    """跨会话的学习/刷题进度（只读，给调试与将来前端用）

    数据在 `runtime/progress.json`，由 `course_agent/progress.py` 维护。
    这里**不做前端**：学生问进度时走的是 `get_learning_progress` 工具
    或开场注入，不靠点按钮查。
    """
    data = progress.snapshot()
    return {"code": 0, "msg": "ok",
            "data": {"summary": progress.summary(), **data}}


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

    ⚠️ **出题轮不逐字推正文**（`QUIZ_TOOLS`，2026-10-18 用户实测后拍板）：
    题库出题的正文里每条都带 `![题目](/qb_images/…%E5%B9%B4…png)` 这种长 URL，
    逐字推出去时学生会先看到一堆半截 URL 与坏图，而且光 URL 就要输出几百 token。
    所以这一轮只发 node 进度、正文等 done 一次给全。判据是**实际工具轨迹**。

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
        # 这一轮是不是"从题库出题"（真调了取题工具才算）→ 见 QUIZ_TOOLS 的注释
        quiz_turn = False
        try:
            for mode, chunk in active_graph.stream(
                initial_state, config=config, stream_mode=["updates", "messages"]
            ):
                if mode == "updates":
                    for node_name, node_output in chunk.items():
                        if _quiz_turn_update(node_output):
                            quiz_turn = True
                        for event in _node_progress_events(node_name, node_output or {}):
                            yield _sse(event)
                elif mode == "messages":
                    if quiz_turn:
                        # 出题轮：正文不逐字推（一堆半截 URL/坏图），done 一次给全
                        continue
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


# 手写助教（`tutor_agent.py`）**没有 HTTP 面** —— 这是 2026-10 的明确定位：
# 它只保留模块与测试（冻结的对照实现），供"手写循环 vs LangGraph"对照学习用。
#
# ⚠️ 这个位置原来有 `POST /tutor/stream`，它的经历值得记一笔：
#   · 1.1.0 我用脚本整段替换 server.py 时**误删**了它，而当时只核对了
#     "`/chat/stream` 还剩几处"，没问"是不是少了别的"；
#   · 1.5.1 根据当时的文档（写着"代码与接口全部保留"）把它**恢复**了；
#   · 1.5.3 按上面的定位**有意移除**，并把前端那一侧（`setAppMode`、请求路径、
#     `currentTutorSessionId`）成套清掉 —— 只清一半会留下半截接线。
#
# 路由清单守卫见 `tests/test_route_surface.py`；前端不得再出现 tutor 接线由
# `tests/test_tutor_reference.py::TestTutorHasNoHttpOrFrontendSurface` 守。


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


def _displayable_messages(messages) -> list:
    """从 state 的 messages 里挑出**能展示给学生**的那些

    只留"学生问的"和"模型答的"：
      · SystemMessage —— 里面是压缩摘要（带 SUMMARY_MARK），内部产物，不是任何一方的原话；
      · ToolMessage —— 检索原文 / 参考答案 / 批改 JSON，**正是要藏起来的东西**；
      · 带 tool_calls 的 AIMessage —— 内部协议声明（ReAct 的中间轮也是这样）。
    中间过程一概不返回，这与前端本来显示的内容一致（它也从没显示过工具过程）。
    """
    out = []
    for msg in messages or []:
        if getattr(msg, "tool_calls", None):
            continue
        if isinstance(msg, HumanMessage):
            role = "user"
        elif isinstance(msg, AIMessage):
            role = "assistant"
        else:
            continue
        content = (getattr(msg, "content", "") or "").strip()
        if content:
            out.append({"role": role, "content": content})
    return out


def _thread_title(app, thread_id: str) -> str:
    """用这个会话里的第一句用户提问当标题

    取的是**最新** state 里的第一条用户消息。对话被摘要压缩过之后，
    最早那条可能已经不在 state 里了 —— 那时标题会往后挪，属于已知取舍：
    为了拿"最初那一问"去逐条扫 checkpointer 历史，代价不值得。
    """
    try:
        state = app.get_state({"configurable": {"thread_id": thread_id}})
        for item in _displayable_messages((state.values or {}).get("messages")):
            if item["role"] == "user":
                return item["content"].replace("\n", " ")[:40]
    except Exception as e:      # 读不出来不该让整个列表挂掉
        logger.debug("会话标题读取失败 %s: %s", thread_id, e)
    return ""


def _thread_exists(saver, thread_id: str) -> bool:
    """这个 thread 在库里到底有没有（用于区分 404 与"还没有可展示的消息"）"""
    try:
        return saver.conn.execute(
            "SELECT 1 FROM checkpoints WHERE thread_id = ? AND checkpoint_ns = '' "
            "LIMIT 1", (thread_id,)).fetchone() is not None
    except Exception as e:
        logger.warning("会话存在性查询失败: %s", e)
        return False


def list_server_conversations(limit: int = 30) -> list:
    """列出落盘的会话（最近的在前）

    为什么这个接口以前不可能有：会话状态原来在 MemorySaver 里（**进程内存**），
    服务一重启就没了，也就没什么可列的。1.2.0 换成落盘的 SqliteSaver 之后，
    服务端才第一次真的"存着"这些会话。

    直接查 checkpointer 的库：`checkpoints` 表按 thread_id 聚合，
    用 `checkpoint_ns = ''` 过滤掉子图（只有顶层才是"一个会话"）。

    ⚠️ **探针会话（`probe-*` / `eval-*`）不进列表**。这是 2026-10-18 用户实测反馈：
    开发/验收时用真模型跑的那些会话（"我要开始学习了！""直接告诉我答案"…）会落到
    同一个 checkpoint 库里，于是**混进了用户的消息列表**。约定与 `progress` 一致
    （`progress.is_ephemeral`）：这类 thread_id 是"测试造的"，不是用户聊出来的。
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
            if progress.is_ephemeral(thread_id):
                continue
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


@app.get("/chat/conversations/{thread_id}")
def chat_conversation_detail(thread_id: str,
                             mode: str = Query("plan", pattern="^(plan|react)$")):
    """取回一个会话里**适合展示**的消息，供前端恢复历史

    为什么需要它：列表接口只给元数据（标题/模式/时间）。前端要恢复一条会话，
    还得拿到当时的对话内容 —— 而这些内容在服务端的 checkpointer 里，
    前端拿不到（它原来只存在浏览器 localStorage）。

    返回的消息**只含学生问的与模型答的**，中间过程（工具结果、内部协议声明、
    压缩摘要）一律过滤，见 _displayable_messages。

    `mode` 必传对：两个模式各用独立的 checkpointer，react 的 thread_id
    在 plan 图里根本不存在（那样会 404，而不是静默当成新会话）。
    """
    app_ = react_graph if mode == "react" else graph
    saver = react_checkpointer if mode == "react" else plan_checkpointer

    if not _thread_exists(saver, thread_id):
        raise HTTPException(status_code=404,
                            detail=f"没有这个会话（mode={mode}）")

    try:
        state = app_.get_state({"configurable": {"thread_id": thread_id}})
    except Exception as e:
        logger.warning("会话内容读取失败 %s: %s", thread_id, e)
        raise HTTPException(status_code=503, detail="会话内容暂时读不出来")

    return {"code": 0, "msg": "ok", "data": {
        "thread_id": thread_id,
        "mode": mode,
        "messages": _displayable_messages((state.values or {}).get("messages")),
    }}


@app.delete("/chat/conversations/{thread_id}")
def delete_conversation(thread_id: str,
                        mode: str = Query("plan", pattern="^(plan|react)$")):
    """删除服务端保存的这个会话（连同它的全部 checkpoint）

    为什么必须由前端显式调用：本地从 localStorage 删掉**不等于**服务端也没了 ——
    下次加载又会被导入回来，用户看到的是"删不掉"。

    顺带就是清理策略：checkpoint 是 append-only（每轮对话多一行），
    `SqliteSaver.delete_thread` 按 thread 整条清掉（checkpoints 与 writes 两张表），
    所以"用户删掉不要的会话"本身就完成了回收 —— 这个个人项目不需要再加定时任务。

    已经不存在时返回 404（而不是假装成功）：前端把 404 也当"已经干净了"处理，
    因为它真正关心的是"服务端别再还回来"。
    """
    saver = react_checkpointer if mode == "react" else plan_checkpointer
    if not _thread_exists(saver, thread_id):
        raise HTTPException(status_code=404, detail=f"没有这个会话（mode={mode}）")
    try:
        saver.delete_thread(thread_id)
    except Exception as e:
        logger.warning("会话删除失败 %s: %s", thread_id, e)
        raise HTTPException(status_code=503, detail="会话暂时删不掉")
    logger.info("已删除会话 %s (mode=%s)", thread_id, mode)
    return {"code": 0, "msg": "ok", "data": {
        "thread_id": thread_id, "mode": mode, "deleted": True,
    }}


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
    file_path: str = Query(..., description="文件路径（**任意位置**都行）"),
    target_dir: str = Query(None,
                            description="复制到 data/ 下的哪个子目录（默认 导入/）"),
    doc_type: str = Query(None, description="文档类型：ppt/notes/textbook，留空自动判断"),
    chapter: str = Query(None, description="章节，留空自动从文件名提取"),
    auth: dict = Depends(any_auth),
):
    """增量导入单个文件 —— **源不限制，目标必须在 data/ 之下**

    · 源文件已在 `data/` 之下 → 就地入库（原行为不变）
    · 源文件在别处（下载目录、桌面、D 盘…）→ **复制进** `data/<target_dir>/` 再入库

    为什么得复制而不能就地读：`_classify_doc` 按"路径相对 data/ 的位置"判类型，
    外部路径一律落成 notes —— 复制进来才能落到 ppt / 课本 / notes 的正确分类上。
    """
    src = FilePath(str(file_path).strip())
    if not src.is_absolute():
        src = PROJECT_ROOT / src
    src = src.resolve()

    # ⚠️ 顺序要紧：**先判策略，再判存在性**。
    # 反过来（先 exists 再判策略）会在"越界被禁止"时用 404/403 的差别
    # **泄露 data/ 之外某个文件存不存在**。而且 CI 里 `.env` 本来就不存在
    # （它在 .gitignore 里），先判存在性会让断言 403 的用例拿到 404。
    inside_data = src == DATA_ROOT or DATA_ROOT in src.parents
    if not inside_data:
        from config import KB_IMPORT_ALLOW_OUTSIDE
        if not KB_IMPORT_ALLOW_OUTSIDE:
            return {"code": 403,
                    "msg": "data/ 之外的路径导入已被关闭"
                           "（KB_IMPORT_ALLOW_OUTSIDE=false）；"
                           "可改用「直接上传」——它不读你机器上的路径"}

    if not src.is_file():
        return {"code": 404, "msg": f"文件不存在或不是文件: {file_path}"}

    try:
        if inside_data:
            resolved, copied = resolve_within_data(str(src)), False
        else:
            # 审计：越界来源要留痕（这是有意放开的一条能力，出事要查得到）
            logger.info("从 data/ 之外导入并复制: %s", src)
            resolved, copied = _place_into_data(src, target_dir), True
    except ValueError as e:
        logger.warning("拒绝导入: %s (%s)", file_path, e)
        return {"code": 403, "msg": str(e)}

    kb = get_knowledge_base()
    from doc_utils import _classify_doc

    # 「往年题」按约定不进知识库（它是题库的源材料，走题库那条管线）。
    # 以前这种情况会返回 `code: 0, added_chunks: 0`，界面显示"导入成功: 0 个分块" ——
    # 用户实测就是这个体验：文件放进 data/操作系统/往年题/ 再导入，看着像失败。
    # 所以这里**明确说清为什么跳过、以及该放哪**。
    if doc_type is None and _classify_doc(resolved, DATA_ROOT) is None:
        return {"code": 0, "msg": "已跳过（该目录不进知识库）", "added_chunks": 0,
                "skipped_reason": "「往年题」目录不进知识库 —— 它是题库的源材料，"
                                  "走题库导入那条管线（命令行："
                                  "`python -m course_agent.question_bank <试卷 PDF>`；"
                                  "/qb 页面只做检索）。"
                                  "要让它进知识库，请放入「课本」或含 ppt 的目录。",
                "path": str(resolved), "copied": copied}
    # 已经入过库、内容也没变 → 说明白"已是最新"，别只说"成功 0 个分块"
    if kb.is_unchanged(str(resolved)):
        return {"code": 0, "msg": "ok（文件没变化）", "added_chunks": 0,
                "skipped_reason": "文件内容与上次入库时一致，已是最新 —— "
                                  "本次没有重复入库（要重新分块请先删除这份文档，"
                                  "或改了文件再导入）。",
                "path": str(resolved), "copied": copied}
    try:
        added = kb.add_document(str(resolved), doc_type=doc_type, chapter=chapter)
        return {"code": 0, "msg": "ok", "added_chunks": added,
                "path": str(resolved), "copied": copied}
    except Exception as e:
        return {"code": 500, "msg": f"导入失败: {str(e)[:200]}"}


@app.post("/kb/upload")
async def kb_upload(
    file: UploadFile = File(..., description="要上传的文件"),
    target_dir: str = Query(None,
                            description="落到 data/ 下的哪个子目录（默认 导入/）"),
    auth: dict = Depends(any_auth),
):
    """**直接上传**文件并入库（浏览器选文件 → 落进 data/ → 入库）

    这是最省事也最安全的一条路：文件内容**经 HTTP 传过来**，
    服务端不需要去读你机器上的任意路径。（传路径那条路也保留，见 `/kb/add`。）
    """
    try:
        dest_dir = _destination_dir(target_dir)
        name = _safe_filename(file.filename or "")
    except ValueError as e:
        return {"code": 403, "msg": str(e)}

    # 落到会"跳过"的目录（往年题）就先拦住：否则文件存下来了却什么也没入库，
    # 用户只会看到"成功 0 个分块"。宁可说清楚、别存这个文件。
    from doc_utils import _classify_doc
    if _classify_doc(dest_dir / (name or "x"), DATA_ROOT) is None:
        return {"code": 400,
                "msg": "这个目标目录不进知识库（「往年题」是题库的源材料，走 /qb 那条管线）；"
                       "请改选「课本」或含 ppt 的目录 —— 文件没有保存"}

    target = dest_dir / name
    n = 2
    while target.exists():          # 重名不覆盖，加序号
        target = dest_dir / f"{FilePath(name).stem}_{n}{FilePath(name).suffix}"
        n += 1

    try:
        content = await file.read()
        target.write_bytes(content)
    except Exception as e:
        target.unlink(missing_ok=True)
        return {"code": 500, "msg": f"保存失败: {str(e)[:200]}"}

    final = target.resolve()
    if final != DATA_ROOT and DATA_ROOT not in final.parents:   # 兜底
        final.unlink(missing_ok=True)
        return {"code": 403, "msg": "目标路径越界"}

    kb = get_knowledge_base()
    try:
        added = kb.add_document(str(final))
        return {"code": 0, "msg": "ok", "added_chunks": added,
                "path": str(final), "bytes": len(content)}
    except Exception as e:
        return {"code": 500, "msg": f"入库失败（文件已保存到 data/）: {str(e)[:200]}"}


@app.get("/kb/import-targets")
def kb_import_targets(auth: dict = Depends(any_auth)):
    """列出可选的导入目标目录

    为什么要给一份清单、而不是让用户随便填：`_classify_doc` 是**按子目录名**判类型的
    （含 ppt → ppt、含 课本 → textbook、含 往年题 → 跳过），填错目录就会被判成 notes。
    这份清单里的 doc_type 是**用生产那条分类规则算出来的**，不是我另写一套。
    """
    from doc_utils import _classify_doc

    targets = [{"dir": DEFAULT_IMPORT_DIR, "doc_type": "notes", "files": 0, "exists": False}]
    if DATA_ROOT.exists():
        for p in sorted(DATA_ROOT.rglob("*")):
            if not p.is_dir():
                continue
            rel = p.relative_to(DATA_ROOT).as_posix()
            if rel.count("/") > 1:
                continue
            # question_bank 是题库的源材料（走另一条导入管线），images 是附件目录
            if any(x in rel for x in ("往年题", "images", "question_bank", "__pycache__")):
                continue
            dtype = _classify_doc(p / "占位.txt", DATA_ROOT)
            if dtype is None:       # 「往年题」这类会被跳过的目录，不列出来
                continue
            targets.append({
                "dir": rel,
                "doc_type": dtype,
                "files": sum(1 for f in p.iterdir() if f.is_file()),
                "exists": True,
            })
    return {"code": 0, "msg": "ok", "data": targets}


@app.post("/kb/add-directory")
def kb_add_directory(
    dir_path: str = Query(..., description="目录路径（必须位于 data/ 目录下）"),
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
        result = kb.add_directory(str(resolved))
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
#
# 落点可配置（config.EXPORT_DIR）：相对路径按项目根解析，也可以填绝对路径。
# 写成函数而不是模块常量 —— 常量在 import 时就定死了，测试和环境变量都不好改。
def _export_root() -> FilePath:
    from config import EXPORT_DIR
    p = FilePath(EXPORT_DIR).expanduser()
    return (p if p.is_absolute() else PROJECT_ROOT / p).resolve()


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
    """把当前对话导出成 Markdown

    落点由 `config.EXPORT_DIR` 决定：默认 `<项目>/exports`，
    也可以配成绝对路径（如 `D:\\我的笔记\\对话记录`）。

    返回里带**真实路径**（`path` 相对项目、`abs_path` 绝对），前端会显示出来 ——
    这样"导出到哪了"永远有答案，不依赖浏览器的下载行为。
    """
    md = req.markdown
    if not md and req.messages:
        md = _render_chat_markdown(req.title, req.messages)
    if not md.strip():
        return {"code": 400, "msg": "没有可导出的内容"}

    export_root = _export_root()
    export_root.mkdir(parents=True, exist_ok=True)
    # 时间戳只到秒 —— 同一秒内导出两次会算出同一个文件名，
    # 后一次会把前一次**静默覆盖**掉（写测试时实测到了）。
    # 所以这里保证不重名：撞了就往后加序号。
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    base = f"{_safe_export_name(req.title)}_{stamp}"
    target = export_root / f"{base}.md"
    n = 2
    while target.exists():
        target = export_root / f"{base}_{n}.md"
        n += 1

    # 文件名已过滤分隔符，这里再兜一道：必须仍落在导出目录之内
    if export_root != target.resolve().parent:
        logger.warning("拒绝越界的导出文件名: %s", target.name)
        return {"code": 403, "msg": "非法文件名"}
    target = target.resolve()

    try:
        target.write_text(md, encoding="utf-8")
    except OSError as e:
        logger.warning("对话导出写入失败: %s", e)
        return {"code": 500, "msg": f"写入失败: {str(e)[:120]}"}

    try:
        rel = target.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        # 导出目录被配到了项目之外（EXPORT_DIR 填了绝对路径）—— 这时没有"相对项目"
        # 的形式，直接给绝对路径（前端只是把它显示给用户看）
        rel = str(target)
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
