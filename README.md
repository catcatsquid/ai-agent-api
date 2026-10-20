# 知识寻回BOT —— 课程资料变成会讲课、会出题、会批改的助教

> 把课件、课本、历年真题 PDF 变成一个真的能用的课程助教：按章节讲、分组出题、批改、讲题、记进度，讲不清就联网查。
> 仓库自带 985 项离线测试和 34 条行为评测用例（走真实 HTTP + SSE）。

`Python 3.12` · `LangGraph` · `FastAPI` · `ChromaDB` · `rank_bm25` · `GLM-4-Flash` · `Docker`


**目录**

- [Demo](#demo)
- [核心能力](#核心能力)
- [架构与代码结构](#架构与代码结构)
- [稳定性与安全](#稳定性与安全)
- [量化效果](#量化效果)
- [快速开始](#快速开始从零复现)
- [Docker](#docker两种别混用)
- [测试与工程化](#测试与工程化)
- [已知限制与设计取舍](#已知限制与设计取舍)
- [Roadmap](#roadmap)
- [这个项目为什么长这样](#这个项目为什么长这样)

---

## Demo

**视频（52 秒）：[demo/demo-full.mp4](demo/demo-full.mp4)**

| RAG混合检索 | 问答：先检索再作答（1 个工具调用） |
|---|---|
| ![](docs/screenshots/03-retrieval.png) | ![](docs/screenshots/04-qa.png) |

| 出题：题号与知识点取自题库 | 学习进度：学到第几块、刷到第几组 |
|---|---|
| ![](docs/screenshots/01-quiz.png) | ![](docs/screenshots/02-progress.png) |



本地启动后访问 `http://127.0.0.1:8003/chat` 直接体验。

---

## 核心能力

### 功能

| 场景 | 能力 |
|---|---|
| 学习 | 按章节讲、书签续读、章末小测、随时不懂就问 |
| 解题 | 分组出题、多种答案格式批改、按点讲解、按知识点抽题、错题重做、自拟练习 |
| 复习 | 知识点速查、针对薄弱环节练几道 |
| 知识问答 | 先检索再作答；库里没有就两级回退，仍不足则联网 |
| 进度 | 跨会话、跨窗口接着上次；换场景不断线 |
| 意图路由 | 一句话切换场景（学习 / 解题 / 复习），不需要手动切模式开关 |
| 知识库 | 上传、路径导入、目录浏览、检索测试、删除 |

### 技术栈

| 层 | 选型 |
|---|---|
| 接入 | FastAPI · SSE 五类事件（`start` / `node` / `content` / `done` / `error`） |
| 编排 | LangGraph：Plan-and-Execute+ ReAct，双图各自独立 checkpointer |
| 检索 | ChromaDB + BM25 混合融合 → LLM rerank → 两级回退 |
| 模型 | GLM-4-Flash（对话）· embedding-3（向量）· glm-4v（图表理解） |
| 状态 | SQLite checkpointer 持久化 + JSON 进度账本 |
| 前端 | 原生 JS，零框架（`chat.html` / `kb.html`） |
| 工程 | pytest 985 项 · eval 34 条判据化用例 · GitHub Actions · Docker |

---

## 架构与代码结构

![架构总览](docs/screenshots/architecture.png)

### 代码结构

| 层 | 位置 | 负责什么 |
|---|---|---|
| 入口 | `course_agent/server.py` | HTTP 入口：26 个路由、SSE 推流、中间件挂载、静态页 |
| 编排 | `course_agent/graph.py` | 两张图的组装与 checkpointer（Plan / ReAct 各一套库文件） |
| | `course_agent/state.py` | `AgentState` 定义 |
| | `course_agent/nodes.py` | 节点实现：planner / executor / synthesizer / llm / tool，以及各类行为守卫 |
| 业务 | `course_agent/quiz_service.py` | 刷题业务的唯一实现：取题、分组、批改、讲题、按知识点抽题 |
| | `course_agent/tools.py` | 17 个 `@tool` 的薄适配；业务逻辑一律下沉到 `quiz_service` |
| | `course_agent/qbank.py` | 题库读取层：扫描 `questions_*.json`、按课程索引、解析答案与分组 |
| | `course_agent/question_bank.py` | 离线题库构建：PDF → 题号定位 → 裁图 → VLM/LLM → 结构化 JSON |
| 状态 | `course_agent/progress.py` | 学习进度账本：书签、刷题位置、错题，跨会话与跨窗口 |
| | `course_agent/memory.py` | 上下文治理：滑动窗口、摘要压缩、摘要缓存按消息 id 摊还 |
| | `course_agent/courses.py` | 课程注册表：课程名来自 `data/*/` 目录，加课不改代码 |
| 检索 | `rag_tool.py` | 全项目唯一检索入口：混合检索、rerank、查询改写、两级回退 |
| | `doc_utils.py` | 文档加载 / 清洗 / 分类（ppt · textbook · notes）、章节推断 |
| | `build_knowledge.py` | 建库 CLI：增量（md5 跳过未变）/ 全量 / 单文件 |
| | `knowledge_pipeline/` | 离线流水线：PPT 按页 + 课本分块 + VLM 图表理解 |
| 横切 | `course_agent/llm.py` | 智谱 HTTP 直调 + `SimpleLLM`（`BaseChatModel` 子类）+ 流式 |
| | `course_agent/auth.py` · `middleware.py` · `logger.py` · `errors.py` | 鉴权 / 限流与 trace_id / 结构化日志 / 三层错误模型与退避 |
| 前端 | `course_agent/static/` | `chat.html` / `kb.html`（原生 JS，零框架） |
| 对照 | `frozen/tutor_agent.py` | 冻结的手写 tool-calling 循环（对照实现，不参与运行） |
| 测试 | `tests/` · `eval/` · `demo/` | 单元/契约/守卫（全离线）· 行为评测用例与归档报告 · 演示脚本与数据集 |


### 对外接口

对表面很窄 —— 一共 **6 组**端点、26 个路由（含 `/`、`/chat`、`/kb` 三个页面）：

| 组 | 端点 | 说明 |
|---|---|---|
| 问答 | `POST /qa/stream` · `POST /qa/ask` | 流式 / 非流式 |
| 知识库 | `GET·POST·DELETE /kb/*` | 统计 / 课程（列表、新增）/ 文档 / 检索测试 / 增量导入 / 上传 / 浏览 / 删除 |
| 题库 | `GET·POST /qb/*` | 列表 / 搜索 / 知识点与来源 |
| 会话 | `GET·DELETE /chat/conversations*` · `POST /chat/export` | 列表 / 详情 / 删除 / 导出 |
| 认证 | `POST /auth/login` | 换 JWT（默认关闭，本地免登录） |
| 运维 | `GET /health` · `GET /progress` | 探活 / 只读进度快照 |


---

## 稳定性与安全

- **防 Agent 失控**：工具调用轮数硬上限；guard node 强制课程问题先检索，防止模型凭记忆作答
- **上下文治理**：读时裁剪、摘要缓存按消息 id 摊还、tool 消息与 `tool_calls` 严格配对
- **SSE 正文过滤（四层判据）**：来源白名单 + 内部调用标签 + 类型白名单 + 非空过滤，工具返回原文不会泄漏进正文频道
- **路径安全**：所有文件操作经 `resolve()` + `relative_to(data/)` 校验，防线设在 API 边界而非库层
- **可观测性**：`contextvars` 存 `trace_id`，贯穿中间件 / 日志 / 错误处理；三层错误模型 + 指数退避重试 + 降级

---

## 量化效果

| 指标 | 结果 | 口径 |
|------|------|------|
| 首字响应 TTFT | **7.5s**（p50）/ 16.6s（p95） | 1.11.1 归档 |
| 单轮总时长 | 24.2s（p50） | 1.11.1 归档 |
| 逐字流式 | 正文逐字出；**出题轮故意不流式**（正文随 `done` 一次给全） | 1.11.x 起 |
| 行为评测 | 判据 **900 / 903** 通过；29 条用例中 26 条稳定 | 1.11.1 归档，每用例 3 attempts |
| 补搜提示准确率 | 域内/域外 12 条样本 **12/12** | 1.7.x 时期修的单项，之后未重测 |
| 测试 | **985** 项 pytest + 107 项前端断言 + 服务探活 | CI 每次 push 自动跑 |

> 评测走真实 HTTP + SSE + 真模型，判据全部机器可判；报告按版本存档，见 [eval/reports/INDEX.md](eval/reports/INDEX.md)。
> 历史对照只留一个：v1.7.1 → v1.7.2 首响 33s → 6.5s（`content` 事件 0 → 517–613 个/轮），证据在两份归档报告里。
> 3 条不稳定用例是模型抖动（`intent_followup_reference` / `knowledge_answer_streams` / `quiz_b11_wrong_redo`），不是回归。

---

## 快速开始（从零复现）

### 0. 前置

| 需要 | 要求 | 为什么 |
|---|---|---|
| Python | **3.12.x** | 锁文件按 3.12 生成；3.13 有包缺 wheel，3.11 会静默降级依赖 |
| Git | 任意版本 | 也可以直接下 ZIP（解压后**多一层目录**，别踩） |
| 磁盘 | ≥ 300 MB | 课件 + 向量库；课件不入库，需自备 |
| 智谱 API Key | — | 免费申请约 2 分钟：[docs.bigmodel.cn](https://docs.bigmodel.cn) |

### 1. 拿到代码并建虚拟环境

```bash
git clone <仓库地址> ai-agent-learning && cd ai-agent-learning

python -m venv .venv
.venv\Scripts\activate          # Windows
source .venv/bin/activate       # macOS / Linux

pip install -r requirements.lock
```

> **别跳过 venv。** 用系统 Python 装，很容易和别的项目的 `chromadb` / `pydantic` 版本打架，
> 而报错都是"import 失败"，极难定位。锁文件保证你装出来的和 CI、镜像完全一致。

### 2. 配置 Key

```bash
cp .env.example .env      # Windows: copy .env.example .env
```

打开 `.env`，**只需填一行**：

```
ZHIPU_API_KEY=你的key
```

其余全部可选：`API_KEY` / `JWT_SECRET`（鉴权，本地不用）、`RATE_LIMIT_PER_MINUTE`（限流）、
`PROGRESS_FILE`（学习进度文件，演示时用它隔离真实进度）。

### 3. 准备课程数据（最容易卡住的一步）

> **只想先看看效果？** 用 [Docker 演示镜像](#docker两种别混用) 可以跳过第 3、4 步。

课件 PDF、试卷 PDF、题图、题库 JSON、向量库**都不入库**（版权 + 体积，合计约 300 MB），需要自己准备。目录结构固定：

```
data/操作系统/ppt/*.pdf          # 课件
data/计算机网络/**/*.pdf
data/<课程>/往年题/*.pdf          # 试卷，用来生成题库
```

- **想刷题** → 先有题库：`python -m course_agent.question_bank data/操作系统/往年题/`
- **想问答** → 放好课件，跑第 4 步建库

> 题图缺失时，出题那一轮会没图（题库 JSON 里有引用，但文件不在）—— 这是预期，不是 bug。

### 4. 建库（一次性）

```bash
python build_knowledge.py --clear --skip-textbook
python build_knowledge.py --stats        # 验证：分块数必须 > 0
```

- 会按批调用 embedding，**消耗额度**，全量约 20 分钟
- `--skip-textbook` 跳过三本课本（最慢的部分）；要全量就去掉它
- 想先快跑通：`--data-dir ./data/操作系统` 只入一门课

### 5. 启动

```bash
python -m uvicorn course_agent.server:app --port 8003
```

| 入口 | URL |
|---|---|
| 聊天页面 | http://127.0.0.1:8003/chat |
| 知识库管理 | http://127.0.0.1:8003/kb |
| API 文档 | http://127.0.0.1:8003/docs |


### 6. 怎么确认"真的跑对了"

```bash
python -m pytest tests/ -q          # 全离线；缺数据的用例 skip 而不是报红
curl http://127.0.0.1:8003/health   # knowledge_chunks > 0 = 库建好了

curl -N -X POST http://127.0.0.1:8003/qa/stream \
  -H 'Content-Type: application/json' \
  -d '{"query":"进程和线程的区别","mode":"plan"}'
  # 看 content 是否逐字出现；整段一次性出现 = 流式没生效
```

### 7. 起不来？先查这五条

| 现象 | 大概率原因 | 解法 |
|---|---|---|
| `import` 失败 / pydantic 版本冲突 | 没在 venv 里装 | 删 `.venv` 重来，确认命令行有 `(.venv)` 前缀 |
| `/health` 里 `knowledge_chunks: 0` | 没建库 / `--data-dir` 指错 | 重跑第 4 步 |
| 问答报 401 或额度错误 | `.env` 里 Key 没填、填错行、带了引号 | 只留 `ZHIPU_API_KEY=xxx` 一行 |
| 端口 8003 被占 | 上一次的服务没关 | 换 `--port 8004`，或关掉旧进程 |
| 出题没有图 | 题图未入库（`.gitignore`） | 属预期，见第 3 步 |

### Docker（两种，别混用）

仓库带两份构建文件，用途不同：

| | 演示镜像 | 自己的完整部署 |
|---|---|---|
| 构建文件 | `Dockerfile.demo` + `docker-compose.demo.yml` | `Dockerfile` |
| 数据 | 挂 `demo/seed/`（随仓库分发：5 份自写笔记 + 11 道原创题 + 预建向量库） | 挂你自己的 `data/` 与 `vector_store/` |
| 需要 API Key | 不需要，向量库已预建 | 需要，建库要调 embedding |
| 用途 | 别人 clone 下来一条命令就能看到检索与刷题在工作 | 装自己的课件，当日常工具用 |

演示镜像：

```bash
docker compose -f docker-compose.demo.yml up -d --build
```

起来就能看 `/chat`。`demo/seed/data` 是**只读**挂载，所以 `/kb/add`、`/kb/upload` 会失败 —— 故意的，防止演示时改坏数据集；`demo/seed/vector_store` 可写（Chroma 打开 sqlite 要写 `-wal`/`-shm`），`runtime` 用命名卷保存会话与进度，删掉卷就清空演示状态。

自己的完整部署：`Dockerfile` 装了 gcc/g++ 并 `COPY data/`，它对应的编排不随仓库分发（各人挂载路径不同），用 `docker run` 或自己的 compose 挂 `./data` 与 `./vector_store` 即可。**前提是宿主机先建过库**，否则起来检索全空（`/health` 会显示 `knowledge_chunks: 0`）。

---

## 测试与工程化

- **测试四层**：单元测试（外部依赖全 mock）/ 接口契约测试 / 历史 bug 回归 / 架构守卫（依赖方向、图隔离、锁文件同步）
- **三条环境守卫**（session 级）：真实会话库不被测试写入、真实 `data/` 不被测试改动、服务运行中拒绝开跑（防 Chroma 并发写损坏）
- **数据可用性**：课件、试卷、题库、向量库都不入库，依赖缺失的用例自动 skip 而非报红，新克隆即可跑通（刷题相关的用例除外）
- **CI**：GitHub Actions 每次 push 跑 pytest + 前端测试 + 服务冒烟；`txt / lock / 环境`三方一致性有测试守着
- **可复现交付**：锁文件用 `uv pip compile --universal`（带 `sys_platform` 标记，Windows/Linux 同一份）→ CI 与镜像都从它装；
  镜像分层先依赖后源码（改代码不触发重装）；`vector_store/` 与 `data/` 走 volume（重建要耗 embedding 额度）；
  healthcheck 用 `python -c requests` 而非 `curl`（slim 镜像里没有 curl）
- **压测**：Locust 三级虚拟用户（只读 / 检索 / 问答），问答级默认关闭防烧额度

---

## 已知限制与设计取舍

- 课件、课本、题库、题图因版权与体积不入库，都要本地准备；题库由试卷 PDF 生成，知识库需本地建库

- 文档解析只有文本层：没有 OCR，也没有表格结构识别，扫描件和复杂表格会退化。题库那条线用"每题裁一张图"绕开了它，知识库那条线没有绕开

- 首响约 7.5 秒（1.11.1 归档 p50）。为了让答案基于检索而不是模型记忆，planner 与检索必须在写答案之前跑完，这段是静默的 —— 这是"先检索再作答"的代价，暂时不是能调掉的延迟

- ReAct 偶发"嘴上说要调工具、实际没调"：已补一轮强制调用，残留是流式打字时可能闪过一下工具调用原文。根治要给 SSE 加回退缓冲，那会让流式变粗，所以保留现状

- 评测要真模型、真额度，因而不进 CI，报告靠手动跑完归档。这也是评测报告做不到每次提交自动更新的原因

- 讲得好不好没有度量：34 条判据只管"该调的工具调了没、该出现的出现没"，不管"讲得清不清楚"

- 单用户本地部署设计，多用户需要在会话与配额层扩展

---

## Roadmap

- [ ] **重跑 34 条用例、归档当前版本报告** —— 最新归档停在 1.11.1 的 29 条

- [ ] 检索层 Recall@K / MRR：把"混合检索比纯向量好多少"变成数字

- [ ] ReAct vs Plan-and-Execute 同题对比：token / 轮次 / 准确率 / 失败模式分类

- [ ] 给知识库分块加 `course` 元数据，去掉"按文件名反查课程"这层耦合（只改 metadata，不用重新 embedding）

- [ ] 新增课程后能在 Web 上传试卷 PDF 并生成题库

- [ ] 讲得好不好：LLM 裁判 + 人工标注校准，先拿 10–20 条对齐再当参考值

- [x] 全局学习进度、刷题位置记忆与错题本 —— 1.8.0 / 1.9.0 已交付

---

## 这个项目为什么长这样

- **题库用历年真题，不用模型自拟题** —— 复习阶段刷真题对考试最有效；模型自拟的题没有权威答案，批改就变成自己给自己打分。
- **状态放在模型之外** —— 出题结果、题图、学习进度都存在显式状态和服务端账本里，不指望模型从对话里记住。换个窗口打开，能接着上次的位置。
- **课程问题强制先检索，一切以课件为准** —— 答案以本课程课件为来源，而不是模型的通用记忆；课件不够退到课本，仍不足才联网。
- **同时保留 Plan-and-Execute 和 ReAct 两条图** —— 开发时为了对比学习这两种 Agent 编排：业务逻辑共用 `quiz_service.py`，差别只在编排层；界面上可以直接切换，两张图各用独立 checkpointer。
- **项目里为什么有一个冻结的tutor_agent** —— 项目里曾有两条 Agent 路径：LangGraph 的图，和手写的工具调用循环。用同一段学生脚本跑了5个场景做等价性验证后决定删除其入口,只保留模块与测试，移到 frozen/。

更多本项目开发的来龙去脉和走过的弯路，见.....(留一个地方以后放技术博客)；评测细节见 [eval/README.md](eval/README.md)。
