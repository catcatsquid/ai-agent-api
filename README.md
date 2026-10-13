# AI Agent 学习项目

> 哈工大CS大三学生 AI Agent 应用层开发实习准备项目（10周学习路线）
>
> **当前进度**: 第8周 — 混合检索 + Docker 部署 + 压测 + Web 管理界面
>
> **当前版本**: `v1.0.0` —— 每个版本变了什么见 [CHANGELOG.md](CHANGELOG.md)
>
> **目标**: AI Agent 应用层开发实习（寒假实习11月下旬投递）

---

## 项目概览

本项目从零构建了一个**课程知识库问答 Agent**，覆盖操作系统和计算机网络两门课程。Agent 能根据问题类型自主选择知识库检索或网络搜索，支持流式输出和多轮对话。配套 Web 管理界面，支持知识库增量导入、题库检索和在线问答。

### 核心能力

| 能力 | 说明 |
|------|------|
| RAG 混合检索 | 向量+BM25融合排序 + LLM Rerank + LLM查询改写 |
| Plan-and-Execute | planner 生成 JSON 计划 → executor 批量执行 → synthesizer 综合回答 |
| ReAct 模式 | LLM 循环推理 + 工具调用 + 守卫节点，可切换 `mode=react` |
| Web 搜索 | 课外问题自动触发 web_search，弥补知识库覆盖不足 |
| SSE 流式输出 | 节点级进度推送 + LLM token 级逐字输出 |
| 多轮对话 | 基于 MemorySaver + thread_id 的会话持久化 + 滑动窗口/LLM摘要压缩 |
| 认证系统 | API Key + JWT 混合认证（默认关闭，见下方"鉴权开关"） |
| 中间件 | 限流 + CORS + 请求日志 + trace_id 全链路追踪 |
| Web 管理界面 | 知识库管理(4Tab) + 聊天页面 + 题库检索 |
| 题库系统 | 125题结构化JSON + VLM图片理解 + 知识点分类 |

> ⚠️ **关于两条 Agent 路径 —— 它们现在都能承载刷题，定位如下**
>
> | | LangGraph（Plan-and-Execute / ReAct） | 手写循环（`tutor_agent.py`） |
> |---|---|---|
> | 定位 | **路线内；推荐作为产品主路径** | **第 9 周「手写 Agent」的实现** + 会话守卫更全 |
> | 刷题主流程 | ✅ 跨轮游标、自拟题、讲题都已补齐 | ✅ |
> | 会话守卫 | ❌ 无（没有"没看到题目"重发、没有一次确定性纠偏） | ✅ 有 |
> | 上下文治理 | 摘要压缩（`memory.py`） | 摘要 + 工具结果压缩 + 历史裁剪 |
>
> **两条路径的业务逻辑已合并到 `course_agent/quiz_service.py` 一份实现**，
> 各自只保留"外壳"（一边返回 JSON + 显式参数，一边返回 dict + 会话游标 + `_guide`），
> 所以维护成本不是双份。
>
> 关于手写这条，需要如实说明两点：
> 1. 它最初兼着一个**对照实验**的目的 —— 跟「Python 状态机版学习/复习模式」
>    比一比"LLM 主导是不是更灵活"。这个对照**没有做完**：另一半
>    （LangGraph 版助教 `lg_tutor.py`）从未实现。而状态机版
>    （`learn.py` / `review.py`）已被它取代并删除。
> 2. 它**不是按路线周次做的**，现在也**不从界面进入** —— 界面上已撤掉模式切换，
>    日常只有 BOT 一条产品路径。手写助教的代码与接口**全部保留**
>    （`course_agent/tutor_agent.py`、`POST /tutor/stream`，测试照跑），
>    定位是**冻结的对照实现**：新能力只加到 BOT，它不再跟着演进。
>    撤掉界面入口的直接原因：两条实现经同题对比后**能力已等价**
>    （同一段学生脚本 5 个场景：16好/3坏 vs 18好/2坏），
>    留两个入口只增加"该用哪个"的困惑，而每加一个能力维护两遍是真实成本。
>
>    完整对比与追平过程已整理成文档，但目前**不随仓库分发**
>    （AI 辅助成稿、作者正在逐篇核实 —— 见 [docs/](docs/README.md)）。
>
> 另外，Plan 图曾经在刷题流程上是**坏的**（会编造题目和图片链接），
> 根因是它**主动丢弃工具结果**导致跨轮状态无处安放 —— 排查过程同上。
>
> 项目现状请以**代码与测试**为准：`tests/` 里的断言是当时真实验证过的不变量。

---

## 技术栈

- **Python 3.12**
- **LangGraph + LangChain** — Agent 图结构编排
- **ChromaDB** — 向量数据库持久化
- **rank_bm25** — BM25 关键词检索
- **FastAPI + Uvicorn** — API 服务
- **智谱 GLM-4-Flash** — LLM（含 web_search 能力）
- **智谱 Embedding-3** — 向量嵌入
- **PyMuPDF** — PDF 文字提取
- **Locust** — 压力测试

---

## 项目结构

```
ai-agent-learning/
├── course_agent/                    # LangGraph Agent 核心
│   ├── graph.py                     # 图结构（Plan-Execute + ReAct 双模式）
│   ├── state.py                     # AgentState（messages + plan）
│   ├── nodes.py                     # 节点函数（planner/executor/synthesizer/llm/tools/guard）
│   ├── tools.py                     # 工具定义（search_knowledge + web_search + get_course_info）
│   ├── llm.py                       # LLM 配置
│   ├── server.py                    # FastAPI 服务（/chat + /kb + /qb + /auth + /health）
│   ├── tutor_agent.py               # LLM 主导的对话式学习助手（路线外实验分支，见上文说明）
│   ├── qbank.py                     # 题库访问层（加载/分组/批改依据，唯一数据入口）
│   ├── auth.py                     # 认证系统（API Key + JWT）
│   ├── middleware.py               # 中间件（限流 + CORS + 请求日志 + trace_id）
│   ├── memory.py                    # 对话记忆（滑动窗口 + LLM摘要压缩）
│   ├── errors.py                    # 全局错误处理（重试装饰器 + 安全调用）
│   ├── logger.py                    # 结构化日志系统（contextvars trace_id）
│   ├── question_bank.py             # 题库处理脚本（PyMuPDF + VLM + LLM拆题）
│   └── static/                      # 前端页面
│       ├── chat.html                # 聊天页面（流式输出 + 推理模式切换）
│       ├── kb.html                  # 知识库管理页面（4Tab）
│       └── qb_images/              # 题库图片
├── knowledge_pipeline/              # 知识处理流水线
│   ├── pipeline.py                  # 文档调度器
│   ├── ppt_processor.py             # PPT 处理（文字提取 + VLM 图文理解）
│   ├── textbook_processor.py        # 课本处理（LLM 生成知识点笔记）
│   └── image_understanding.py       # VLM 图片理解模块
├── tests/                           # 单元测试 + 压测
│   ├── test_errors.py               # 错误处理测试
│   ├── test_config.py               # 配置管理测试
│   ├── test_rag_tool.py             # RAG 工具函数测试
│   ├── test_memory.py              # 对话记忆测试
│   └── locustfile.py               # Locust 压测脚本
├── rag_tool.py                      # RAG 工具（混合检索 + Rerank + 查询改写）
├── build_knowledge.py               # 知识库构建脚本（命令行）
├── config.py                         # 全局配置（环境变量管理）
├── Dockerfile                        # Docker 镜像构建
├── docker-compose.yml               # Docker Compose 编排
├── data/                            # 课程知识库数据
│   ├── 操作系统/{ppt,课本,往年题}/
│   ├── 计算机网络/{lys ppt,往年题}/
│   └── question_bank/              # 题库（JSON + 图片）
├── docs/                            # 文档（原有文档已移出仓库，见 docs/README.md）
├── archive/                         # 归档（旧版本代码）
└── README.md
```

---

## 快速开始

### 1. 环境配置

```bash
# 安装依赖 —— 用**锁文件**，与 CI、开发机完全一致
pip install -r requirements.lock

# 配置 .env 文件
# 需要：智谱 GLM-4-Flash API Key + Embedding-3
cp .env.example .env
```

> `requirements.txt` 是人改的**直接依赖**声明；`requirements.lock` 是 uv 生成的**完整锁定**
> （含所有传递依赖、版本精确、并带环境标记 —— 所以**同一份文件在 Windows 与 Linux 上都对**：
> `uvloop` 带 `sys_platform != "win32"`、`pywin32` 带 `sys_platform == "win32"`）。
>
> 改了 `requirements.txt` 之后必须重新生成锁文件：
> ```bash
> pip install uv          # uv 只用于生成锁文件，本身不进锁文件（CI / 镜像里不需要它）
> uv pip compile --universal --python-version 3.12 requirements.txt -o requirements.lock
> ```
> 忘了生成会被 `tests/test_lockfile.py` 直接测出来。

### 2. 构建知识库

```bash
python build_knowledge.py --clear --skip-textbook
```

### 3. 启动服务

```bash
python -m uvicorn course_agent.server:app --port 8003
```

### 4. 访问

| 入口 | URL |
|------|-----|
| 聊天页面 | http://127.0.0.1:8003/chat |
| 知识库管理 | http://127.0.0.1:8003/kb |
| API 文档 | http://127.0.0.1:8003/docs |

### 5. Docker 部署

```bash
docker compose up -d --build
```

Docker 部署的详细说明已随文档移出仓库（见 [docs/](docs/README.md)）；上面的命令可直接使用。

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查（docker-compose 探活用这个，不要用 `/`） |
| GET | `/` | 跳转到 `/chat` |
| POST | `/tutor/stream` | LLM 助教 SSE（对话式刷题：出题/批改/自拟题） |
| POST | `/chat/stream` | SSE 流式问答（节点进度 + 逐字输出） |
| POST | `/auth/login` | 登录获取 JWT |
| GET | `/kb` | 知识库管理页面 |
| GET | `/kb/stats` | 知识库统计 |
| GET | `/kb/documents` | 文档列表 |
| POST | `/kb/search` | 知识库检索测试 |
| GET | `/kb/browse` | 目录浏览 |
| POST | `/kb/add` | 增量导入单文件 |
| POST | `/kb/add-directory` | 增量导入目录 |
| DELETE | `/kb/documents/{source}` | 删除文档 |
| GET | `/qb/list` | 题库文件列表 |
| POST | `/qb/search` | 题库搜索 |
| GET | `/qb/topics` | 题库知识点和来源 |

---

## Agent 架构

### Plan-and-Execute 模式（默认）

```
用户问题 → planner → [route] → executor → [route_after_executor] → synthesizer → END
                         ↓                    ↓
                    search_knowledge     web_search（自适应追加）
```

### ReAct 模式

```
用户问题 → llm → should_continue → tools/force_search/END
                              ↑       ↓
                              └───────┘（循环推理）
```

含守卫节点（force_search）防止 LLM 跳过工具调用，限制1次重试避免无限循环。

### 检索流水线

```
用户问题 → LLM查询改写 → 每个变体混合检索（向量+BM25）→ 合并去重 + 质量过滤
         → 不足时回退搜课本 → LLM Rerank → 格式化输出
```

---

## 知识库

### 统计

| 指标 | 数值 |
|------|------|
| 总文档数 | 19 |
| 总分块数 | **4199** |
| 课本分块 | **2319**（三本课本，占 55%） |
| PPT 分块 | 1827（OS 13份 + 计网 7份 + docx 1份） |
| 笔记分块 | 53（补充知识点笔记） |
| 向量模型 | 智谱 Embedding-3（2048 维单位向量） |
| BM25 索引 | rank_bm25（懒加载） |

> 这组数字是**实测值**，可用 `python debug/check_textbook_coverage.py` 复算。
> 之前这里写的是 1880，是早期建库时的快照，早就过时了。

### 题库

| 指标 | 数值 |
|------|------|
| 总题目数 | 125 |
| 来源数 | 6份试卷（OS 4 + 计网 2） |
| 知识点 | 89 个 |
| 图片文件 | **125张**（与题目一一对应） |
| 题型 | 选择题 80 + 综合题 27 + 整页综合 7 + 填空题 6 + 判断题 5 |
| 自动批改覆盖 | 125/125 题都有批改依据 |

> **题数变化说明**：题库一度是 191 题 / 8 份卷子，后来人工筛掉了两份 ——
> `2019秋本部计算机网络期末考试回忆版`（50 题）与
> `2018秋操作系统期末试题回忆版_WuuTang`（16 题），共 66 题，
> 连同**只被它们引用**的 66 张题图一并删除，现在是 125 题 / 6 份卷子。
>
> 巧合的是：这两份正好就是当初那个过期快照 `question_bank.json` 漏掉的两份，
> 所以删完后题数与该快照一致（都是 125）。但**数据来源完全不同** ——
> 现在是从 6 个单卷文件合并而来，不再依赖那个快照。

> 图片数也清理过两轮：292 → 191（删掉试卷改名留下的 101 张孤儿图，46 MB）
> → 125（随上述两份卷子一起删）。现在文件数与引用数**恒等**，
> 这一点有测试守着（`tests/test_qbank.py`）。

> 题库加载入口是 `course_agent/qbank.py`，它扫描 `data/question_bank/questions_*.json`
> 并去重合并。`question_bank.json` 那个**过期快照已删除**（它只有 125 题，
> 却长得像"全部题库"，是当初"只加载到 125/191"的根因 ——
> 它漏了 2019 计网期末 50 题和 2018 OS WuuTang 16 题）；
> `/qb/list` 里的"全部题库（合并）"现在是虚拟条目，直接由 `qbank` 提供。

### 两条 Agent 路径的工具

LangGraph 路径（`/qa/stream`、`/chat/stream`）与手写助教（`/tutor/stream`）
**共用同一份业务逻辑**（`course_agent/quiz_service.py`），各自只保留外壳。
两边现在都能完成刷题全流程：

| 工具 | 作用 |
|------|------|
| `list_papers` | 按课程列出题库试卷（含题型构成与分组数） |
| `get_paper_questions` | 取一组题：选择/填空/判断 5 题一组，综合题 1 题一组 |
| `grade_answers` | 批改：选择/判断精确判定，填空/综合依参考要点由阅卷模型评分 |
| `explain_question` | 取题面 + 参考答案 + 参考要点（讲题用；出题时刻意不给这些） |
| `prepare_custom_quiz` | 取"临时出题"的素材（题目由模型自己写） |
| `search_knowledge_base` | 知识库检索（可按课程过滤） |
| `get_practice_stats` / `record_practice` | 练习次数统计 |

设计要点：**呈现约束放在工具返回值里，不写死在 system prompt 里。**
一组的题量由 `get_paper_questions` 决定；出题接口只返回题号/题型/知识点/图片
（题干和答案根本不进上下文，模型无从泄题）；每个工具结果的 `_guide` 字段说明
这一步该怎么呈现。因此 system prompt 只描述人设与能力，交互保持自然。

---

## 测试

### 单元测试

```bash
python -m pytest tests/ -v                 # 413 项
node tests/js/auth_block_test.mjs          # 前端认证块行为测试（52 项断言，需 Node）
```

当前 **413 项**通过，覆盖：错误处理、配置管理、RAG 工具、对话记忆、认证与默认拒绝守卫、
路径包含性校验、题库访问层（加载/分组/批改解析/共享工具层）、ReAct 轮数上限、
两张图的会话隔离、跨轮会话游标、依赖方向（循环依赖守卫）、LLM 助教主循环与工具层。

### 数据集与 CI

仓库里的 `data/` 只有**题库 JSON**（251 KB，6 份卷子）纳入版本控制 ——
课本 / 课件 / 题图 / 向量库加起来 300 MB，是刻意 gitignore 的本地数据。
`tests/conftest.py` 会集中声明哪些数据集可用，**缺数据的用例跳过而不是报红**，
并在测试开头打印：

```
数据集: question_bank=有, images=有, vector_store=有
```

所以新克隆可以直接 `pytest tests/` 看到结果（缺图相关用例会被 skip）。
CI（`.github/workflows/ci.yml`）跑 pytest + 前端测试 + 服务探活。

### 鉴权开关

默认 **`AUTH_ENFORCE=false`（本地免登录）** —— 服务只绑 `127.0.0.1`，外面访问不到。
关掉鉴权**不等于**裸奔：真正危险的那条链（`/kb/add` 任意文件入库 → `/kb/search` 外泄）
靠的是处理函数内部的路径校验 `resolve_within_data()`，与鉴权解耦，
实测 `README.md` / `.env` / `../../etc/passwd` 在无鉴权模式下依然全部 403。

对外部署前请设置：

```bash
AUTH_ENFORCE=true
ADMIN_PASSWORD=<你的管理密码>      # 前端会弹登录框，用它换 JWT
```

### 压力测试

```bash
python -m locust -f tests/locustfile.py --host http://localhost:8003 \
  --users 50 --spawn-rate 10 --run-time 30s --headless --html docs/benchmark_report.html
```

---

## 学习进度

### 已完成

| 周次 | 主题 | 核心产出 |
|------|------|---------|
| 第1周 | 基础 API 服务 | FastAPI + SSE 流式响应 |
| 第2周 | Prompt Engineering | RAG + 向量搜索 |
| 第3周 | ReAct 多工具 Agent | 工具编排 + Function Calling |
| 第4周 | LangGraph 课程 Agent | ReAct 循环 + RAG 集成 + 知识库流水线 + 守卫节点 + B3 测试 |
| 第5周 | 多工具扩展与自主规划 | Web 搜索 + Plan-and-Execute + 多轮对话持久化 |
| 第6周 | SSE 流式接口 + v1 产出 | 双模式流 + 端到端测试 + 知识库盲区修复 |
| 第7周 | 健壮性 + 工程化 | 全局错误处理 + 结构化日志 + trace_id + 配置规范化 + 单元测试(88项) |
| 第8周 | 混合检索 + Docker + 压测 | 向量+BM25融合 + LLM Rerank + 查询改写 + Docker部署 + 压测报告 + Web管理界面 + 题库系统 |

### 路线外（不计入周次）

| 内容 | 性质 | 状态 |
|------|------|------|
| LLM 助教 · 对话式刷题 | **需求驱动的产品特性**（我要一个不用反复声明学习流程的学习/复习 Agent），同时兼做状态机版的对照 | 可用；对照未完成（缺 LangGraph 版） |
| 题库访问层 `qbank.py` | 上面那个特性和已有题库功能的共同底座 | 两个模式共用 |
| 全局缺陷修复 | 排查过程中顺带产出 | 已整理成文档（当前不随仓库分发，见 [docs/](docs/README.md)） |

### 待完成

| 周次 | 主题 |
|------|------|
| 第9周 | 开源阅读 + 手写 Agent |
| 第10周 | 面试冲刺 |

> 说明：第 9 周是**两件事**——读开源项目 + 手写一个 Agent。
> `tutor_agent.py` 里那个手写 tool-calling 循环在形式上覆盖了"手写 Agent"，
> 但它不是按第 9 周的计划做的，也没有配套的开源阅读，所以第 9 周仍算未完成。
