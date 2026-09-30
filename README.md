# AI Agent 学习项目

> 哈工大CS大三学生 AI Agent 应用层开发实习准备项目（10周学习路线）
>
> **当前进度**: 第8周 — 混合检索 + Docker 部署 + 压测 + Web 管理界面
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
| 认证系统 | API Key + JWT 混合认证 |
| 中间件 | 限流 + CORS + 请求日志 + trace_id 全链路追踪 |
| Web 管理界面 | 知识库管理(4Tab) + 聊天页面 + 题库检索 |
| 题库系统 | 224题结构化JSON + VLM图片理解 + 知识点分类 |

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
│   ├── server.py                    # FastAPI 服务（/chat + /kb + /qb + /auth）
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
├── docs/                            # 文档
│   ├── deployment.md               # Docker 部署指南
│   └── reference/                   # 参考资料
├── archive/                         # 归档（旧版本代码）
└── README.md
```

---

## 快速开始

### 1. 环境配置

```bash
# 配置 .env 文件
# 需要：智谱 GLM-4-Flash API Key + Embedding-3
cp .env.example .env
```

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

详见 [Docker 部署指南](docs/deployment.md)。

---

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
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
| 总分块数 | 1880 |
| PPT 分块 | 1827（OS 13份 + 计网 7份 + docx 1份） |
| 笔记分块 | 53（补充知识点笔记） |
| 向量模型 | 智谱 Embedding-3 |
| BM25 索引 | rank_bm25（懒加载） |

### 题库

| 指标 | 数值 |
|------|------|
| 总题目数 | 224 |
| 来源数 | 7份试卷（OS 5 + 计网 2） |
| 图片文件 | 219张 |
| 题型 | 选择题 + 综合题 |

---

## 测试

### 单元测试（88项通过）

```bash
python -m pytest tests/ -v
```

覆盖：错误处理、配置管理、RAG工具、对话记忆。

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

### 待完成

| 周次 | 主题 |
|------|------|
| 第9周 | 开源阅读 + 手写 Agent |
| 第10周 | 面试冲刺 |
