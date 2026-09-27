# AI Agent 学习项目

> 哈工大CS大三学生 AI Agent 应用层开发实习准备项目（10周学习路线）
>
> **当前进度**: 第7周 — 全局错误处理 + 结构化日志 + 配置规范化 + 单元测试
>
> **目标**: AI Agent 应用层开发实习（寒假实习11月下旬投递）

---

## 项目概览

本项目从零构建了一个**课程知识库问答 Agent**，覆盖操作系统和计算机网络两门课程。Agent 能根据问题类型自主选择知识库检索或网络搜索，支持流式输出和多轮对话。

### 核心能力

| 能力 | 说明 |
|------|------|
| RAG 知识库检索 | ChromaDB 向量库，1320 块，查询扩展 + 多路召回 + 分层检索 |
| Plan-and-Execute | planner 生成 JSON 计划 → executor 批量执行 → synthesizer 综合回答 |
| Web 搜索 | 课外问题自动触发 web_search，弥补知识库覆盖不足 |
| SSE 流式输出 | 节点级进度推送 + LLM token 级逐字输出 |
| 多轮对话 | 基于 MemorySaver + thread_id 的会话持久化 |
| ReAct 模式 | 保留 Week 4 的 ReAct 图，可切换 `mode=react` 使用 |

---

## 技术栈

- **Python 3.12**
- **LangGraph + LangChain** — Agent 图结构编排
- **ChromaDB** — 向量数据库持久化
- **FastAPI + Uvicorn** — API 服务
- **智谱 GLM-4-Flash** — LLM（含 web_search 能力）
- **智谱 Embedding-3** — 向量嵌入
- **PyMuPDF** — PDF 文字提取
- **pypdf** — PDF 备用提取

---

## 项目结构

```
ai-agent-learning/
├── course_agent/                    # LangGraph Agent 核心
│   ├── graph.py                     # 图结构（Plan-and-Execute + ReAct 双模式）
│   ├── state.py                     # AgentState（messages + plan）
│   ├── nodes.py                     # 节点函数（planner/executor/synthesizer/llm/tools/guard）
│   ├── tools.py                     # 工具定义（search_knowledge + web_search + get_course_info）
│   ├── llm.py                       # LLM 配置
│   ├── server.py                    # FastAPI 服务（/qa/ask + /qa/stream）
│   ├── errors.py                    # 全局错误处理（重试装饰器 + 安全调用）
│   └── logger.py                    # 结构化日志系统
├── knowledge_pipeline/              # 知识处理流水线
│   ├── pipeline.py                  # 文档调度器
│   ├── ppt_processor.py             # PPT 处理（文字提取 + VLM 图文理解）
│   ├── textbook_processor.py        # 课本处理（LLM 生成知识点笔记）
│   └── image_understanding.py       # VLM 图片理解模块
├── tests/                           # 单元测试（pytest）
│   ├── test_errors.py               # 错误处理测试
│   ├── test_config.py               # 配置管理测试
│   └── test_rag_tool.py             # RAG 工具函数测试
├── rag_tool.py                      # RAG 工具（查询扩展 + 多路召回 + 批量 embedding）
├── build_knowledge.py               # 知识库构建脚本（命令行）
├── config.py                         # 全局配置（环境变量管理）
├── data/                            # 课程知识库数据
│   ├── 操作系统/{ppt,课本,往年题}/
│   └── 计算机网络/{lys ppt,往年题}/
├── debug/                           # 测试脚本
│   ├── 003-exam-qa-test-v7.py       # 最新版考试测试
│   ├── 017-test-plan-execute.py     # Plan-Execute 测试
│   ├── 019-test-adaptive-plan.py    # 自适应规划层测试
│   └── 020-test-error-handling.py   # 错误处理测试
├── docs/                            # 文档与学习资料
│   ├── week1-6-review/              # 前6周复习
│   └── reference/                   # 参考资料
├── archive/                         # 归档（旧版本代码）
│   ├── week1-week3/                 # Week 1-3 学习历程
│   ├── week4-paper_agent/           # Week 4 Agent
│   ├── week5-web_app/               # Week 5 Web App（已废弃）
│   └── debug_old_versions/         # debug 旧版本脚本
└── README.md
```

---

## 快速开始

### 1. 环境配置

```bash
# 克隆项目后，配置 .env 文件
# 需要的 API Key：智谱 GLM-4-Flash（含 web_search）+ Embedding-3
```

### 2. 构建知识库

```bash
# 完整构建（含课本，较慢）
python build_knowledge.py

# 快速构建（跳过课本，~2 分钟）
python build_knowledge.py --clear --skip-textbook

# 流水线模式（PPT VLM 图片理解）
python build_knowledge.py --pipeline --skip-textbook
```

### 3. 启动服务

```bash
# Agent API 服务（端口 8003）
python -m uvicorn course_agent.server:app --port 8003

# 运行单元测试
python -m pytest tests/ -v
```

### 4. API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| POST | `/qa/ask` | 同步问答（支持 plan/react 模式 + thread_id 多轮） |
| POST | `/qa/stream` | SSE 流式问答（节点进度 + 逐字输出） |

**请求参数:**

```json
{
  "query": "什么是死锁？四个必要条件是什么？",
  "mode": "plan",          // "plan" = Plan-and-Execute（默认），"react" = ReAct
  "thread_id": null         // 传入相同值续接多轮对话，null 则新建会话
}
```

接口文档: http://127.0.0.1:8003/docs

---

## Agent 工具

| 工具 | 功能 |
|------|------|
| `search_knowledge` | 搜索课程知识库（OS + 计网），支持 doc_types 过滤 |
| `web_search` | 网络搜索（智谱 GLM-4-Flash web_search API） |
| `get_course_info` | 获取知识库统计信息 |

---

## Agent 架构

### Plan-and-Execute 模式（默认）

```
用户问题 → planner（生成JSON计划）→ executor（批量执行工具）→ synthesizer（综合回答）→ END
                                         ↑
                                    search_knowledge / web_search
```

planner 根据问题类型决定使用哪些工具：
- 课程知识题 → 只规划 `search_knowledge`
- 课外问题 → 规划 `web_search`
- 混合问题 → 两者都规划

### ReAct 模式（可切换）

```
用户问题 → llm → should_continue → tools/force_search/END
                              ↑       ↓
                              └───────┘（循环推理）
```

Week 4 的 ReAct 图保留，通过 `mode=react` 切换。含守卫节点（force_search）防止 LLM 跳过工具调用。

### SSE 流式接口

```
客户端 ← SSE 事件流 ← 服务端
  type=start    // 会话开始
  type=node     // 节点执行进度（planner → executor → synthesizer）
  type=token    // LLM 逐字输出（仅 synthesizer 的 token）
  type=done     // 完成
  type=error    // 错误
```

通过 `stream_mode=["updates", "messages"]` 同时捕获节点事件和 LLM token。

---

## 知识库

### 数据来源

| 类型 | 来源 | 说明 |
|------|------|------|
| PPT | 课程课件 PDF | 文字提取 + VLM 图文理解（可选） |
| notes | 手写知识点笔记 | 针对图表密集章节补充 |
| textbook | 课本 PDF | OSTEP + 计算机网络（自顶向下方法） |
| docx | 复习整理文档 | 计算机网络复习材料 |

### 知识库统计

- 总分块数：1320
- 分块策略：fixed/300（300 字符固定长度）
- 向量模型：智谱 Embedding-3
- 批量 embedding：每 20 块批量调用，构建时间 ~2 分钟

### 检索策略

- **查询扩展**：对原始查询进行 LLM 扩展，生成多个相关查询
- **多路召回**：多个查询结果合并去重
- **分层检索**：优先搜索 ppt + notes，结果不足时 fallback 到 textbook
- **隐藏分数**：score 不暴露给 LLM，避免 LLM 根据 score 判断是否回答

### 知识库覆盖

**操作系统**: 进程管理、线程、调度算法（FCFS/SJF/RR/MLFQ）、死锁（预防/避免/检测/银行家算法）、内存管理（分页/分段/虚拟内存/页面置换）、文件系统（inode/分配方式/目录结构）、磁盘调度、I/O 管理、同步互斥

**计算机网络**: 物理层、数据链路层（CSMA/CD、MAC、ARP）、网络层（IP/路由/子网划分）、传输层（TCP/UDP/拥塞控制）、应用层（HTTP/DNS/CDN）、无线网络

---

## 测试结果

### SSE 流式接口测试（3/3 通过）

| 场景 | 问题 | 结果 |
|------|------|------|
| 课程知识题 | 死锁四个必要条件 | 1 次工具调用，回答正确，来源标注准确 |
| 闲聊题 | 你好你是谁 | 0 步规划，直接回答，跳过工具 |
| 多轮对话 | 怎么预防死锁？ | thread_id 续接成功，理解上下文 |

### web_search 工具测试（3/3 通过）

| 场景 | 问题 | web_search | 结果 |
|------|------|-----------|------|
| 课程知识题 | 虚拟内存是什么？ | 否 | PASS |
| 课外问题 | 2024年Linux内核CFS新变化？ | 是 | PASS |
| 混合问题 | TCP拥塞控制 + BBR算法区别？ | 是 | PASS |

### 知识库盲区修复验证（6/6 通过）

| 主题 | 补充前 | 补充后 |
|------|--------|--------|
| 银行家算法 | 3/3 低分 | 回答正确 |
| 页面置换算法 | 4/4 低分 | 回答正确 |
| 磁盘调度算法 | 3/3 低分 | 回答正确 |
| 文件系统 | 3/3 低分 | 回答正确 |
| 多级反馈队列 | 1/1 低分 | 回答正确 |
| 上下文切换 | 1/1 低分 | 回答正确 |

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
| 第6周 | SSE 流式接口 + v1 产出 | 双模式流（节点+token）+ 端到端测试 + 知识库盲区修复 |
| 第7周 | 健壮性 + 工程化 | 全局错误处理 + 结构化日志 + 配置规范化 + 单元测试(50项) |

### 待完成

| 周次 | 主题 |
|------|------|
| 第7-8周 | 健壮性 + 工程化 + Docker |
| 第9周 | 开源阅读 + 手写 Agent |
| 第10周 | 面试冲刺 |

---

## 架构演进

```
Week 1          Week 2          Week 3          Week 4          Week 5          Week 6
简单API    →    RAG集成    →    多工具     →    LangGraph  →    自主规划    →    流式输出
线性处理        向量搜索        工具编排        图结构执行      Plan-Execute      SSE双模式
                                                              +多轮持久化        +v1产出
```

### 关键技术决策

- **LangGraph 双模式**：Plan-and-Execute 适合多工具批量场景，ReAct 适合逐步推理，两种图共存可切换
- **守卫节点**：防止 LLM 在自信话题上跳过工具调用，强制至少 1 次知识库检索
- **批量 embedding**：从逐块调用（40分钟）优化为批量调用（2分钟）
- **隐藏检索分数**：不暴露 score 给 LLM，避免 LLM 根据 score 自行判断是否回答
- **知识库分层**：ppt/notes 优先，textbook 兜底，兼顾检索质量和速度
