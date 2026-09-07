# AI Agent 学习项目

AI Agent 应用开发学习路线的代码仓库，包含从零开始搭建的三个核心模块。

## 模块一：AI 聊天 API 服务（第 1 周）

基于 FastAPI 搭建的 AI 聊天后端服务，支持 SSE 流式响应。

### 技术栈

- Python 3.12
- FastAPI（Web 框架）
- Uvicorn（ASGI 服务器）
- 智谱 GLM-4-Flash API（大模型）
- SSE（Server-Sent Events 流式响应）

### 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | 健康检查 |
| POST | `/chat` | 普通聊天（一次性返回） |
| POST | `/chat/stream` | 流式聊天（打字机效果） |
| GET  | `/test` | 流式聊天测试页面 |

### 启动

```bash
pip install -r requirements.txt
python -m uvicorn api_server:app --reload
```

访问接口文档：http://127.0.0.1:8000/docs

---

## 模块二：RAG 问答 API 服务（第 2 周）

基于向量检索的 RAG（检索增强生成）问答服务，支持自定义知识库。

### 技术栈

- FastAPI（Web 框架）
- 智谱 Embedding-3（向量化模型）
- 智谱 GLM-4-Flash（生成模型）
- 余弦相似度检索

### 核心流程

```
用户问题 → 向量化 → 检索 Top-K 相关文档 → 拼入 Prompt → 大模型生成回答 → 返回答案+来源
```

### 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | 健康检查 |
| POST | `/rag/ask` | RAG 问答（基于知识库回答） |

### 启动

```bash
pip install -r requirements.txt
python rag_api.py
```

访问接口文档：http://127.0.0.1:8001/docs

### 可调参数

- `query`：用户问题（必填）
- `top_k`：检索条数，默认 3
- `temperature`：回答温度，默认 0.1

---

## 模块三：多工具 Agent API 服务（第 3 周）

基于 ReAct 范式的多工具 Agent 服务，支持天气查询、计算器、知识搜索、时间查询、翻译5种工具的自主编排。

### 技术栈

- FastAPI（Web 框架）
- 智谱 GLM-4-Flash（大模型 + Function Calling）
- ReAct 范式（Reasoning + Acting 循环）

### 核心流程

```
用户问题 → AI思考并选择工具 → 执行工具 → 观察结果 → 继续思考 → ... → 最终回答
```

### 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | 健康检查 |
| POST | `/agent/ask` | Agent 问答（自主调用工具） |

### 启动

```bash
pip install -r requirements.txt
python -m uvicorn agent_api:app --reload --port 8002
```

访问接口文档：http://127.0.0.1:8002/docs

### 可调参数

- `query`：用户问题（必填）
- `max_steps`：最大推理轮数，默认 8

### Agent 内置工具

| 工具 | 功能 |
|------|------|
| `get_weather` | 查询城市天气（温度、天气、湿度） |
| `calculator` | 数学表达式计算 |
| `search_knowledge` | AI 知识库搜索（RAG、Agent、Embedding 等） |
| `get_time` | 获取当前日期、时间、星期 |
| `translate` | 文本翻译 |

---

## 项目结构

```
ai-agent-learning/
├── .env                # API Key 配置（不传 Git）
├── .gitignore          # Git 忽略规则
├── requirements.txt   # 依赖列表
├── config.py           # 配置管理（API Key、URL）
├── schemas.py          # 请求/响应模型 + 统一返回体
├── api_server.py       # 模块一：聊天 API 主入口
├── rag_api.py          # 模块二：RAG 问答 API 服务
├── agent_api.py        # 模块三：多工具 Agent API 服务
├── stream_test.html    # 流式聊天测试页面
└── routers/            # 聊天 API 路由模块
    ├── __init__.py
    ├── health.py       # 健康检查接口
    └── chat.py         # 聊天接口（普通 + 流式）
```

## 环境配置

在 `.env` 文件中配置：

```
ZHIPU_API_KEY=你的智谱API Key
```

## 学习进度

- ✅ 第 1 周：环境搭建 → API 调用 → FastAPI → SSE 流式 → 项目模板
- ✅ 第 2 周：Prompt Engineering → RAG 检索 → RAG API 服务
- ✅ 第 3 周：工具调用 → ReAct Agent → 多工具编排 → Agent API 服务
