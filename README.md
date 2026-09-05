# AI Agent API 服务

基于 FastAPI 搭建的 AI 聊天后端服务，支持 SSE 流式响应。

## 技术栈

- Python 3.12
- FastAPI（Web 框架）
- Uvicorn（ASGI 服务器）
- 智谱 GLM-4-Flash API（大模型）
- SSE（Server-Sent Events 流式响应）

## 项目结构

```
ai-agent-learning/
├── .env                # API Key 配置（不传 Git）
├── .gitignore          # Git 忽略规则
├── requirements.txt   # 依赖列表
├── config.py           # 配置管理（API Key、URL）
├── schemas.py          # 请求/响应模型 + 统一返回体
├── api_server.py       # 主入口，组装路由
├── stream_test.html    # 流式聊天测试页面
├── routers/            # 路由模块
│   ├── __init__.py
│   ├── health.py       # 健康检查接口
│   └── chat.py         # 聊天接口（普通 + 流式）
└── chat_v2.py          # 命令行聊天工具（练习用）
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | `/` | 健康检查 |
| POST | `/chat` | 普通聊天（一次性返回） |
| POST | `/chat/stream` | 流式聊天（打字机效果） |
| GET  | `/test` | 流式聊天测试页面 |

## 运行方法

1. 安装依赖：
```bash
pip install -r requirements.txt
```

2. 在 `.env` 文件中配置 API Key：
```
ZHIPU_API_KEY=你的API Key
```

3. 启动服务：
```bash
python -m uvicorn api_server:app --reload
```

4. 访问接口文档：http://127.0.0.1:8000/docs

5. 体验流式聊天：http://127.0.0.1:8000/test

## 使用示例

### 普通聊天
```bash
curl -X POST http://127.0.0.1:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "model": "glm-4-flash"}'
```

### 流式聊天
```bash
curl -X POST http://127.0.0.1:8000/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "你好", "model": "glm-4-flash"}'
```
