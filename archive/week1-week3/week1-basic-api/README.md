# 第1周：AI聊天API服务基础

## 学习目标
- 掌握 FastAPI 框架基础
- 实现基本的AI对话API
- 学习 SSE 流式响应
- 理解API服务架构

## 核心技术
- Python 3.12
- FastAPI（Web框架）
- Uvicorn（ASGI服务器）
- 智谱 GLM-4-Flash API
- SSE（Server-Sent Events）

## 主要功能
1. 普通聊天接口（一次性返回）
2. 流式聊天接口（打字机效果）
3. 健康检查接口

## 文件说明
- `src/api_server.py` - 主API服务
- `src/chat.py` - 聊天功能实现
- `src/chat_v2.py` - 改进版聊天（包含错误处理）
- `config.py` - 配置文件
- `requirements.txt` - 依赖列表

## 启动方式
```bash
python src/api_server.py
```

访问：http://127.0.0.1:8000/docs

## 学习要点
- FastAPI路由定义
- 请求/响应模型
- 流式响应实现
- 错误处理机制