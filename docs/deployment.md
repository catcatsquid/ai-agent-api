# Docker 部署指南

## 前置要求

- Docker Desktop 4.x+（已验证 Docker 29.7.2）
- 项目根目录下 `.env` 文件已配置智谱 API Key
- `vector_store/` 和 `data/` 目录存在（首次部署会自动创建）

## 快速部署

```bash
# 构建并启动
docker compose up -d --build

# 查看日志
docker logs -f course-agent

# 停止
docker compose down
```

启动后访问：
- 聊天页面: http://localhost:8003/chat
- 知识库管理: http://localhost:8003/kb
- API 文档: http://localhost:8003/docs

## 架构说明

```
┌─────────────────────────────────────────┐
│  Docker Container (course-agent)        │
│                                         │
│  uvicorn :8003                          │
│  ├── /chat        → 聊天页面            │
│  ├── /kb         → 知识库管理页面      │
│  ├── /chat/stream → SSE 流式问答        │
│  ├── /kb/*       → 知识库 API          │
│  ├── /qb/*       → 题库 API            │
│  └── /auth/*     → 认证 API            │
│                                         │
│  Volumes:                               │
│   ├── ./vector_store → /app/vector_store│
│   └── ./data        → /app/data        │
└─────────────────────────────────────────┘
```

## Dockerfile 说明

| 步骤 | 说明 |
|------|------|
| 基础镜像 | `python:3.12-slim`（国内 daocloud 镜像） |
| 依赖安装 | 先复制 `requirements.txt` 再 `pip install`，利用分层缓存 |
| 代码复制 | 仅复制 `course_agent/`, `knowledge_pipeline/`, `rag_tool.py`, `config.py`, `build_knowledge.py` |
| 暴露端口 | 8003 |
| 启动命令 | `uvicorn course_agent.server:app --host 0.0.0.0 --port 8003` |

## docker-compose.yml 说明

| 配置 | 说明 |
|------|------|
| `ports: 8003:8003` | 宿主机与容器同端口 |
| `volumes: vector_store` | 向量库持久化，容器删除不丢数据 |
| `volumes: data` | 知识库文档和题库持久化 |
| `env_file: .env` | 注入智谱 API Key 等环境变量 |
| `restart: unless-stopped` | Docker 重启后自动恢复 |
| `healthcheck` | 每 30s 检查 `/` 端点 |

## 常见操作

### 重新构建（代码更新后）

```bash
docker compose up -d --build
```

### 进入容器调试

```bash
docker exec -it course-agent bash
```

### 重建知识库（容器内）

```bash
docker exec -it course-agent python build_knowledge.py --clear
```

### 查看资源占用

```bash
docker stats course-agent
```

## 注意事项

1. `.dockerignore` 排除了 `.venv/`, `__pycache__/`, `.env`, `.git/` 等，镜像不含密钥
2. `data/` 和 `vector_store/` 通过 volume 挂载，不进镜像
3. 国内环境使用清华 pip 镜像加速依赖安装
4. 容器内 uvicorn 监听 `0.0.0.0` 而非 `127.0.0.1`，确保外部可访问
5. Docker Desktop 需从开始菜单手动启动，不可通过终端后台运行
