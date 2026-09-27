# 📚 AI Agent 学习路径 Archive

> 记录从第1周到第3周的学习历程，展示技术演进和项目成长

## 🗂️ 目录结构

```
archive/
├── week1-week3/                    # 前三周学习代码
│   ├── week1-basic-api/            # 第1周：基础API服务
│   ├── week2-prompt-engineering/   # 第2周：提示词工程
│   └── week3-react-agent/          # 第3周：ReAct多工具Agent
├── week4-paper_agent/              # 第4周早期版本（已被 course_agent/ 取代）
│   ├── state.py                    # State: messages + step_count
│   ├── nodes.py                    # 节点: 含 MAX_STEPS 步数限制
│   ├── graph.py                    # 图组装: dict 映射条件边
│   ├── server.py                   # API + static/chat.html 前端页面
│   └── static/chat.html            # 聊天 UI
└── learning-evolution.md          # 学习演进历程
```

## 📖 学习路线回顾

### 第1周：AI聊天API服务基础
- **目标**: 掌握FastAPI框架，实现基础对话功能
- **技术栈**: FastAPI + Uvicorn + 智谱API
- **成果**: 普通聊天、流式响应、健康检查接口

### 第2周：Prompt Engineering
- **目标**: 学习提示词设计，实现RAG系统
- **技术栈**: Embedding + 向量搜索 + 知识库
- **成果**: 文档向量化、语义搜索、RAG问答

### 第3周：ReAct多工具Agent
- **目标**: 实现多工具自主编排的Agent
- **技术栈**: ReAct范式 + Function Calling + 工具编排
- **成果**: 5种工具的自主选择和执行

### 第4周：LangGraph王牌项目（当前）
- **目标**: 使用LangGraph构建课程知识库Agent
- **技术栈**: LangGraph + ChromaDB + RAG
- **成果**: 结构化Agent框架、多工具集成
- **演进**: paper_agent/（早期版本，含 step_count、static 前端）→ course_agent/（重构版，简洁 ReAct 流程）

## 🔄 代码演进

从简单的API服务 → 单工具 → 多工具 → 结构化Agent，每个阶段都有：

1. **新技术的引入**
   - 第1周：Web框架和API设计
   - 第2周：向量和语义搜索
   - 第3周：ReAct和工具编排
   - 第4周：LangGraph框架

2. **架构的演进**
   - 线性处理 → 循环推理 → 图结构执行
   - 手动管理 → 框架管理
   - 单一功能 → 复杂系统集成

3. **代码质量的提升**
   - 错误处理完善
   - 模块化设计
   - 可维护性增强

## 💡 学习建议

1. **对比学习**: 对比archive和当前代码，理解技术选型的原因
2. **演进思维**: 理解每个阶段的设计决策和改进点
3. **复用思想**: 将前期学到的概念应用到新的框架中
4. **面试准备**: Archive展示了完整的学习历程，适合作为面试素材

## 🎯 保持内容

- ✅ 展示学习路径的渐进性
- � 体现技术深度和广度
- � 记录决策和思考过程
- � 便于回顾和对比

## 🗑️ 已移至Archive的文件

这些文件已经整理到archive中，当前项目不再使用但保留了学习价值：

- `labs/` - 所有实验文件
- `agent_api.py` - 第3周的多工具Agent（已重构为course_agent）
- `old_api_server.py` - 早期的API服务
- `paper_agent/` - 第4周早期 LangGraph Agent（已重构为 course_agent，含 step_count、static 前端页面）
- 其他实验性和迭代代码