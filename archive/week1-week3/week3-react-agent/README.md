# 第3周：ReAct多工具Agent

## 学习目标
- 掌握ReAct（Reasoning + Acting）范式
- 实现多工具自主编排
- 学习Function Calling
- 理解Agent决策流程

## 核心技术
- ReAct范式
- Function Calling
- 工具编排
- 自主决策

## 主要功能
1. 天气查询工具
2. 数学计算工具
3. 知识搜索工具
4. 时间查询工具
5. 翻译工具
6. 多工具自主选择和执行

## 文件说明
- `src/` - 第3周实验代码
  - `agent_api.py` - 多工具Agent API服务
  - `react_lab.py` - ReAct模式实验
  - `tool_lab.py` - 工具定义和使用
  - `multi_tool_agent.py` - 多工具Agent实现
- `tools/` - 各类工具实现

## 核心概念
- **ReAct**: 推理和行动的循环模式
- **Function Calling**: LLM调用函数的能力
- **Tool Use**: 工具的选择和使用
- **Self-Correction**: 自我纠错和迭代

## 学习要点
- ReAct循环实现
- 工具定义和调用
- Agent决策逻辑
- 多工具编排策略