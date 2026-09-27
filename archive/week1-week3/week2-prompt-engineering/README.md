# 第2周：Prompt Engineering

## 学习目标
- 掌握Prompt Engineering基础
- 学习提示词工程技巧
- 实现RAG（检索增强生成）
- 理解向量化和语义搜索

## 核心技术
- FastAPI
- 智谱 Embedding-3（向量化）
- 余弦相似度计算
- 知识库构建

## 主要功能
1. 文档向量化
2. 语义相似度搜索
3. RAG问答系统
4. 知识库管理

## 文件说明
- `src/` - 第2周实验代码
  - `prompt_lab.py` - Prompt Engineering实验
  - `deepseek_demo.py` - DeepSeek模型对比
  - `vector_store/` - 向量存储实验
- `data/` - 示例知识库数据

## 核心概念
- **Embedding**: 将文本转换为向量表示
- **Cosine Similarity**: 计算文本相似度
- **RAG**: 检索增强生成，结合外部知识
- **Top-K检索**: 选择最相关的K个文档

## 学习要点
- 提示词设计原则
- 向量化技术
- 语义搜索实现
- 知识库构建流程