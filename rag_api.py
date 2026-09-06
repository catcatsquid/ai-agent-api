import os
import math
import requests
from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel

load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY")
CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"

app = FastAPI(title="RAG问答API", version="1.0")


# ============================================================
# 工具函数
# ============================================================

def get_embedding(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": "embedding-3", "input": text}
    resp = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["data"][0]["embedding"]

def cosine_similarity(v1, v2):
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    return dot / (norm1 * norm2)

def chat(prompt, system=None, temperature=0.1):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    payload = {"model": "glm-4-flash", "messages": messages, "temperature": temperature}
    resp = requests.post(CHAT_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


# ============================================================
# 知识库 + 向量数据库（启动时初始化）
# ============================================================

KNOWLEDGE_BASE = """
智谱AI成立于2019年，是由清华大学计算机系技术成果转化而来的公司。智谱AI致力于让机器像人一样思考，是国内领先的通用大模型研发企业。

智谱GLM-4是智谱AI推出的第四代大语言模型，支持中英双语，具有强大的理解和生成能力。GLM-4支持128K上下文，可以处理超长文本。

GLM-4-Flash是智谱AI推出的轻量级模型，主打快速响应和低成本。它适合对话、问答、文本生成等基础任务，适合学习和开发测试使用。

智谱API的调用方式是发送HTTP请求到open.bigmodel.cn，需要使用API Key进行认证。支持同步调用和流式调用两种方式。

Embedding模型embedding-3是智谱提供的文本向量化模型，可以把文本转成向量，用于语义搜索和相似度计算。

智谱AI的API遵循OpenAI兼容格式，可以使用OpenAI的SDK直接调用，只需要修改base_url和api_key即可。
"""

# 切块
def chunk_text(text, chunk_size=200, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

# 启动时建库
chunks = chunk_text(KNOWLEDGE_BASE, chunk_size=200, overlap=50)
vector_db = []
for chunk in chunks:
    emb = get_embedding(chunk)
    vector_db.append({"text": chunk, "embedding": emb})

print(f"[启动] 知识库初始化完成，共 {len(vector_db)} 条文档向量")


# ============================================================
# 请求/响应模型
# ============================================================

#请求格式
class RAGRequest(BaseModel):
    query: str
    top_k: int = 3
    temperature: float = 0.1
#响应格式
class RAGResponse(BaseModel):
    code: int = 0
    msg: str = "ok"
    answer: str = None
    sources: list = []


# ============================================================
# 接口
# ============================================================

@app.get("/")
def health():
    return {"code": 0, "msg": "ok", "data": "RAG问答API正在运行"}


@app.post("/rag/ask", response_model=RAGResponse)
def rag_ask(req: RAGRequest):
    """RAG问答接口：用户提问 → 检索知识库 → 大模型基于检索内容回答"""
    try:
        # 1. 把问题向量化
        query_emb = get_embedding(req.query)

        # 2. 检索最相关的top_k条
        results = []
        for item in vector_db:
            sim = cosine_similarity(query_emb, item["embedding"])
            results.append({"text": item["text"], "similarity": sim})
        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_results = results[:req.top_k]

        # 3. 拼上下文
        context = "\n\n".join([r["text"] for r in top_results])

        # 4. 构造prompt
        rag_prompt = f"""请根据以下参考资料回答用户的问题。如果参考资料中没有答案，请说"根据已有资料无法回答"。

参考资料：
{context}

用户问题：{req.query}

请基于参考资料回答："""

        system_prompt = "你是一个基于检索内容的问答助手。只能根据提供的参考资料回答，不要编造信息。"

        # 5. 调大模型
        answer = chat(rag_prompt, system=system_prompt, temperature=req.temperature)

        # 6. 返回结果
        sources = [
            {"text": r["text"][:80] + "..." if len(r["text"]) > 80 else r["text"],
             "similarity": round(r["similarity"], 4)}
            for r in top_results
        ]

        return RAGResponse(code=0, msg="ok", answer=answer, sources=sources)

    except Exception as e:
        return RAGResponse(code=1, msg=f"出错了: {str(e)}", answer=None, sources=[])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
