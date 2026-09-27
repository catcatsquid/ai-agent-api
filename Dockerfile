FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY course_agent/ course_agent/
COPY knowledge_pipeline/ knowledge_pipeline/
COPY rag_tool.py config.py build_knowledge.py ./

RUN mkdir -p data vector_store

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "course_agent.server:app", "--host", "0.0.0.0", "--port", "8003"]
