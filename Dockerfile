FROM docker.m.daocloud.io/library/python:3.12-slim

WORKDIR /app

RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 不装 gcc/g++：requirements.lock 里的依赖在 linux/amd64 + cp312 下都有预编译 wheel
# （实测：不装编译器也能 pip install 完，且构建更快、镜像更小）。
# 若将来某个依赖需要现场编译，用下面这段（已换成国内 Debian 源）：
# RUN sed -i 's|deb.debian.org|mirrors.tuna.tsinghua.edu.cn|g' \
#         /etc/apt/sources.list.d/debian.sources \
#  && apt-get update && apt-get install -y --no-install-recommends gcc g++ \
#  && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY course_agent/ course_agent/
COPY knowledge_pipeline/ knowledge_pipeline/
COPY rag_tool.py config.py build_knowledge.py doc_utils.py ./

RUN mkdir -p data vector_store

EXPOSE 8003

CMD ["python", "-m", "uvicorn", "course_agent.server:app", "--host", "0.0.0.0", "--port", "8003"]
