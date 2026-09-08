"""LLM 配置 - 智谱 GLM-4-Flash

对比第3周 agent_api.py 里用 requests.post 手动拼 JSON 调 API：
  payload = {"model": "glm-4-flash", "messages": [...], "tools": TOOLS}
  resp = requests.post(URL, headers=headers, json=payload)

LangChain 的 ChatZhipuAI 把这些封装好了：
  llm.invoke(messages) 直接返回 AIMessage 对象
  llm.bind_tools(tools) 自动处理 function calling
"""

import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI

load_dotenv()

llm = ChatZhipuAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY"),
    temperature=0.7,
)
