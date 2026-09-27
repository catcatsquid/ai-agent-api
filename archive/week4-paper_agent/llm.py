import os
from dotenv import load_dotenv
from langchain_community.chat_models import ChatZhipuAI

load_dotenv()

# 创建 LLM 实例（全局复用）
llm = ChatZhipuAI(
    model="glm-4-flash",
    api_key=os.getenv("ZHIPU_API_KEY"),
    temperature=0.7,
)
