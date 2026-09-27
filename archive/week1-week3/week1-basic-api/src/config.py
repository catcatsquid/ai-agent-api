import os
from dotenv import load_dotenv

load_dotenv()

# 智谱
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_CHAT_URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
ZHIPU_EMBEDDING_URL = "https://open.bigmodel.cn/api/paas/v4/embeddings"

# DeepSeek
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
DEEPSEEK_CHAT_URL = "https://api.deepseek.com/chat/completions"
