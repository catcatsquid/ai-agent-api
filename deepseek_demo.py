import os
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
URL = "https://api.deepseek.com/chat/completions"


def chat(prompt, model="deepseek-chat", system=None, temperature=0.7, top_p=0.9):
    """调用 DeepSeek。和 GLM 的区别只有三处：URL、Key、模型名，其余一模一样。"""
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json",
    }
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
    }

    try:
        resp = requests.post(URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError:
        print(f"API错误: {resp.status_code} - {resp.text}")
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"网络错误: {e}")
        sys.exit(1)
    except (KeyError, IndexError):
        print(f"响应格式异常，原始数据: {resp.text}")
        sys.exit(1)


if __name__ == "__main__":
    if not DEEPSEEK_API_KEY:
        print("错误: 请在 .env 文件中设置 DEEPSEEK_API_KEY")
        sys.exit(1)

    print("=" * 50)
    print("DeepSeek 对话测试")
    print("=" * 50)

    # 1. 基础问答
    print("\n--- 基础问答 ---")
    print(chat("用一句话介绍你自己"))

    # 2. temperature 对比（注意：DeepSeek 的 temperature 范围是 0~2，比 GLM 的 0~1 更宽）
    print("\n--- temperature=0.1（稳定，几乎每次一样）---")
    print(chat("用一个比喻解释什么是递归", temperature=0.1)[:80])

    print("\n--- temperature=1.5（放飞，GLM 到不了这个值）---")
    print(chat("用一个比喻解释什么是递归", temperature=1.5)[:80])

    # 3. 推理模型 deepseek-reasoner（R1）：会额外返回 reasoning_content 思考过程
    #    这是 DeepSeek 和 GLM 的一个真实差异点，面试能讲
    print("\n--- deepseek-reasoner 推理模型 ---")
    answer = chat("小明有5个苹果，给了小红2个，又买了3个，现在有几个？", model="deepseek-reasoner")
    print(answer)
