import os
import requests
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ZHIPU_API_KEY")
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

def chat(prompt, model="glm-4-flash", system=None, temperature=0.7, top_p=0.9):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
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

    resp = requests.post(URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


print("=" * 50)
print("实验1：temperature对比（同一问题问3次）")
print("=" * 50)

question = "用一个比喻解释什么是递归"

print("\n--- temperature=0.1（死板，每次回答几乎一样）---")
for i in range(3):
    result = chat(question, temperature=0.1)
    print(f"第{i+1}次: {result[:80]}...")

print("\n--- temperature=0.9（放飞，每次回答不同）---")
for i in range(3):
    result = chat(question, temperature=0.9)
    print(f"第{i+1}次: {result[:80]}...")


print("\n" + "=" * 50)
print("实验2：system prompt控制人设")
print("=" * 50)

print("\n--- 没有system prompt ---")
result = chat("我的代码报错了怎么办")
print(result[:150])

print("\n--- system: 你是一个毒舌但专业的程序员mentor ---")
result = chat("我的代码报错了怎么办", system="你是一个毒舌但专业的程序员mentor，回答简短，先吐槽再给建议")
print(result[:150])

print("\n--- system: 你是一个温柔耐心的老师 ---")
result = chat("我的代码报错了怎么办", system="你是一个温柔耐心的编程老师，语气鼓励，给出具体步骤")
print(result[:150])


print("\n" + "=" * 50)
print("实验3：Few-shot（给AI看几个例子再让它做）")
print("=" * 50)

few_shot_prompt = """请把下面的句子分类为"正面"、"负面"或"中性"。

例子：
输入：这家餐厅的菜太好吃了！
输出：正面

输入：服务态度差，再也不来了。
输出：负面

输入：今天天气不错。
输出：中性

现在请分类：
输入：这个产品的性价比很高，推荐购买。
输出："""

print("\n--- Few-shot分类结果 ---")
result = chat(few_shot_prompt, temperature=0.1)
print(result)


print("\n" + "=" * 50)
print("实验4：Chain-of-Thought（让AI一步步推理）")
print("=" * 50)

print("\n--- 不加CoT ---")
result = chat("小明有5个苹果，给了小红2个，又买了3个，现在有几个？")
print(result)

print("\n--- 加CoT（要求一步步想）---")
result = chat("小明有5个苹果，给了小红2个，又买了3个，现在有几个？请一步一步推理，最后给出答案。", temperature=0.1)
print(result)
