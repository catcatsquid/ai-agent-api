import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ZHIPU_API_KEY")
URL = "https://open.bigmodel.cn/api/paas/v4/chat/completions"

def call_model(prompt, model="glm-4-flash"):
    """调用模型"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = requests.post(URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# 任务1：输出格式控制（JSON模式）
def test_json_output():
    """测试JSON输出格式"""
    prompt = """请从以下文本中提取关键信息，并以JSON格式返回：
    张三，25岁，北京人，软件工程师，喜欢编程。
    李四，30岁，上海人，数据分析师，喜欢数据挖掘。"""

    result = call_model(prompt)
    print("JSON输出测试结果：")
    print(result)

# 任务2：从混乱文本提取结构化JSON
def test_structured_extraction():
    """测试从混乱文本提取结构化JSON"""
    prompt = """请从以下混乱文本中提取结构化信息：
    名字：张三，年龄：25，城市：北京，职业：软件工程师。
    名字：李四，年龄：30，城市：上海，职业：数据分析师。"""

    result = call_model(prompt)
    print("结构化提取测试结果：")
    print(result)

# 任务3：角色扮演 + 拒绝超范围问题
def test_role_playing():
    """测试角色扮演和拒绝超范围问题"""
    prompt1 = """你是一个严谨的数学老师，请回答：2+2等于多少？"""
    prompt2 = """你是一个严谨的数学老师，请回答：今天天气怎么样？"""

    result1 = call_model(prompt1)
    result2 = call_model(prompt2)

    print("角色扮演测试结果1：", result1)
    print("角色扮演测试结果2：", result2)

# 任务4：Zero-shot vs Few-shot vs CoT 对比
def test_zero_few_cot_comparison():
    """测试三种方法的对比"""
    question = "请计算：1+2+3+4+5+6+7+8+9+10"

    # Zero-shot
    zero_shot = call_model(f"请计算：{question}")

    # Few-shot
    few_shot = call_model(f"""请计算：{question}
    示例：1+2=3，2+3=5，3+4=7
    请按照示例的方式计算：{question}""")

    # CoT
    cot = call_model(f"""请计算：{question}
    请逐步思考：
    1. 1+2=3
    2. 3+3=6
    3. 6+4=10
    4. 10+5=15
    5. 15+6=21
    6. 21+7=28
    7. 28+8=36
    8. 36+9=45
    9. 45+10=55
    最终结果：55""")

    print("Zero-shot 结果：", zero_shot)
    print("Few-shot 结果：", few_shot)
    print("CoT 结果：", cot)

# 任务5：Self-Refine（自我修正）
def test_self_refine():
    """测试自我修正"""
    prompt = """请回答：什么是机器学习？
    然后评估你的回答是否全面，如果不全面请补充。"""

    result = call_model(prompt)
    print("自我修正测试结果：")
    print(result)

# 任务6：输出工具调用指令
def test_tool_calling():
    """测试输出工具调用指令"""
    prompt = """请判断以下问题需要调用什么工具，并输出工具调用指令：
    1. 计算表达式：3+5*2
    2. 查询北京天气
    3. 翻译'hello'到中文"""

    result = call_model(prompt)
    print("工具调用指令测试结果：")
    print(result)

if __name__ == "__main__":
    test_json_output()
    test_structured_extraction()
    test_role_playing()
    test_zero_few_cot_comparison()
    test_self_refine()
    test_tool_calling()