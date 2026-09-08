import json
import time
import ast
import operator
import requests
from fastapi import FastAPI
from pydantic import BaseModel
from config import ZHIPU_API_KEY, ZHIPU_CHAT_URL, DEEPSEEK_API_KEY, DEEPSEEK_CHAT_URL, DEEPSEEK_API_KEY, DEEPSEEK_CHAT_URL
from schemas import APIResponse

app = FastAPI(title="Agent API", version="1.0")


# ============================================================
# 安全计算器（替代 eval，只允许四则运算和幂）
# ============================================================

SAFE_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
}


def _safe_eval(expr: str) -> float:
    """用 AST 解析数学表达式，只允许数字和四则运算，禁止任意代码执行。"""
    node = ast.parse(expr, mode="eval").body

    def _eval(n):
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.UnaryOp):
            return SAFE_OPS[type(n.op)](_eval(n.operand))
        if isinstance(n, ast.BinOp):
            return SAFE_OPS[type(n.op)](_eval(n.left), _eval(n.right))
        raise ValueError(f"不支持的操作: {type(n).__name__}")

    return _eval(node)


# ============================================================
# 工具定义
# ============================================================

def get_weather(city):
    weather_data = {
        "北京": {"temp": 25, "weather": "晴", "humidity": 40},
        "上海": {"temp": 28, "weather": "多云", "humidity": 65},
        "广州": {"temp": 32, "weather": "雷阵雨", "humidity": 80},
        "深圳": {"temp": 30, "weather": "阵雨", "humidity": 75},
    }
    city = city.replace("市", "").replace("的", "")
    if city in weather_data:
        return weather_data[city]
    return {"error": f"暂不支持{city}的天气查询"}


def calculator(expression):
    try:
        result = _safe_eval(expression)
        return {"result": result}
    except Exception as e:
        return {"error": f"计算失败: {e}"}


def search_knowledge(query):
    knowledge = {
        "rag": "RAG是检索增强生成，先从文档检索相关内容，再让大模型基于检索结果回答",
        "agent": "Agent是能自主思考和行动的AI，通过ReAct模式循环推理和调用工具",
        "embedding": "Embedding是把文本转成向量，用于语义相似度计算",
        "react": "ReAct是Reasoning+Acting的缩写，让AI交替进行推理和行动",
        "tool": "Tool Calling是让大模型根据用户意图选择并调用外部工具的能力",
    }
    for key in knowledge:
        if key in query.lower():
            return {"answer": knowledge[key]}
    return {"answer": "未找到相关知识"}


def get_time():
    from datetime import datetime
    now = datetime.now()
    return {
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": ["一", "二", "三", "四", "五", "六", "日"][now.weekday()],
    }


def translate(text, target_lang):
    translations = {
        ("你好", "english"): "Hello",
        ("谢谢", "english"): "Thank you",
        ("人工智能", "english"): "Artificial Intelligence",
        ("hello", "chinese"): "你好",
        ("thank you", "chinese"): "谢谢",
    }
    key = (text.lower(), target_lang.lower())
    if key in translations:
        return {"translation": translations[key]}
    return {"translation": f"[模拟翻译] {text} -> {target_lang}"}


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的天气信息，包括温度、天气状况和湿度",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名称，如：北京、上海"}
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "数学计算器，可以计算数学表达式",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如：3+5*2"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索AI相关知识库，可以回答RAG、Agent、Embedding等概念问题",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "获取当前日期、时间和星期几",
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate",
            "description": "翻译文本到目标语言",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "要翻译的文本"},
                    "target_lang": {"type": "string", "description": "目标语言，如：english、chinese"},
                },
                "required": ["text", "target_lang"],
            },
        },
    },
]

TOOL_MAP = {
    "get_weather": get_weather,
    "calculator": calculator,
    "search_knowledge": search_knowledge,
    "get_time": get_time,
    "translate": translate,
}


# ============================================================
# Agent 核心逻辑
# ============================================================

def run_agent(user_question, max_steps=8, timeout=60):
    """
    Agent 执行器
    返回: (answer, action_log, elapsed)
    """
    headers = {
        "Authorization": f"Bearer {ZHIPU_API_KEY}",
        "Content-Type": "application/json",
    }

    messages = [{"role": "user", "content": user_question}]
    action_log = []
    start_time = time.time()

    for step in range(1, max_steps + 1):
        if time.time() - start_time > timeout:
            return "Agent执行超时", action_log, round(time.time() - start_time, 2)

        payload = {
            "model": "glm-4-flash",
            "messages": messages,
            "tools": TOOLS,
            "tool_choice": "auto",
        }

        try:
            resp = requests.post(ZHIPU_CHAT_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.Timeout:
            return "API请求超时", action_log, round(time.time() - start_time, 2)
        except requests.exceptions.RequestException as e:
            return f"API请求失败: {e}", action_log, round(time.time() - start_time, 2)

        data = resp.json()
        message = data["choices"][0]["message"]

        if message.get("tool_calls"):
            for tool_call in message["tool_calls"]:
                func_name = tool_call["function"]["name"]
                func_args = json.loads(tool_call["function"]["arguments"])

                try:
                    result = TOOL_MAP[func_name](**func_args)
                except Exception as e:
                    result = {"error": f"工具执行异常: {e}"}

                action_log.append({
                    "step": step,
                    "tool": func_name,
                    "args": func_args,
                    "result": result,
                })

                messages.append(message)
                messages.append({
                    "role": "tool",
                    "content": json.dumps(result, ensure_ascii=False),
                    "tool_call_id": tool_call["id"],
                })
        else:
            elapsed = round(time.time() - start_time, 2)
            return message["content"], action_log, elapsed

    return "Agent无法在限定步数内完成回答", action_log, round(time.time() - start_time, 2)


# ============================================================
# 接口
# ============================================================

class AgentRequest(BaseModel):
    query: str
    max_steps: int = 8


@app.get("/")
def health():
    return APIResponse(data="Agent API 正在运行")


@app.post("/agent/ask")
def agent_ask(req: AgentRequest):
    """Agent问答接口：用户提问 → Agent自主调用工具 → 返回答案+调用记录"""
    try:
        answer, action_log, elapsed = run_agent(req.query, max_steps=req.max_steps)

        tools_used = [
            {"step": log["step"], "tool": log["tool"], "args": log["args"]}
            for log in action_log
        ]

        return APIResponse(data={
            "answer": answer,
            "tools_used": tools_used,
            "elapsed": elapsed,
        })
    except Exception as e:
        return APIResponse(code=1, msg=f"Agent执行出错: {e}")
