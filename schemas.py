from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    model: str = "glm-4-flash"
    system: Optional[str] = None


class ChatResponse(BaseModel):
    code: int = 0
    msg: str = "ok"
    reply: Optional[str] = None

#msg=状态描述
def success(data=None, msg="ok"):
    return {"code": 0, "msg": msg, "data": data}

#code=状态码，0表示成功，其他数字是错误码
def error(msg="出错", code=1):
    return {"code": code, "msg": msg, "data": None}
