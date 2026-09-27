from typing import Optional, Any

from pydantic import BaseModel


class ChatRequest(BaseModel):
    prompt: str
    model: str = "glm-4-flash"
    system: Optional[str] = None


class APIResponse(BaseModel):
    """统一返回体：所有接口都用这个。data 字段放业务数据。"""
    code: int = 0
    msg: str = "ok"
    data: Optional[Any] = None
