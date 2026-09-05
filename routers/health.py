from fastapi import APIRouter
from schemas import success

router = APIRouter()#创建一个小路由器
#路由 = "哪个网址交给哪个函数处理" 的映射规则。


@router.get("/")
def health():
    return success("AI Agent API is running")
