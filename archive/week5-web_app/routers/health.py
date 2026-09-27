from fastapi import APIRouter
from schemas import APIResponse

router = APIRouter()


@router.get("/health")
def health():
    return APIResponse(data="AI Agent API 正在运行")
