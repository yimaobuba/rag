from fastapi import FastAPI
from pydantic import BaseModel
import rag_core

# 启动时初始化（加载模型和数据）
rag_core.init()

app = FastAPI(
    title="高性价比人生指南 RAG API",
    description="基于《高性价比人生指南》的检索增强问答接口",
    version="1.0.0"
)

# ===== 请求体模型 =====
class AskRequest(BaseModel):
    question: str
    top_k: int = 3

    class Config:
        json_schema_extra = {
            "example": {
                "question": "人应该追求什么？",
                "top_k": 3
            }
        }

# ===== 响应体模型 =====
class Source(BaseModel):
    text: str
    score: float

class AskResponse(BaseModel):
    answer: str
    sources: list[Source]

# ===== 接口 =====
@app.get("/", summary="健康检查")
def root():
    return {"message": "RAG API is running"}

@app.post("/ask", response_model=AskResponse, summary="提问接口")
def ask_endpoint(req: AskRequest):
    """
    输入问题，返回基于指南的回答和参考来源。

    - **question**: 用户问题
    - **top_k**: 检索最相关的文本块数量，默认 3
    """
    result = rag_core.ask(req.question, req.top_k)
    return result