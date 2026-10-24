"""Chat / Q&A schemas.

问答接口的请求体、来源条目与响应体.
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """一次问答请求."""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    kb_id: str = Field(default="default", description="检索目标知识库 ID; 公共/私有受 ACL 约束")
    session_id: Optional[str] = Field(default=None, description="会话 ID; 不传则由后端生成 (sess_<8位hex>)")
    mode: Literal["rag", "agent"] = Field(
        default="rag",
        description=(
            "rag=常规检索问答 (跳过路由判断, 直接检索); "
            "agent=先由路由判断要不要检索 / 用不用工具, 再分流到检索、工具或直答"
        ),
    )
    top_k: int = Field(default=5, ge=1, le=20, description="最终喂给 LLM 的上下文条数")
    stream: bool = Field(
        default=False,
        description="是否走 SSE 流式返回; 当前前端通过 /stream 接口显式触发流式",
    )


class SourceItem(BaseModel):
    """答案引用的一个检索来源 chunk."""

    doc_id: str = Field(..., description="来源文档 ID")
    chunk_id: str = Field(..., description="来源 chunk ID (doc_id_c<N>)")
    content: str = Field(..., description="该 chunk 截取的前 300 字符预览")
    score: float = Field(..., description="相关性分数: 优先 rerank_score, 无则回退 rrf_score")
    metadata: dict = Field(default={}, description="chunk 元数据 (keywords/summary/hyde_question 等)")


class ChatResponse(BaseModel):
    """一次性 (非流式) 问答的响应."""

    answer: str = Field(..., description="LLM 生成的答案文本")
    session_id: str = Field(..., description="本次会话 ID")
    sources: list[SourceItem] = Field(default=[], description="引用的检索来源列表")
    agent_trace: Optional[list[dict]] = Field(
        default=None,
        description="agent 模式下各节点 (router/retriever/tool:*/direct/answer) 的执行 trace",
    )
    usage: dict = Field(default={}, description="用量统计占位 (当前未填入 token 数)")


class ChatChunk(BaseModel):
    """SSE 流式返回的单个事件块.

    type 取值:
      - token  : 流式答案文本片段
      - source : 一条检索来源 (先于 token 推给前端做引用展示)
      - trace  : agent 模式的一条推理轨迹 (rag 模式不发)
      - notice : 给用户看的提示, 如"本次未使用知识库" (不进入答案正文)
      - done   : 正常结束 (无 data 字段)
      - error  : 异常 (data 含 msg; 出错时**不发** done)
    """

    type: Literal["token", "source", "trace", "notice", "done", "error"] = Field(
        default="token", description="事件类型"
    )
    data: dict = Field(default={}, description="与 type 对应的载荷")
