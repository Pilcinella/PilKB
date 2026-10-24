# -*- coding: utf-8 -*-
"""聊天端点: 只做 HTTP 适配, RAG 编排全部在 `app.rag.pipeline.stream_rag`.

这个模块原先自己写了一份完整编排 (检索 -> 拼提示词 -> 调 LLM -> 组装输出),
与 `pipeline.run_rag` 那份并存, 且两边的输出口径已经分叉。现在两个端点都消费
`stream_rag()` 的事件流, 这里只负责:

  - 鉴权 / 配额 / ACL (必须在 `StreamingResponse` 建立**之前**, 才能返回真 429/403)
  - 把事件映射成 SSE 帧 (线格式与前端约定, 不改)
  - 把事件聚合成 `ChatResponse`
"""
from __future__ import annotations
import json
import uuid

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from loguru import logger

from app.api.deps import get_current_user
from app.core.kb_registry import ensure_kb_for_read
from app.core.net import client_ip
from app.core.rate_limit import check_chat_quota
from app.rag.pipeline import (
    DoneEvent,
    NoticeEvent,
    SourcesEvent,
    TokenEvent,
    TraceEvent,
    build_rag_prompt,
    stream_rag,
)
from app.schemas.chat import ChatRequest, ChatResponse, SourceItem


def _check_chat_quota_or_429(user: str, request: Request):
    ip = client_ip(request)
    decision = check_chat_quota(user, ip)
    if not decision.allowed:
        from fastapi import HTTPException, status

        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=decision.reason,
            headers={"Retry-After": str(int(decision.retry_after) + 1)},
        )


router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# 引用来源正文的预览长度 (完整 chunk 走 RAG 提示词, 这里只是给前端展示的摘要).
_SOURCE_PREVIEW_CHARS = 300


def _to_source_item(c: dict) -> SourceItem:
    """canonical 候选 (pipeline.normalize_source 的形状) -> API 的 SourceItem.

    这是**唯一**的源映射。合并前 chat.py 里有 4 处手写同样的字段拼装, 落一处就会
    丢掉 doc_id / metadata —— 前端拿不到来源文档, 引用溯源就断了 (注释里记过这个坑).
    """
    meta = c.get("metadata") or {}
    return SourceItem(
        doc_id=meta.get("doc_id", ""),
        chunk_id=c.get("id", ""),
        content=(c.get("document") or "")[:_SOURCE_PREVIEW_CHARS],
        score=float(c.get("score", 0.0) or 0.0),
        metadata=meta,
    )


def _build_prompt(question, contexts):
    """(仅测试使用, 生产链路请直接用 pipeline.build_rag_prompt)

    保留这个薄包装是为了不破坏 tests/test_context_snippet_limit.py —— 它直接
    import 这个函数, 并对提示词的分节标记与截断长度做精确断言。提示词模板本身
    已统一到 pipeline.build_rag_prompt, 两条链路原先各写一份, 措辞已经悄悄分叉
    ("在回答末尾标注引用来源" vs "在引用处使用 [1][2] 等标注"), 改一处另一处不生效.
    """
    return build_rag_prompt(question, [c.get("document", "") or "" for c in contexts])


def _sse(payload: dict) -> str:
    """SSE 帧. 格式是前端解析器的约定 (frontend/lib/api.ts 的 streamChat):
    以 `data:` 开头、`\\n\\n` 结尾的单行 JSON。"""
    return "data: " + json.dumps(payload, ensure_ascii=False) + "\n\n"


def _source_frame(c: dict) -> str:
    return _sse({"type": "source", "data": _to_source_item(c).model_dump()})


@router.post("", response_model=ChatResponse, summary="一次性问答")
async def chat(req: ChatRequest, request: Request, user: str = Depends(get_current_user)):
    _check_chat_quota_or_429(user, request)
    # ACL: 公共 kb 直接放行; 私有 kb 必须 owner == 当前用户 (403)
    ensure_kb_for_read(req.kb_id, user)

    is_agent = req.mode == "agent"
    answer = ""
    sources: list[dict] = []
    agent_trace: list[dict] = []
    async for ev in stream_rag(
        req.question,
        kb_id=req.kb_id,
        top_k=req.top_k,
        use_agent=is_agent,
        stream_tokens=False,
        user_id=user,
        session_id=req.session_id,
    ):
        if isinstance(ev, SourcesEvent):
            sources = ev.sources
        elif isinstance(ev, DoneEvent):
            answer, agent_trace = ev.answer, ev.agent_trace

    sid = req.session_id or ("sess_" + uuid.uuid4().hex[:8])
    return ChatResponse(
        answer=answer,
        session_id=sid,
        sources=[_to_source_item(c) for c in sources],
        agent_trace=agent_trace if is_agent else None,
        usage={},
    )


@router.post("/stream", summary="SSE 流式问答")
async def chat_stream(req: ChatRequest, request: Request, user: str = Depends(get_current_user)):
    _check_chat_quota_or_429(user, request)
    # ACL: 进入流式生成前先校验, 避免生成到一半才抛 403
    ensure_kb_for_read(req.kb_id, user)

    async def event_gen():
        try:
            async for ev in stream_rag(
                req.question,
                kb_id=req.kb_id,
                top_k=req.top_k,
                use_agent=(req.mode == "agent"),
                stream_tokens=True,
                user_id=user,
                session_id=req.session_id,
            ):
                if isinstance(ev, TraceEvent):
                    yield _sse({"type": "trace", "data": ev.entry})
                elif isinstance(ev, SourcesEvent):
                    for c in ev.sources:
                        yield _source_frame(c)
                elif isinstance(ev, NoticeEvent):
                    # 提示必须走独立帧: trace 帧前端直接丢弃, 而 answer 正文会被
                    # 送进 RAGAS 打分与历史落库, 两边都不能顺便捎带它.
                    yield _sse({"type": "notice", "data": {"text": ev.text}})
                elif isinstance(ev, TokenEvent):
                    yield _sse({"type": "token", "data": {"text": ev.text}})
                elif isinstance(ev, DoneEvent):
                    yield 'data: {"type":"done"}\n\n'
        except Exception as e:
            # 一旦开始 yield, HTTP 状态码已经固定成 200, 失败只能以 error 事件下发.
            # 注意这里**不发** done 帧: 前端据此判断"正常结束", 出错后再补一个
            # done 会让它把半截答案当成完整结果.
            logger.exception("stream error: " + str(e))
            # 下发通用文案: str(e) 可能带 provider 报文 / 内部路径等实现细节,
            # 细节留在服务端日志里即可.
            yield _sse({"type": "error", "data": {"msg": "服务器内部错误，请稍后重试"}})

    return StreamingResponse(event_gen(), media_type="text/event-stream")
