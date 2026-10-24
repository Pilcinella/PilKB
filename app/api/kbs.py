"""Knowledge bases API: 列出 / 显式创建知识库.

多库访问的前端入口. 在此之前前端把 kb_id 写死成 DEMO_KB 常量, 后端的多库 +
ACL 能力完全用不上 —— 选择器需要知道"当前用户能用哪些 kb" (GET), "新建知识库"
按钮需要一个不经过上传就能建库的端点 (POST).

权限语义刻意与既有链路单一来源:
  - 可见性 = list_kbs_for_user, 与 ensure_kb_for_read 同一套判据;
  - 建库 = ensure_kb_for_write, kb_id 校验 / 10 库上限 / 他人占用 403 全部复用,
    POST 与"首次上传自动建库"行为一致, 不引入第二套规则.
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends
from pydantic import BaseModel

from app.api.deps import get_current_user
from app.core.kb_registry import (
    can_write_kb,
    ensure_kb_for_rebuild,
    ensure_kb_for_write,
    get_kb,
    list_kbs_for_user,
    remove_kb,
)
from app.rag.retriever import delete_kb_storage

router = APIRouter(prefix="/api/v1/kbs", tags=["kbs"])


class KbInfo(BaseModel):
    kb_id: str
    owner: str
    is_public: bool
    can_write: bool


class KbCreateRequest(BaseModel):
    kb_id: str


@router.get("", response_model=list[KbInfo], summary="列出当前用户可访问的知识库")
async def list_kbs(user: str = Depends(get_current_user)) -> list[KbInfo]:
    # registry 首访后有进程内缓存, 这里不再套 to_thread (与 get_kb 的既有用法一致).
    return [
        KbInfo(
            kb_id=kb.kb_id,
            owner=kb.owner,
            is_public=kb.is_public,
            can_write=can_write_kb(kb.kb_id, user),
        )
        for kb in list_kbs_for_user(user)
    ]


@router.post("", summary="创建知识库 (校验/归属/上限与上传自动建库一致)")
async def create_kb(req: KbCreateRequest, user: str = Depends(get_current_user)):
    # 先查存在性, 才能如实区分 created / exists; ensure_kb_for_write 对已存在
    # 且属于自己的 kb 是幂等放行的, 单独看它的返回值分不出这两种情况.
    existed = get_kb(req.kb_id.strip()) is not None
    kb = ensure_kb_for_write(req.kb_id, user)
    return {
        "status": "exists" if existed else "created",
        "kb_id": kb.kb_id,
        "owner": kb.owner,
        "is_public": kb.is_public,
    }


@router.delete("/{kb_id}", summary="删除知识库 (向量库 + BM25 + 注册表; 仅 owner / 公共库管理员)")
async def delete_kb(kb_id: str, user: str = Depends(get_current_user)):
    # 权限判据与 BM25 重建同源 (ensure_kb_for_rebuild): 私有 kb = owner,
    # 公共 kb = 管理员 —— 重建与删除都是对 kb 索引的运维级操作, 谁能修谁才能删;
    # 删除不可恢复, 前端 (上传页) 有内联二次确认。
    ensure_kb_for_rebuild(kb_id, user)
    # 存储清理是重 IO (删 collection / unlink pkl), 卸线程池.
    await asyncio.to_thread(delete_kb_storage, kb_id)
    remove_kb(kb_id)
    return {"status": "deleted", "kb_id": kb_id}
