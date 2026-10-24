"""Chat history API: 列出 / 获取 / 删除历史会话。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from loguru import logger
from pydantic import BaseModel, Field

from app.api.deps import get_current_user
from app.core.config import settings


router = APIRouter(prefix="/api/v1/history", tags=["history"])


# 会话历史目录. 之前这里硬编码 parents[2]/data/history, 把 HISTORY_STORAGE_DIR
# 配置架空成了死项 (conftest 重定向该变量也不生效). 改为读配置: 默认值与原硬编码
# 路径一致, 且 config 侧已按项目根目录锚定, 不随 CWD 漂移.
_HISTORY_DIR = Path(settings.history_storage_dir)

# _load -> 修改 -> _save 是 read-modify-write, 并发请求会互相覆盖丢消息.
# 每个用户一把锁, 串行化同一用户的读写 (不同用户之间不互斥).
_USER_LOCKS: dict[str, threading.Lock] = {}
_USER_LOCKS_GUARD = threading.Lock()


def _user_lock(user: str) -> threading.Lock:
    with _USER_LOCKS_GUARD:
        lock = _USER_LOCKS.get(user)
        if lock is None:
            lock = threading.Lock()
            _USER_LOCKS[user] = lock
        return lock


class HistoryItem(BaseModel):
    role: str  # user / assistant
    content: str
    sources: list[dict] = Field(default_factory=list)
    ts: float = Field(default_factory=lambda: time.time())


class SessionMeta(BaseModel):
    session_id: str
    title: str
    kb_id: str = "default"
    mode: str = "rag"
    created_at: float
    updated_at: float
    message_count: int


class SessionDetail(SessionMeta):
    messages: list[HistoryItem]


# 直接作文件名的用户名: 全小写 ASCII (Windows 文件系统大小写不敏感, "Alice" 与
# "alice" 会撞同一个文件) 且避开 Windows 保留设备名.
_SAFE_FILENAME_RE = re.compile(r"^[a-z0-9_-]{1,64}$")
_WIN_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def _user_path(user: str) -> Path:
    _HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    if _SAFE_FILENAME_RE.match(user) and user not in _WIN_RESERVED:
        # 安全用户名维持原名, 老数据不用迁移.
        return _HISTORY_DIR / f"{user}.json"
    # 其余用户名走确定性哈希. 旧逻辑按字符过滤会撞文件: "a.b" 与 "ab" 同名,
    # 中文用户名全部过滤成空串共享同一个 .json —— 历史记录跨用户可见、互相覆盖.
    digest = hashlib.sha256(user.encode("utf-8")).hexdigest()[:24]
    return _HISTORY_DIR / f"u_{digest}.json"


def _load(user: str) -> dict:
    p = _user_path(user)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save(user: str, data: dict) -> None:
    # tmp + os.replace 原子落盘: 直接 write_text 在写一半崩溃/被撕裂读后, _load
    # 的兜底 except 会返回 {}, 下一次 append 把整个历史文件覆盖丢失. 原子替换保证
    # 任何时候读到的都是完整的旧文件或完整的新文件 (同 kb_registry._save_unlocked).
    p = _user_path(user)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, p)


@router.get("", response_model=list[SessionMeta], summary="列出当前用户全部会话")
async def list_sessions(user: str = Depends(get_current_user)):
    data = _load(user)
    out: list[SessionMeta] = []
    for sid, sess in data.items():
        try:
            out.append(SessionMeta(
                session_id=sid,
                title=sess.get("title", "新会话"),
                kb_id=sess.get("kb_id", "default"),
                mode=sess.get("mode", "rag"),
                created_at=sess.get("created_at", 0),
                updated_at=sess.get("updated_at", 0),
                message_count=len(sess.get("messages", [])),
            ))
        except Exception:
            continue
    # tie-breaker: time.time() 在 Windows 上精度只有 ~15ms, 同一毫秒内创建的
    # 两个会话 updated_at 完全相同. 只按 updated_at 排序时结果取决于 dict 插入
    # 顺序, 会话列表顺序会在请求之间随机跳动. 补上 created_at / session_id
    # 让排序完全确定.
    out.sort(key=lambda x: (x.updated_at, x.created_at, x.session_id), reverse=True)
    return out


@router.get("/{session_id}", response_model=SessionDetail, summary="获取会话详情")
async def get_session(session_id: str, user: str = Depends(get_current_user)):
    data = _load(user)
    sess = data.get(session_id)
    if not sess:
        raise HTTPException(status_code=404, detail="会话不存在")
    return SessionDetail(
        session_id=session_id,
        title=sess.get("title", "新会话"),
        kb_id=sess.get("kb_id", "default"),
        mode=sess.get("mode", "rag"),
        created_at=sess.get("created_at", 0),
        updated_at=sess.get("updated_at", 0),
        message_count=len(sess.get("messages", [])),
        messages=[HistoryItem(**m) for m in sess.get("messages", [])],
    )


@router.post("", summary="创建新会话")
async def create_session(payload: Optional[dict] = None, user: str = Depends(get_current_user)):
    payload = payload or {}
    sid = "sess_" + uuid.uuid4().hex[:12]
    now = time.time()
    sess = {
        "title": payload.get("title", "新会话"),
        "kb_id": payload.get("kb_id", "default"),
        "mode": payload.get("mode", "rag"),
        "created_at": now,
        "updated_at": now,
        "messages": [],
    }
    with _user_lock(user):
        data = _load(user)
        data[sid] = sess
        _save(user, data)
    return {"session_id": sid}


@router.post("/{session_id}/append", summary="追加一条消息")
async def append_message(session_id: str, item: HistoryItem, user: str = Depends(get_current_user)):
    with _user_lock(user):
        data = _load(user)
        sess = data.get(session_id)
        if not sess:
            sess = {
                "title": item.content[:30] or "新会话",
                "kb_id": "default",
                "mode": "rag",
                "created_at": time.time(),
                "updated_at": time.time(),
                "messages": [],
            }
            data[session_id] = sess
        sess["messages"].append(item.model_dump())
        sess["updated_at"] = time.time()
        if sess.get("title") == "新会话" and item.role == "user":
            sess["title"] = item.content[:30] or "新会话"
        data[session_id] = sess
        _save(user, data)
    return {"session_id": session_id, "message_count": len(sess["messages"])}


@router.delete("/{session_id}", summary="删除会话")
async def delete_session(session_id: str, user: str = Depends(get_current_user)):
    with _user_lock(user):
        data = _load(user)
        if session_id in data:
            del data[session_id]
            _save(user, data)
    # checkpoint 已落 SQLite, 不删就**永久残留**; 删除必须走 aforget_thread ——
    # 异步 saver 的同步 delete_thread 在事件循环线程里会自死锁 (见 graph.py).
    try:
        from app.agents.graph import aforget_thread

        await aforget_thread(f"{user}:{session_id}")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"aforget_thread 失败 (session={session_id}): {e}")
    return {"ok": True}
