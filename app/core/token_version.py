"""token_version: 给 JWT 补上吊销能力.

背景: JWT 是无状态的, 登出只清 cookie —— token 本身在过期前 (默认 24h) 一直
有效, 泄露后没有任何手段作废。加一个每用户的 token_version:

  - users 表新增 token_version 列 (默认 0), 登录时写进 JWT payload ("ver");
  - get_current_user 校验 payload.ver == DB 当前值, 不等即 401 (Token revoked);
  - 登出时 bump, 该用户名下所有旧 token 立即失效 (将来改密也走同一入口).

校验要查 DB, 为了不把每个请求都变成一次 SQLite 查询, 用进程内 TTL 缓存挡住
(与限流器相同的单副本取舍, 见 rate_limit.py 模块说明)。TTL 由
AUTH_TOKEN_CACHE_TTL_SEC 控制; 登出在本进程内是即时失效 (bump 顺手清缓存),
TTL 只影响"别的线程已取到的旧值"这一小段窗口.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import Optional, Tuple

from app.core.config import get_settings
from app.core.logging import logger

_LOCK = threading.Lock()
# username -> (version, 缓存到期时间戳). OrderedDict 当 LRU 用, 防海量用户名灌爆。
_CACHE: "OrderedDict[str, Tuple[int, float]]" = OrderedDict()
_MAX_ENTRIES = 10_000


def _lookup(username: str) -> Optional[int]:
    from app.db import SessionLocal, User

    db = SessionLocal()
    try:
        row = db.query(User.token_version).filter(User.username == username).first()
        return int(row[0]) if row else None
    finally:
        db.close()


def get_current_version(username: str) -> Optional[int]:
    """读某用户当前 token_version (TTL 缓存). 用户不在 DB 返回 None (如 dev-user).

    None 不做缓存: "注册后立刻登录"这类场景不能被一条负缓存卡到 TTL 结束。
    """
    cfg = get_settings()
    ttl = cfg.auth_token_cache_ttl_sec
    now = time.monotonic()
    with _LOCK:
        hit = _CACHE.get(username)
        if hit is not None:
            _CACHE.move_to_end(username)
            version, expires_at = hit
            if ttl <= 0 or now < expires_at:
                return version
    version = _lookup(username)
    if version is None:
        return None
    with _LOCK:
        _CACHE[username] = (version, now + max(ttl, 0.0))
        _CACHE.move_to_end(username)
        while len(_CACHE) > _MAX_ENTRIES:
            _CACHE.popitem(last=False)
    return version


def bump_version(username: str) -> None:
    """token_version + 1 并清缓存: 该用户所有旧 token 失效. 用户不存在则 no-op."""
    from app.db import SessionLocal, User

    with _LOCK:
        _CACHE.pop(username, None)
    db = SessionLocal()
    try:
        n = (
            db.query(User)
            .filter(User.username == username)
            .update({User.token_version: User.token_version + 1}, synchronize_session=False)
        )
        db.commit()
        if n:
            logger.info(f"audit: token revoked (token_version bumped) for user={username}")
    finally:
        db.close()


def reset_for_tests() -> None:
    with _LOCK:
        _CACHE.clear()
