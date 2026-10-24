"""进程内轻量限流器.

设计:
1. 不引入新三方依赖 (题目要求). 用标准库 threading + time.
2. 两个独立的桶:
   - LOGIN_FAIL_BUCKET: 同 username + IP 维度, 滑动窗口 (5min 内 5 次失败).
     超过则锁定 15min (LOGIN_LOCK_KEY 单独存解锁时间戳).
   - CHAT_TOKEN_BUCKET: 每用户 / 每 IP 固定窗口 (每分钟 N 次). 超额返回 429.
3. 存储是 dict + Lock; 测试时可调 reset() 清空.
4. 生产环境如果 Redis 已就绪, 未来可换 storage. 当前实现只覆盖单进程; 多进程
   部署会各算各的, 但 login 暴力破解只要 5 次失败就锁定, 单进程拦截足够防爆.
"""
from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Tuple

from loguru import logger

from app.core.config import get_settings


# 计数桶用 OrderedDict 当 LRU 用: 每次访问 move_to_end, 淘汰时从头部删.
#
# 为什么必须有上界: 登录失败桶按 (ip, username) 建键, 撞库攻击灌进海量组合,
# 每个 key 即使只剩空 deque 也永不释放 -> 内存被打满。光清理"已过期"的桶挡不住
# 窗口内的短时爆发 (5 分钟窗口里攻击者能灌进几百万个 key), 所以还要按 LRU 淘汰.
_FAIL_WINDOW: "OrderedDict[Tuple[str, str], Deque[float]]" = OrderedDict()
_FAIL_LOCK: Dict[Tuple[str, str], float] = {}  # 解锁时间戳
_CHAT_BUCKET: "OrderedDict[str, Deque[float]]" = OrderedDict()

_LOCK = threading.Lock()

# 任一桶的条目数超过这个量级才触发一次清理. 清理后回落到 _GC_TARGET.
_GC_THRESHOLD = 10_000
_GC_TARGET = 5_000
# 锁定表的硬上限. 达到后不再新增锁定记录 (当前请求照常拒绝), 避免被灌爆;
# 不淘汰已有锁定, 否则攻击者靠"灌满再重试"就能洗掉自己的锁定.
_FAIL_LOCK_MAX = 50_000


def _prune_expired(bucket: Deque[float], cutoff: float) -> None:
    """丢掉滑出窗口的时间戳 (deque 左端最旧)."""
    while bucket and bucket[0] < cutoff:
        bucket.popleft()


def _gc_lru_locked(store: "OrderedDict", cutoff: float, now: float) -> None:
    """必须在 _LOCK 内调用. 清掉过期桶, 仍超阈值则按 LRU 淘汰到 _GC_TARGET."""
    if len(store) < _GC_THRESHOLD:
        return
    # 1) 先弹掉每个桶里滑出窗口的时间戳
    for dq in store.values():
        _prune_expired(dq, cutoff)
    # 2) 删掉已经空掉的桶
    for key in [k for k, dq in store.items() if not dq]:
        del store[key]
    # 3) 仍然超阈值 -> 按最久未使用顺序淘汰, 保证内存有硬上界
    excess = len(store) - _GC_TARGET
    if excess > 0:
        for key in list(store.keys())[:excess]:
            del store[key]


def _gc_locked(now: float) -> None:
    """必须在 _LOCK 内调用. 抑制限流状态的长期内存增长."""
    cfg = get_settings()
    _gc_lru_locked(_CHAT_BUCKET, now - 60.0, now)
    _gc_lru_locked(_FAIL_WINDOW, now - cfg.rate_limit_login_window_sec, now)
    # 锁定表只清过期的: LRU 淘汰会削弱防爆破能力 (见 _FAIL_LOCK_MAX 注释).
    if len(_FAIL_LOCK) >= _GC_THRESHOLD:
        for key in [k for k, until in _FAIL_LOCK.items() if until <= now]:
            del _FAIL_LOCK[key]


def reset_for_tests() -> None:
    """仅供测试: 清空所有限流状态."""
    with _LOCK:
        _FAIL_WINDOW.clear()
        _FAIL_LOCK.clear()
        _CHAT_BUCKET.clear()


def set_enabled(enabled: bool) -> None:
    """仅供测试: 切换限流开关. 通过设 env var 让 get_settings() 重新解析.

    注意: 因为 Settings 是 pydantic BaseModel + BaseSettings, 直接 setattr 会
    触发 frozen/validated model 的报错. 这里用环境变量统一覆盖.
    """
    import os
    os.environ["RATE_LIMIT_ENABLED"] = "1" if enabled else "0"
    get_settings.cache_clear()


@dataclass
class RateLimitDecision:
    allowed: bool
    retry_after: float = 0.0
    reason: str = ""


def _now() -> float:
    return time.monotonic()


def _bucket_key(scope: str, identifier: str) -> Tuple[str, str]:
    return (scope, identifier)


def check_login_failure(username: str, ip: str) -> RateLimitDecision:
    """登录前询问: 该 (username, ip) 是否被锁定."""
    cfg = get_settings()
    if not cfg.rate_limit_enabled:
        return RateLimitDecision(allowed=True)
    with _LOCK:
        key = _bucket_key("login", f"{ip}|{username}")
        now = _now()
        _gc_locked(now)
        # 1) 是否在锁定窗口内
        until = _FAIL_LOCK.get(key)
        if until and until > now:
            return RateLimitDecision(
                allowed=False,
                retry_after=until - now,
                reason="login locked: too many failures",
            )
        # 2) 滑动窗口计数
        dq = _FAIL_WINDOW.setdefault(key, deque())
        _FAIL_WINDOW.move_to_end(key)  # LRU: 标记最近使用
        cutoff = now - cfg.rate_limit_login_window_sec
        _prune_expired(dq, cutoff)
        if len(dq) >= cfg.rate_limit_login_fail_max:
            # 锁定. 锁定表达到硬上限时不再新增记录, 但本次请求照常拒绝 --
            # 宁可让表停在一个规模, 也不能被随机组合灌爆内存.
            if len(_FAIL_LOCK) < _FAIL_LOCK_MAX:
                _FAIL_LOCK[key] = now + cfg.rate_limit_login_lock_sec
            logger.warning(f"login locked: {ip}/{username} after {len(dq)} failures")
            return RateLimitDecision(
                allowed=False,
                retry_after=cfg.rate_limit_login_lock_sec,
                reason="login locked: too many failures",
            )
    return RateLimitDecision(allowed=True)


def record_login_failure(username: str, ip: str) -> None:
    cfg = get_settings()
    if not cfg.rate_limit_enabled:
        return
    with _LOCK:
        now = _now()
        _gc_locked(now)
        key = _bucket_key("login", f"{ip}|{username}")
        dq = _FAIL_WINDOW.setdefault(key, deque())
        _FAIL_WINDOW.move_to_end(key)  # LRU: 标记最近使用
        dq.append(now)


def record_login_success(username: str, ip: str) -> None:
    """登录成功清掉该 (user, ip) 的失败计数, 不影响其他 IP/账号."""
    with _LOCK:
        key = _bucket_key("login", f"{ip}|{username}")
        _FAIL_WINDOW.pop(key, None)
        _FAIL_LOCK.pop(key, None)


def check_fixed_quota(
    scope: str, identifier: str, limit: int, window_sec: float
) -> RateLimitDecision:
    """通用固定窗口配额: 每个 identifier 在 window_sec 内最多 limit 次.

    chat (每用户) / register (每 IP) / upload (每用户) 共用 _CHAT_BUCKET 存储,
    scope 字符串区分命名空间, 一并复用 LRU + GC 防灌爆机制 (见模块头注释).
    """
    cfg = get_settings()
    if not cfg.rate_limit_enabled:
        return RateLimitDecision(allowed=True)
    if not identifier:
        return RateLimitDecision(allowed=True)
    with _LOCK:
        key = _bucket_key(scope, identifier)
        now = _now()
        _gc_locked(now)
        dq = _CHAT_BUCKET.setdefault(key, deque())
        _CHAT_BUCKET.move_to_end(key)  # LRU: 标记最近使用
        cutoff = now - window_sec
        _prune_expired(dq, cutoff)
        if len(dq) >= limit:
            retry = window_sec - (now - dq[0]) if dq else window_sec
            logger.warning(f"{scope} rate limit hit: {identifier}")
            return RateLimitDecision(
                allowed=False,
                retry_after=max(retry, 0.001),
                reason=f"{scope} quota exceeded",
            )
        dq.append(now)
    return RateLimitDecision(allowed=True)


def check_chat_quota(user: str, ip: str) -> RateLimitDecision:
    """chat/chat-stream 配额: 每个 user 每分钟 N 次 (委托通用固定窗口实现).

    对已认证请求用 user, 未认证 (理论上 chat 都要求 auth) 退化为 IP.
    """
    identifier = user or ip
    return check_fixed_quota(
        "chat", identifier, get_settings().rate_limit_chat_per_min, 60.0
    )