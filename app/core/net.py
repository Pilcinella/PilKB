"""Client IP 解析: 只有直连 peer 可信时, 才采信 X-Forwarded-For.

为什么不能无条件读 XFF: 它只是普通请求头, 客户端想填什么填什么。原先 auth /
chat 两处都是"有 XFF 就取第一个", 等于把限流键的 ip 维度交给攻击者控制 ——
每次失败换一个假 IP, (ip, username) 键就永远不重复, 5 次锁定形同虚设。

正确姿势 (与 nginx / uvicorn --forwarded-allow-ips 的语义一致):
  1. 直连 peer 不在 TRUSTED_PROXIES 名单内 → XFF 整个丢弃, 用直连地址;
  2. peer 可信 → 从 XFF **从右往左**找第一个不在名单里的地址。右端由最近一台
     可信代理追加 (真实来源), 左端可能是客户端自带的原生伪造头 —— 所以旧的
     "取第一个"即使在真反代后面也是取到了最容易被伪造的那一端.
"""
from __future__ import annotations

from fastapi import Request

from app.core.config import get_settings

# 解析结果按原始字符串缓存: client_ip 在每个 chat / login 请求的热路径上,
# 没必要每次 split。settings 本身是 lru_cache, 这里再省一次字符串切分;
# 测试改 env + cache_clear 后 raw 变化, 自动触发重算。
_trusted_cache: tuple[str, frozenset[str]] | None = None


def _trusted_proxies() -> frozenset[str]:
    global _trusted_cache
    raw = (get_settings().trusted_proxies or "").strip()
    if _trusted_cache is not None and _trusted_cache[0] == raw:
        return _trusted_cache[1]
    parsed = frozenset(x.strip() for x in raw.split(",") if x.strip())
    # 竞态最坏也只是两个线程各解析一次, 结果幂等, 不值得加锁。
    _trusted_cache = (raw, parsed)
    return parsed


def _is_trusted_peer(peer: str, trusted: frozenset[str]) -> bool:
    # "*" 只表示"无条件信任直连方" (本机调试用), 不意味着 XFF 里随便哪一跳都可信。
    return "*" in trusted or peer in trusted


def client_ip(request: Request) -> str:
    """提取发起请求的客户端 IP (可信代理边界内解析 XFF, 见模块 docstring)."""
    peer = request.client.host if request.client else ""
    trusted = _trusted_proxies()
    if not trusted or not peer or not _is_trusted_peer(peer, trusted):
        # 未配置可信代理 / 直连方不在名单: XFF 不可信, 直接用直连地址。
        return peer or "unknown"

    fwd = request.headers.get("x-forwarded-for", "")
    hops = [h.strip() for h in fwd.split(",") if h.strip()]
    for hop in reversed(hops):
        if hop not in trusted:
            return hop
    # 整条链都是可信代理 (少见): 拿不到真实来源, 退回直连地址。
    return peer or "unknown"


def reset_cache_for_tests() -> None:
    global _trusted_cache
    _trusted_cache = None
