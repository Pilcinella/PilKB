"""知识库注册中心: 记录每个 kb_id 的 owner 与是否公共, 解决 IDOR 越权访问.

设计:
1. 数据结构极简: 持久化到 ./data/kb_registry.json, 一个 {kb_id: {owner, is_public, created_at}}.
2. "公共" 有两个来源, 读校验两者都认: settings 的 public_kb_ids 白名单 (系统级
   只读 kb, 如 default), 以及 registry 里 is_public=True 的 kb (所有者把自己的 kb
   标成公开; 目前没有改这个标记的 API, 但字段从现在起真正参与决策, 不再是只存不判).
3. 写操作 (upload) 会保证 kb 存在并属于当前用户; 若 kb 已被别人占用则返回 403.
   公开的判定只拦白名单 kb; registry 里 is_public 的 kb 归 owner 本人继续可写.
4. 读操作 (list / chat) 对公共 kb 直接放行; 对私有 kb 必须 owner == 当前用户.
5. 并发保护: 单进程内用 threading.Lock 串行化; 多进程写同一文件的最坏情况由
   操作系统原子写 (os.replace) 保证不会损坏文件, 偶发的覆盖竞争在 demo 场景可接受.
6. 进程内缓存 (write-through): 注册表首访加载进内存, 之后读写都走内存、写时落盘.
   ensure_kb_for_read 在每个 chat / list 请求的热路径上, 原先每次都 read_text +
   json parse, 等于把磁盘 IO 串行化塞进最热链路. 缓存的代价是多进程下看不到别
   进程的写入 (直到重启), 与第 5 点的单副本取舍一致.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from pathlib import Path
from typing import Optional

from fastapi import HTTPException, status

from app.core.config import DATA_DIR, get_settings
from app.core.logging import logger

# 注册表文件路径 (同 Chroma / BM25 等持久化目录一致).
# 锚定到 data/, 避免随 CWD 漂移.
_REGISTRY_PATH = DATA_DIR / "kb_registry.json"
_LOCK = threading.Lock()
# 进程内注册表缓存 (None = 尚未从磁盘加载). 所有读写都在 _LOCK 内进行,
# _save_unlocked 落盘的同时更新缓存, 保证两者永不分叉。
_CACHE: dict[str, dict] | None = None
# public_kb_ids 白名单的解析缓存, 键是原始配置串: 测试改 env + cache_clear 后
# 原始串变化, 自动触发重算, 不需要单独的失效钩子。
_PUBLIC_IDS_CACHE: tuple[str, frozenset[str]] | None = None
# admin_users 白名单的解析缓存, 同上一模式.
_ADMIN_USERS_CACHE: tuple[str, frozenset[str]] | None = None

# kb_id 会被直接拼进文件名 (./data/bm25_{kb_id}.pkl) 和 Chroma collection 名
# (kb_{kb_id}), 不校验就等于把路径穿越 (../../etc/x) 和非法 collection 名
# 交给用户控制. 这里用严格白名单: 字母/数字/下划线/连字符, 1~64 位.
# 同时也满足 Chroma collection 命名约束 (3-63 字符, 字母数字与 _-).
_KB_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_valid_kb_id(kb_id: str) -> bool:
    """纯函数版校验, 供非 HTTP 层 (retriever/评估脚本) 复用同一套规则."""
    return bool(_KB_ID_RE.match((kb_id or "").strip()))


def validate_kb_id(kb_id: str) -> str:
    """校验并返回规范化的 kb_id, 非法直接 400.

    放在 registry 层是因为所有读写入口 (upload/list/delete/chat) 都会先过
    ensure_kb_for_read / ensure_kb_for_write, 在这里拦一次就能覆盖全链路。
    """
    kb_id = (kb_id or "").strip()
    if not _KB_ID_RE.match(kb_id):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="非法的 kb_id: 只允许字母、数字、下划线和连字符, 长度 1-64",
        )
    return kb_id


class KnowledgeBase:
    """单个知识库的元信息."""

    __slots__ = ("kb_id", "owner", "is_public", "created_at")

    def __init__(
        self,
        kb_id: str,
        owner: str,
        is_public: bool = False,
        created_at: Optional[float] = None,
    ):
        self.kb_id = kb_id
        self.owner = owner
        self.is_public = is_public
        self.created_at = created_at or time.time()

    def to_dict(self) -> dict:
        return {
            "owner": self.owner,
            "is_public": self.is_public,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, kb_id: str, data: dict) -> "KnowledgeBase":
        return cls(
            kb_id=kb_id,
            owner=data.get("owner", ""),
            is_public=bool(data.get("is_public", False)),
            created_at=float(data.get("created_at", 0.0)) or None,
        )


def _load_unlocked() -> dict[str, dict]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    data: dict[str, dict] = {}
    if _REGISTRY_PATH.exists():
        try:
            data = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            logger.warning(f"kb_registry.json 解析失败, 将重新初始化: {e}")
            data = {}
    _CACHE = data
    return data


def _save_unlocked(data: dict[str, dict]) -> None:
    global _CACHE
    _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _REGISTRY_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, _REGISTRY_PATH)
    _CACHE = data


def _load_all() -> dict[str, dict]:
    """线程安全的读."""
    with _LOCK:
        return _load_unlocked()


def _save_all(data: dict[str, dict]) -> None:
    """线程安全的写 (atomic replace)."""
    with _LOCK:
        _save_unlocked(data)


def _public_kb_ids() -> frozenset[str]:
    """从 settings 读公共 kb 白名单 (解析结果按原始配置串缓存, 见 _PUBLIC_IDS_CACHE).

    接受两种格式:
    - 逗号分隔: "default,shared,public-docs"
    - JSON 数组: '["default","shared"]'
    默认 fallback 到 ["default"].

    注意: 通过 get_settings() 而非模块级 settings, 避免 lru_cache 复用导致
    测试改 env 不生效.
    """
    global _PUBLIC_IDS_CACHE
    cfg = get_settings()
    raw = getattr(cfg, "public_kb_ids", None) or "default"
    if _PUBLIC_IDS_CACHE is not None and _PUBLIC_IDS_CACHE[0] == raw:
        return _PUBLIC_IDS_CACHE[1]
    items: list[str] = []
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        s = raw.strip()
        if not s:
            items = []
        elif s.startswith("["):
            try:
                import json as _json
                parsed = _json.loads(s)
                if isinstance(parsed, list):
                    items = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                items = []
        if not items:
            items = [x.strip() for x in s.split(",") if x.strip()]
    result = frozenset(items or ["default"])
    # 竞态最坏是两个线程各解析一次, 结果幂等, 不值得加锁。
    _PUBLIC_IDS_CACHE = (raw, result)
    return result


def is_public_kb(kb_id: str) -> bool:
    """判断某个 kb 是否在公共白名单 (无论是否在 registry 中登记)."""
    return kb_id in _public_kb_ids()


def _admin_user_set() -> frozenset[str]:
    """解析 settings.admin_users 白名单 (逗号分隔或 JSON 数组).

    与 _public_kb_ids 同一模式: 按原始配置串缓存, 测试改 env 后原串变化自动重算.
    """
    global _ADMIN_USERS_CACHE
    cfg = get_settings()
    raw = getattr(cfg, "admin_users", None) or ""
    if _ADMIN_USERS_CACHE is not None and _ADMIN_USERS_CACHE[0] == raw:
        return _ADMIN_USERS_CACHE[1]
    items: list[str] = []
    if isinstance(raw, list):
        items = [str(x).strip() for x in raw if str(x).strip()]
    elif isinstance(raw, str):
        s = raw.strip()
        if s.startswith("["):
            try:
                parsed = json.loads(s)
                if isinstance(parsed, list):
                    items = [str(x).strip() for x in parsed if str(x).strip()]
            except Exception:
                items = []
        if not items:
            items = [x.strip() for x in s.split(",") if x.strip()]
    result = frozenset(items)
    _ADMIN_USERS_CACHE = (raw, result)
    return result


def is_admin_user(user: str) -> bool:
    """是否为配置授权的管理员 (settings.admin_users 白名单)."""
    if not user:
        return False
    return user in _admin_user_set()


def list_kb_ids() -> list[str]:
    """列出全部已知 kb_id: registry 里登记过的 + 公共白名单.

    用于启动自检这类需要遍历所有知识库的场景. 排序后返回, 保证多次调用 / 多个
    进程的输出顺序一致, 便于比对日志.
    """
    return sorted(set(_load_all().keys()) | _public_kb_ids())


def list_kbs_for_user(user: str) -> list[KnowledgeBase]:
    """列出当前用户可访问的全部 kb: 自己的私有 kb + 公共 kb (白名单或 is_public).

    供 GET /api/v1/kbs (前端知识库选择器) 使用. 可见性判据与 ensure_kb_for_read
    单一来源对齐: owner == user, 或 kb 在公共白名单, 或 registry 里标了 is_public.

    只出现在白名单、从未注册过的 kb (如 default) 合成 owner="" 的占位对象,
    让前端无需区分"真注册过"与"仅配置存在".
    """
    data = _load_all()
    public_set = _public_kb_ids()
    visible: dict[str, KnowledgeBase] = {}
    for kb_id, raw in data.items():
        kb = KnowledgeBase.from_dict(kb_id, raw)
        if kb.owner == user or kb.is_public or kb_id in public_set:
            visible[kb_id] = kb
    for kb_id in public_set:
        if kb_id not in visible:
            visible[kb_id] = KnowledgeBase(kb_id=kb_id, owner="", is_public=True)
    return sorted(visible.values(), key=lambda kb: kb.kb_id)


def get_kb(kb_id: str) -> Optional[KnowledgeBase]:
    data = _load_all()
    raw = data.get(kb_id)
    if not raw:
        return None
    return KnowledgeBase.from_dict(kb_id, raw)


def ensure_kb_for_read(kb_id: str, user: str) -> KnowledgeBase:
    """读权限校验: 公共 kb 直接放行; 私有 kb 要求 owner==user.

    "公共"认两处 (见模块 docstring 第 2 点): settings 白名单, 或 registry 里
    is_public=True. 后者此前只存不判 —— 数据结构看着支持逐 kb 公开标记, 决策却
    只认配置文件, 属于双源真相, 现在让字段真正参与决策.

    Raises:
        400: kb_id 含非法字符
        404: kb 不存在 (私有 kb 用户无法感知其存在, 视为不存在)
        403: kb 存在但不属于当前用户
    """
    kb_id = validate_kb_id(kb_id)
    kb = get_kb(kb_id)
    # 1) 公共: 白名单 (不必注册) 或 registry 里标记了 is_public
    if is_public_kb(kb_id) or (kb is not None and kb.is_public):
        if kb is None:
            # 公共 kb 没有显式注册过, 视为 owner="" (系统) 即可
            return KnowledgeBase(kb_id=kb_id, owner="", is_public=True)
        return kb

    # 2) 私有 kb: 必须存在且 owner 匹配
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库不存在: {kb_id}",
        )
    if kb.owner != user:
        # 不区分 "不存在" 与 "无权访问", 避免泄露 kb 是否存在
        logger.warning(f"KB access denied: user={user} tried kb_id={kb_id} owned by={kb.owner}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问该知识库",
        )
    return kb


def ensure_kb_for_write(kb_id: str, user: str) -> KnowledgeBase:
    """写权限校验: 若 kb 不存在则自动创建并归属当前用户; 若存在则必须 owner==user.

    Raises:
        400: kb_id 含非法字符
        403: kb 存在但属于其他用户 (包括公共 kb: 不允许往默认公共 kb 写)
    """
    kb_id = validate_kb_id(kb_id)
    public_set = _public_kb_ids()
    # 公共 kb 不允许普通用户写入 (保持只读); 如果业务需要可再放开.
    if kb_id in public_set:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"公共知识库 '{kb_id}' 不可写入, 请上传到自己的知识库",
        )

    with _LOCK:
        data = _load_unlocked()
        raw = data.get(kb_id)
        if raw is None:
            # 每用户 kb 数上限: 每个 kb 都会在 retriever 实例缓存里常驻一份
            # BM25 docs + collection, 无上限可被批量建 kb 耗尽内存.
            owned = sum(1 for v in data.values() if v.get("owner") == user)
            if owned >= get_settings().max_kbs_per_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"每个用户最多创建 {get_settings().max_kbs_per_user} 个知识库",
                )
            kb = KnowledgeBase(kb_id=kb_id, owner=user, is_public=False)
            data[kb_id] = kb.to_dict()
            _save_unlocked(data)
            logger.info(f"Created new KB: kb_id={kb_id} owner={user}")
            return kb
        kb = KnowledgeBase.from_dict(kb_id, raw)
        if kb.owner != user:
            logger.warning(
                f"KB write denied: user={user} tried to write kb_id={kb_id} owned by={kb.owner}"
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="无权访问该知识库",
            )
        return kb


def remove_kb(kb_id: str) -> None:
    """从注册表删除条目 (存在才删, 幂等).

    只动 registry 这一处; 向量库 / BM25 的数据清理由调用方 (DELETE /kbs,
    见 app.api.kbs) 通过 retriever.delete_kb_storage 完成 —— 注册表模块刻意
    不 import 检索层, 保持它零依赖、可被任何层安全引用的现状。
    """
    with _LOCK:
        data = _load_unlocked()
        if data.pop(kb_id, None) is not None:
            _save_unlocked(data)
            logger.info(f"Removed KB from registry: kb_id={kb_id}")


def ensure_kb_for_rebuild(kb_id: str, user: str, *, audit: bool = True) -> KnowledgeBase:
    """BM25 索引重建的权限校验: 私有 kb 要 owner, 公共 kb 要管理员.

    为什么不复用 ensure_kb_for_write:
      - 它会**自动创建**不存在的 kb (重建时凭空建库没有意义);
      - 它一律拒绝公共 kb, 而公共 kb 恰恰是损坏最常发生、也最需要有人能修的地方
        (它是全站共享的只读库, 没有 owner 可以承担这个职责).

    也不复用 ensure_kb_for_read: 读权限下任意登录用户都能触发公共 kb 的全量重建
    (O(N) 读向量库 + 全量落盘), 属于把运维级操作开放给所有访客.

    audit=False 供 can_rebuild_bm25 这类**能力探测**复用: 拒绝时的 warning 是
    "有人试图越权重建"的审计记录, 而探测 ("能不能重建") 不是尝试 —— documents/list
    每次都会探测公共 kb, 打出来既污染审计流, 又会在排障时误导方向 (满屏假 denied).

    Raises:
        400: kb_id 含非法字符
        404: 私有 kb 不存在
        403: 私有 kb 非 owner, 或公共 kb 非管理员
    """
    kb_id = validate_kb_id(kb_id)
    kb = get_kb(kb_id)

    # 1) 公共 kb (白名单 或 registry 里标了 is_public): 仅管理员
    if is_public_kb(kb_id) or (kb is not None and kb.is_public):
        if is_admin_user(user):
            return kb if kb is not None else KnowledgeBase(kb_id=kb_id, owner="", is_public=True)
        if audit:
            logger.warning(f"KB rebuild denied: user={user} tried public kb_id={kb_id} (not admin)")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="公共知识库的索引重建仅限管理员",
        )

    # 2) 私有 kb: 必须存在且 owner 匹配 (与读校验同构)
    if kb is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"知识库不存在: {kb_id}",
        )
    if kb.owner != user:
        if audit:
            logger.warning(
                f"KB rebuild denied: user={user} tried kb_id={kb_id} owned by={kb.owner}"
            )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="无权访问该知识库",
        )
    return kb


def can_rebuild_bm25(kb_id: str, user: str) -> bool:
    """能否重建该 kb 的 BM25 (纯判据, 不抛异常).

    供 API 返回值使用 (前端据此决定是否显示重建按钮). 判据与
    ensure_kb_for_rebuild 单一来源, 避免两处规则漂移. 探测不产生
    "rebuild denied" 审计日志 (见 ensure_kb_for_rebuild 的 audit 参数).
    """
    try:
        ensure_kb_for_rebuild(kb_id, user, audit=False)
        return True
    except HTTPException:
        return False


def can_write_kb(kb_id: str, user: str) -> bool:
    """能否往该 kb 写入 (供前端决定是否显示删除等写入口). 纯判据, 无副作用.

    规则与 ensure_kb_for_write 一致: 白名单公共 kb 一律不可写; 私有 kb 必须
    owner (registry 里标了 is_public 的 kb, owner 本人仍可写 —— 与写校验的既有
    语义保持一致).

    有两处刻意的差异, 所以不能直接 try/except 包 ensure_kb_for_write:
      1. 不抛异常 (前端要的是布尔值);
      2. **无副作用** —— ensure_kb_for_write 对不存在的 kb 会**自动创建**并归属
         当前用户, 拿它当判据会让"查看文档列表"这种读操作凭空建库.
    对不存在的 kb 返回 False: 列表页此时没有可删除的对象.
    """
    if not is_valid_kb_id(kb_id):
        return False
    if is_public_kb(kb_id):
        return False
    kb = get_kb(kb_id)
    if kb is None:
        return False
    return kb.owner == user


def reset_for_tests() -> None:
    """仅供测试使用: 清空注册表 (文件 + 内存缓存), 让每个测试隔离."""
    global _CACHE
    with _LOCK:
        _CACHE = None
        if _REGISTRY_PATH.exists():
            _REGISTRY_PATH.unlink()