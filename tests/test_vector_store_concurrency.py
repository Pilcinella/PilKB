"""ChromaStore 的引导并发安全 + reset 后可复用.

背景 (都是实测出来的, 不是推理):
`ChromaStore._ensure()` 原先是一段裸的 check-then-act:

    if self._client is None:
        self._client = chromadb.PersistentClient(...)
        self._col = self._client.get_or_create_collection(...)
    return self._col

两个缺陷:

1. **引导竞态.** chromadb 的首次引导 (建 sqlite / 跑 schema 迁移 / 建
   default_tenant / 建 collection) 不是并发安全的. 实测同一进程内 4 线程同时
   首次触达: 同一 store 60 次调用 26 次抛异常 (43%), 不同 store 但共用同一
   persist 目录 48 次调用 12 次抛异常 (25%). 抛的是
   `ValueError: Could not connect to tenant default_tenant` 和
   `sqlite3.OperationalError: duplicate column name: dimension` —— 后者说明多个
   线程在同一个 sqlite 上并行跑迁移. 因为所有 kb 共用同一个 persist 目录
   (settings.chroma_persist_dir), 竞争是**跨 kb** 的, 实例级锁挡不住, 所以修复
   用的是模块级 `_CHROMA_INIT_LOCK`.

2. **守卫变量选错.** 守卫看 `_client`, 但 `reset()` 只把 `_col` 置 None ——
   于是 reset 之后 `_ensure()` 会跳过重建并返回 None, store 永久失效
   (任何后续操作都 AttributeError). 现在守卫看 `_col`.

这两个测试的价值在于"修复前必失败": 去掉 vector_store.py 里的模块级锁后,
test_concurrent_cold_start_across_stores 会挂; 把守卫改回 `_client` 后,
test_reset_keeps_store_usable 会挂.
"""
from __future__ import annotations

import threading

import pytest

pytest.importorskip("chromadb", reason="需要真实 chromadb 才能验证引导竞态")

from app.rag import vector_store as vs_mod  # noqa: E402
from app.rag.vector_store import ChromaStore  # noqa: E402


def _new_store(tmp_path, name: str) -> ChromaStore:
    return ChromaStore(persist_dir=str(tmp_path / "chroma"), collection_name=name)


def _run_threads(target, n: int):
    """n 个线程用 Barrier 同时起跑, 返回 (结果列表, 异常列表).

    Barrier 是必须的: 不用它线程会被 GIL 调度错开, 竞态窗口被撞上的概率大幅下降,
    测试就变成"偶尔能复现"而不是稳定复现 (修复前实测失败率 25%~43%).
    """
    results, errors = [], []
    barrier = threading.Barrier(n)
    lock = threading.Lock()

    def worker(i: int):
        barrier.wait()
        try:
            r = target(i)
            with lock:
                results.append(r)
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return results, errors


# ---------------------------------------------------------------------------
# 1) 冷启动并发: 同一个 store, 多个线程同时首触
# ---------------------------------------------------------------------------
def test_concurrent_cold_start_same_store(tmp_path):
    """4 线程同时首触同一个冷 store, 不应抛异常, 且拿到同一个 collection."""
    for rnd in range(4):
        store = _new_store(tmp_path, f"kb_same_{rnd}")
        results, errors = _run_threads(lambda i: store._ensure(), 4)
        assert not errors, f"第 {rnd} 轮并发引导抛异常: {errors}"
        assert all(r is not None for r in results), "引导完成后 _ensure() 不得返回 None"
        assert len({id(r) for r in results}) == 1, "所有线程应拿到同一个 collection 对象"


# ---------------------------------------------------------------------------
# 2) 冷启动并发: 不同 store / 不同 collection, 但共用同一个 persist 目录
#    这是真实部署形态 (每个 kb 一个 ChromaStore, 共享 settings.chroma_persist_dir),
#    也是**必须模块级锁**才能挡住的那个场景.
# ---------------------------------------------------------------------------
def test_concurrent_cold_start_across_stores(tmp_path):
    """4 个不同 kb 的 store 同时首触共享 persist 目录, 不应抛异常."""
    for rnd in range(6):
        # 每轮换一个全新目录, 保证是"真冷"启动 (chromadb 内部按 path 缓存 system,
        # 复用同一目录会让第二轮走热路径, 竞态窗口就消失了).
        root = tmp_path / f"cold_{rnd}"
        results, errors = _run_threads(
            lambda i: ChromaStore(
                persist_dir=str(root), collection_name=f"kb_x{i}"
            ).count(),
            4,
        )
        assert not errors, f"第 {rnd} 轮跨 store 并发引导抛异常: {errors}"
        assert results == [0, 0, 0, 0], "新建的空 collection 计数应为 0"


# ---------------------------------------------------------------------------
# 3) 热路径不碰锁: 并发读的吞吐不能被这把锁吃掉
# ---------------------------------------------------------------------------
class _CountingLock:
    """记录被进入次数的假锁, 用来断言热路径完全不取锁."""

    def __init__(self):
        self.acquired = 0

    def __enter__(self):
        self.acquired += 1
        return self

    def __exit__(self, *exc):
        return False


def test_hot_path_does_not_take_init_lock(tmp_path, monkeypatch):
    """引导完成后, _ensure() 不应再进入引导锁 (否则并发读会被无谓串行化)."""
    store = _new_store(tmp_path, "kb_hot")
    assert store.count() == 0  # 先完成引导

    spy = _CountingLock()
    monkeypatch.setattr(vs_mod, "_CHROMA_INIT_LOCK", spy)

    for _ in range(20):
        store._ensure()

    assert spy.acquired == 0, "热路径进入了引导锁, 并发读会被串行化"


# ---------------------------------------------------------------------------
# 4) reset() 之后 store 仍可用 (守卫从 _client 改回 _col 的回归测试)
# ---------------------------------------------------------------------------
def test_reset_keeps_store_usable(tmp_path):
    """reset() 之后必须能继续写入/查询 —— 修复前 _ensure() 会返回 None."""
    store = _new_store(tmp_path, "kb_reset")
    store.add(["a"], ["保修期一年"], [[1.0, 0.0]], [{"doc_id": "d1"}])
    assert store.count() == 1

    store.reset()
    assert store.count() == 0, "reset() 后应为空库, 而不是 500/AttributeError"

    # 不只是能读, 还要能重新写入
    store.add(["b"], ["退换货七天"], [[0.0, 1.0]], [{"doc_id": "d2"}])
    assert store.count() == 1
    hits = store.query([0.0, 1.0], top_k=1)
    assert len(hits) == 1 and hits[0]["id"] == "b"


def test_ensure_not_none_when_client_already_connected(tmp_path):
    """直接复现并发窗口留下的状态: _client 已连上但 _col 为空.

    这正是 reset() 之后 (以及并发引导中途) 的状态. 守卫若看 `_client`,
    `_ensure()` 会跳过重建并返回 None.
    """
    store = _new_store(tmp_path, "kb_state")
    store.count()  # 完成引导, _client 就绪
    store._col = None  # 模拟 reset 后 / 并发窗口

    col = store._ensure()

    assert col is not None, "守卫看 _client 时会在这里返回 None (store 永久失效)"
    assert col.count() == 0
