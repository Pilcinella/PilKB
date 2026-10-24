"""P1-7 测试: BM25 惰性重建.

覆盖点:
1. add() 之后立即 query, 应能命中 (lazy build 被触发).
2. 连续多次 add() 只在首次 query 时 rebuild 一次.
3. 无 docs 时 query 返回 [], 不抛异常.
4. dirty 状态: add() 后 _bm25 仍为 None (并未真正重建).
5. delete() 同样走 dirty 路径, 待下次 query 才重建.
6. 并发 query 也不会引发多次重建 (锁保证仅一次).
7. 显式 rebuild(force=True) 立即重建.
8. 持久化 + load 之后 _dirty=True, 首次 query 触发重建.
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest


@pytest.fixture
def store(monkeypatch, tmp_path):
    """每个用例拿到全新 BM25Store (持久化目录位于 tmp_path)."""
    monkeypatch.chdir(tmp_path)
    from app.rag.bm25_store import BM25Store
    return BM25Store(persist_path=str(tmp_path / "bm25.pkl"))


def _add(store, ids, docs):
    metas = [{"i": i} for i in range(len(docs))]
    store.add(ids, docs, metas)


def test_add_then_query_triggers_lazy_build(store):
    """add() 不触发重建; 首次 query 才会触发."""
    assert store._bm25 is None
    _add(store, ["a"], ["你好世界 hello"])
    # dirty 已置位, 但 _bm25 仍为 None
    assert store._dirty is True
    assert store._bm25 is None
    hits = store.query("你好")
    assert len(hits) == 1
    assert hits[0]["id"] == "a"
    # 重建完成后 dirty=False
    assert store._dirty is False
    assert store._bm25 is not None


def test_multiple_adds_only_rebuild_once(store):
    """连续 3 次 add(), 仅在首次 query 时重建一次."""
    _add(store, ["a"], ["foo bar"])
    _add(store, ["b"], ["baz qux"])
    _add(store, ["c"], ["hello world"])
    assert store._bm25 is None  # 尚未 query
    assert store.count() == 3
    hits = store.query("foo")
    assert len(hits) >= 1
    # 一次 query 之后, _bm25 应已是 build 过的对象
    assert store._bm25 is not None
    # 再 query 一次也不应重新 build (dirty=False)
    bm25_before = store._bm25
    _ = store.query("hello")
    assert store._bm25 is bm25_before


def test_query_with_no_docs_returns_empty():
    """docs 为空时不抛异常, 直接给出 []."""
    from app.rag.bm25_store import BM25Store
    s = BM25Store()
    assert s.query("anything") == []
    assert s.count() == 0


def test_dirty_flag_set_after_add(store):
    """add() 必然将 dirty 置为 True."""
    _add(store, ["x"], ["test"])
    assert store._dirty is True


def test_dirty_flag_reset_after_query(store):
    """query 之后 dirty 应被清除."""
    _add(store, ["x"], ["test"])
    store.query("test")
    assert store._dirty is False


def test_delete_marks_dirty_and_triggers_rebuild(store):
    """delete() 同样走 lazy 路径."""
    _add(store, ["a", "b"], ["alpha beta", "gamma delta"])
    # 先 query 一次以建立索引
    store.query("alpha")
    assert store._bm25 is not None
    assert store._dirty is False
    # 执行删除 -> dirty
    n = store.delete(["a"])
    assert n == 1
    assert store.count() == 1
    assert store._dirty is True
    # 立刻 query 将触发重建
    hits = store.query("alpha")
    assert all(h["id"] != "a" for h in hits)
    assert store._dirty is False


def test_delete_nonexistent_returns_zero(store):
    _add(store, ["a"], ["x"])
    store.query("x")
    n = store.delete(["zzz"])
    assert n == 0


def test_concurrent_queries_only_rebuild_once(store):
    """并发 query 场景下, 重建锁保证仅 build 一次."""
    _add(store, [f"d{i}" for i in range(50)], [f"term{i}" for i in range(50)])
    assert store._bm25 is None  # 处于 dirty
    results: list = []
    errors: list = []

    def worker():
        try:
            results.append(store.query("term1"))
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert len(results) == 8
    assert store._bm25 is not None
    # 所有 query 给出相同的排序结果 (只因 build 了一次)
    first = [r[0]["id"] for r in results if r]
    assert len(set(first)) == 1  # 结果完全一致


def test_rebuild_force(store):
    """显式调用 rebuild(force=True) 立即重建."""
    _add(store, ["a"], ["x"])
    # 尚未 query, _bm25 为 None
    assert store._bm25 is None
    store.rebuild(force=True)
    assert store._bm25 is not None
    assert store._dirty is False


def test_rebuild_no_force_clears_dirty(store):
    """非强制 rebuild 同样会清除 dirty 标记 (因为已经 build 过)."""
    _add(store, ["a"], ["x"])
    store.rebuild(force=False)
    assert store._dirty is False


def test_persist_and_reload_resets_dirty(store, tmp_path):
    """持久化 + load 之后 _dirty=True, 首次 query 才 build."""
    _add(store, ["a", "b"], ["foo bar", "baz qux"])
    # 首次 query 触发 build
    store.query("foo")
    # 此时 _bm25 应已存在, 落盘也保留了 docs
    assert Path(store.persist_path).exists()

    # 重新实例化并 load 持久化数据
    from app.rag.bm25_store import BM25Store
    s2 = BM25Store(persist_path=store.persist_path)
    s2.load()
    assert s2.count() == 2
    # load 之后 dirty 应为 True (设计如此: 启动时按需重建)
    assert s2._dirty is True
    assert s2._bm25 is None
    hits = s2.query("foo")
    assert len(hits) >= 1
    assert s2._dirty is False
    assert s2._bm25 is not None


def test_query_keeps_returning_after_rebuild(store):
    """多次 rebuild + query 不影响结果的稳定性."""
    for round_i in range(3):
        _add(store, [f"r{round_i}"], [f"text {round_i}"])
        hits = store.query("text")
        ids = {h["id"] for h in hits}
        assert f"r{round_i}" in ids


def test_concurrent_query_with_delete_does_not_raise_indexerror(store):
    """Regression: query() 与 delete() 并发时不应抛出 IndexError.

    旧实现在 query() 中以 `self.docs[idx]` 取回文档, 而 idx 来自 _bm25 建索引
    时那份语料的下标. 一旦另一线程在此期间 delete() 截短了 self.docs, idx 即越界.
    修复手法是在 query() 内对 docs 取局部快照, 并按两者中较小长度收敛.
    """
    n = 400
    _add(store, [f"d{i}" for i in range(n)], [f"关键词 文档 {i}" for i in range(n)])
    store.query("关键词")  # 先把索引建好

    errors: list[BaseException] = []
    stop = threading.Event()

    def _query_loop():
        while not stop.is_set():
            try:
                store.query("关键词", top_k=10)
            except BaseException as e:  # noqa: BLE001
                errors.append(e)
                return

    def _delete_loop():
        # 反复删除尾部文档, 制造 docs 长度的变化
        for i in range(n - 1, 0, -1):
            store.delete([f"d{i}"])
        stop.set()

    qt = threading.Thread(target=_query_loop, daemon=True)
    dt = threading.Thread(target=_delete_loop, daemon=True)
    qt.start()
    dt.start()
    dt.join(timeout=30)
    stop.set()
    qt.join(timeout=30)

    assert not errors, f"并发 query/delete 抛出了异常: {errors[:3]}"