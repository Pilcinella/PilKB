"""P1-8 测试: HybridRetriever 按 kb_id 维度的实例缓存与写并发锁.

覆盖点:
1. 同一 kb_id 连续 get_retriever() 拿到同一对象.
2. kb_id 不同则对象不同.
3. reset_retriever_cache() 清空后再 get_retriever() 会新建实例.
4. 并发 get_retriever() 不重复创建 (有锁保护).
5. add_documents 持写锁: 两个并发 add 被串行化.
6. delete_documents 同步删除两侧存储.
7. retrieve 属于读操作, 无需写锁, 可并发执行.

不依赖真实 chromadb / sentence_transformers:
- 涉及 add/delete 的测试会把 HybridRetriever 的 vector_store / embedding 换成 fake.
"""
from __future__ import annotations

import threading
import time

import pytest


class _FakeVectorStore:
    """替身实现: 摆脱 chromadb 依赖, 接口与 ChromaStore.add/delete/count 对齐."""
    def __init__(self):
        self._ids: list[str] = []
        self._write_lock = threading.Lock()

    def add(self, ids, documents, embeddings, metadatas):
        with self._write_lock:
            for i in ids:
                self._ids.append(i)

    def delete(self, ids):
        with self._write_lock:
            before = len(self._ids)
            self._ids = [i for i in self._ids if i not in set(ids)]
            return before - len(self._ids)

    def count(self):
        return len(self._ids)


class _FakeEmbedding:
    """替身实现: 摆脱 sentence_transformers 依赖."""
    dim = 4

    def embed(self, docs):
        return [[0.0] * self.dim for _ in docs]

    def embed_query(self, q):
        return [0.0] * self.dim


@pytest.fixture(autouse=True)
def _clean_cache(monkeypatch, tmp_path):
    # 把 BM25Store 引到 tmp_path, 免得上一个测试的持久化文件污染下一个.
    # 注意: BM25 的 persist 目录现取自 settings.bm25_persist_dir (绝对路径, 不随
    # CWD 漂移), 单靠 chdir 已无法隔离, 必须显式覆盖该配置项.
    from app.core.config import settings
    from app.rag.retriever import reset_retriever_cache

    monkeypatch.setattr(settings, "bm25_persist_dir", str(tmp_path))
    reset_retriever_cache()
    yield
    reset_retriever_cache()


def _patch_retriever_for_test(retriever):
    """绕开真实 chromadb / sentence_transformers."""
    retriever.vector_store = _FakeVectorStore()
    retriever.embedding = _FakeEmbedding()


def test_get_retriever_returns_same_instance():
    """同一 kb_id 应返回同一对象 (省去每次重建 ChromaStore/BM25Store 的开销)."""
    from app.rag.retriever import get_retriever

    a1 = get_retriever("alpha")
    a2 = get_retriever("alpha")
    assert a1 is a2


def test_different_kb_ids_get_different_instances():
    from app.rag.retriever import get_retriever

    a = get_retriever("alpha")
    b = get_retriever("beta")
    assert a is not b
    assert a.kb_id == "alpha"
    assert b.kb_id == "beta"


def test_reset_clears_cache():
    from app.rag.retriever import get_retriever, reset_retriever_cache

    a1 = get_retriever("alpha")
    reset_retriever_cache()
    a2 = get_retriever("alpha")
    # 已是新对象
    assert a1 is not a2


def test_concurrent_get_retriever_no_double_init():
    """N 个线程并发 get_retriever(kb_id), 实例只允许创建一次."""
    from app.rag.retriever import get_retriever

    instances = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        r = get_retriever("shared")
        instances.append(id(r))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(set(instances)) == 1, "只应创建出 1 个实例"


def test_add_documents_serializes_with_lock():
    """两个并发 add 应被写锁串行化, 不产生数据竞争.

    思路: 把 embedding.embed patch 成带 sleep 的版本, 再并发发起两次 add.
    无锁时两个 embed 会同时进行; 有锁则两次依次执行, 总耗时约为 2 倍 sleep.
    """
    from app.rag.retriever import get_retriever

    retriever = get_retriever("lock_test")
    _patch_retriever_for_test(retriever)

    call_log: list[str] = []
    call_log_lock = threading.Lock()

    def slow_embed(docs):
        with call_log_lock:
            call_log.append("start:" + docs[0][:5])
        time.sleep(0.08)
        with call_log_lock:
            call_log.append("end:" + docs[0][:5])
        return [[0.0] * 4 for _ in docs]

    retriever.embedding.embed = slow_embed  # type: ignore[assignment]

    t1 = threading.Thread(target=retriever.add_documents, args=(["a"], ["alpha"], [{}]))
    t2 = threading.Thread(target=retriever.add_documents, args=(["b"], ["beta"], [{}]))
    t0 = time.monotonic()
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    elapsed = time.monotonic() - t0

    # 串行表现为 start A -> end A -> start B -> end B (或反序).
    # 任何 "start" 之后紧跟另一个 "start" 都意味着锁没起作用.
    for i, line in enumerate(call_log):
        if line.startswith("start:"):
            if i + 1 < len(call_log):
                nxt = call_log[i + 1]
                assert nxt.startswith("end:"), \
                    f"并发 add 未被串行化: start 之后直接跟着 {nxt}"
    # 串行总耗时约 0.16s, 并行约 0.08s; 留出少量 buffer.
    assert elapsed > 0.13, f"两次 add 理应串行 (~0.16s), 实测 {elapsed:.3f}s — 锁未生效"


def test_delete_documents_removes_from_both_stores():
    """delete_documents 需同时清理 BM25 与 Chroma 两侧."""
    from app.rag.retriever import get_retriever

    retriever = get_retriever("del_test")
    _patch_retriever_for_test(retriever)
    retriever.add_documents(
        ids=["c1", "c2", "c3"],
        documents=["alpha beta", "gamma delta", "epsilon zeta"],
        metadatas=[{}, {}, {}],
    )
    assert retriever.bm25_store.count() == 3
    assert retriever.vector_store.count() == 3

    n = retriever.delete_documents(["c1", "c2"])
    assert n == 2
    assert retriever.bm25_store.count() == 1
    assert retriever.vector_store.count() == 1


def test_delete_documents_with_no_ids():
    from app.rag.retriever import get_retriever

    retriever = get_retriever("del_test2")
    _patch_retriever_for_test(retriever)
    retriever.bm25_store.add(["x"], ["y"], [{}])
    n = retriever.delete_documents([])
    assert n == 0


def test_retrieve_does_not_hold_write_lock():
    """retrieve 属读操作, 不应与 add 互斥.

    思路: 先触发一个慢 retrieve, 期间并发 add, 验证 add 未被 retrieve 长时间阻塞.
    """
    from app.rag.retriever import get_retriever

    retriever = get_retriever("rw_test")
    _patch_retriever_for_test(retriever)

    real_retrieve = retriever.retrieve

    def slow_retrieve(query, top_k=None):
        time.sleep(0.2)
        return []

    retriever.retrieve = slow_retrieve  # type: ignore[assignment]
    try:
        t0 = time.monotonic()
        # 先起一个慢 retrieve
        r_thread = threading.Thread(target=retriever.retrieve, args=("foo",))
        r_thread.start()
        time.sleep(0.02)  # 给 retrieve 留一点启动时间
        # add 应能立即执行 (不被 retrieve 挡住)
        retriever.add_documents(["b"], ["baz"], [{}])
        elapsed = time.monotonic() - t0
        r_thread.join()
        # add 已在 retrieve 进行中完成, 总耗时约 0.22s
        assert elapsed < 0.35, f"add 被 retrieve 卡住了: {elapsed}s"
    finally:
        retriever.retrieve = real_retrieve  # type: ignore[assignment]
