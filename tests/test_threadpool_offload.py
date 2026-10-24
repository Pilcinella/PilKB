"""P1-6 测试: 同步 CPU 密集型调用卸到 threadpool, 不阻塞事件循环.

验证:
1. HybridRetriever.retrieve_async / add_documents_async 真的把同步方法丢到 threadpool
   (不是阻塞主事件循环).
2. asyncio.to_thread 包装后, 在 async 函数里 await 不会 block 其他并发协程.

(sync_iter_to_async 的测试随 app/core/streaming.py 移除: LLM 流式改走
langchain 原生 astream, 不再需要同步迭代器搬运.)
"""
from __future__ import annotations

import asyncio
import threading
import time

import pytest


@pytest.mark.asyncio
async def test_add_documents_async_does_not_block_loop():
    """add_documents_async 必须真正把同步调用抛到另一个线程."""
    from app.rag.retriever import HybridRetriever

    retriever = HybridRetriever(kb_id="default")

    main_thread = threading.get_ident()
    worker_threads: list[int] = []

    real_add = retriever.add_documents

    def wrapped(*args, **kwargs):
        worker_threads.append(threading.get_ident())
        # 在子线程里短暂 sleep, 模拟 CPU 密集
        time.sleep(0.05)
        # 返回 None, 不真入库
        return None

    retriever.add_documents = wrapped  # type: ignore[assignment]
    try:
        # 跑两次, 确认 worker 线程 != 主事件循环线程
        await retriever.add_documents_async(["a"], ["x"], [{}])
        # asyncio.to_thread 会在事件循环的默认 executor 跑, executor 线程 != 主线程
        # 这里主线程是 pytest 测试线程 (不是 asyncio loop thread),
        # 所以只要 worker 跟调用线程不同就行.
        assert len(worker_threads) >= 1
        assert all(t != main_thread for t in worker_threads), \
            "add_documents 应当在另一个线程跑, 不能在调用线程跑"
    finally:
        retriever.add_documents = real_add  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_retrieve_async_does_not_block_loop():
    """retrieve_async 也是 to_thread 包装."""
    from app.rag.retriever import HybridRetriever

    retriever = HybridRetriever(kb_id="default")

    main_thread = threading.get_ident()
    worker_threads: list[int] = []

    def fake_retrieve(query, top_k=None):
        worker_threads.append(threading.get_ident())
        time.sleep(0.05)
        return []

    retriever.retrieve = fake_retrieve  # type: ignore[assignment]
    try:
        await retriever.retrieve_async("hi")
        assert len(worker_threads) == 1
        assert worker_threads[0] != main_thread
    finally:
        retriever.retrieve = lambda q, top_k=None: []  # type: ignore[assignment]


@pytest.mark.asyncio
async def test_async_routes_run_concurrently():
    """3 个并发 retrieve_async 在主线程 sleep 期间都能完成, 证明不阻塞.

    直观思路: 假设每个 retrieve 要 sleep 100ms (模拟 embedding).
    如果是阻塞调用, 串行需要 ~300ms; 如果并发行, ~100ms.
    """
    from app.rag.retriever import HybridRetriever

    retriever = HybridRetriever(kb_id="default")

    def slow_retrieve(query, top_k=None):
        time.sleep(0.1)
        return [{"id": "x", "document": "ok"}]

    retriever.retrieve = slow_retrieve  # type: ignore[assignment]

    t0 = time.monotonic()
    results = await asyncio.gather(
        retriever.retrieve_async("a"),
        retriever.retrieve_async("b"),
        retriever.retrieve_async("c"),
    )
    elapsed = time.monotonic() - t0
    # to_thread 用的是默认 threadpool, 三个并发请求应该能在 ~100ms 内完成
    assert len(results) == 3
    assert elapsed < 0.25, f"并发 retrieve_async 应该并行, 但耗时 {elapsed:.3f}s"
