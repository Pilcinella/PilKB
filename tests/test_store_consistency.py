"""测试: 存储一致性自检 (BM25 内存 docs vs 向量库 chunk 数).

背景: 这两份存储相互独立, 任何一边出问题都不会抛异常 —— BM25 空掉之后检索会
静默退化为纯向量路, 接口仍是 200, 只有父文档召回不再生效. 启动自检负责把这种
状态用日志暴露出来, 这些用例守住它的判据与"只读不修"的约定.
"""
from __future__ import annotations

import pytest

from tests._fakes import FakeBM25Store, FakeEmbeddingService, FakeVectorStore


@pytest.fixture(autouse=True)
def _install_fakes(monkeypatch):
    """用 fake 替换 Chroma / BM25 / embedding, 让本文件不依赖 chromadb 与 torch."""
    import os

    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    os.environ["PILKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    # retriever.py 在 import 时 capture 了两个类的引用, 只改原模块不够.
    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)

    rt_mod.reset_retriever_cache()  # 实例缓存会把上一个用例的 store 带过来
    yield
    rt_mod.reset_retriever_cache()
    emb_mod.get_embedding_service.cache_clear()
    os.environ.pop("PILKB_FAKE_EMBEDDING", None)


def _seed(kb_id: str, n: int = 3):
    """灌 n 个 chunk, 两路都写 (与真实上传链路一致)."""
    from app.rag.retriever import get_retriever

    r = get_retriever(kb_id)
    r.add_documents(
        ids=[f"d{i}_c0" for i in range(n)],
        documents=[f"文档 {i} 的内容" for i in range(n)],
        metadatas=[{"doc_id": f"d{i}", "chunk_index": 0} for i in range(n)],
    )
    return r


def test_consistent_after_ingest():
    from app.rag.retriever import check_store_consistency

    _seed("kb_ok", 3)
    st = check_store_consistency("kb_ok")
    assert st["vector_count"] == 3
    assert st["bm25_count"] == 3
    assert st["consistent"] is True


def test_empty_kb_is_consistent():
    """全新部署: 两边都是空, 应当判为一致 (不能把空库误报成故障)."""
    from app.rag.retriever import check_store_consistency

    st = check_store_consistency("kb_empty")
    assert st["vector_count"] == 0
    assert st["bm25_count"] == 0
    assert st["consistent"] is True


def test_detects_lost_bm25_updates():
    """模拟多进程丢更新: BM25 内存只剩一部分, 向量库仍是全量.

    这正是 --workers 2 下的真实故障形态 (每个进程各持一份内存 docs, 各自整表
    覆盖写同一个 pickle), 也是自检要抓到的那种"不报错但已降级"的状态.
    """
    from app.rag.retriever import check_store_consistency, get_retriever

    _seed("kb_drift", 3)
    r = get_retriever("kb_drift")
    r.bm25_store.docs = r.bm25_store.docs[:1]  # 覆盖写丢了 2 条

    st = check_store_consistency("kb_drift")
    assert st["bm25_count"] == 1
    assert st["vector_count"] == 3
    assert st["consistent"] is False


def test_check_is_read_only():
    """自检只读: 发现不一致也不得顺手修复或改动任何状态."""
    from app.rag.retriever import check_store_consistency, get_retriever

    _seed("kb_readonly", 2)
    r = get_retriever("kb_readonly")
    r.bm25_store.docs = r.bm25_store.docs[:1]

    check_store_consistency("kb_readonly")
    assert len(r.bm25_store.docs) == 1  # 没有被"顺手修复"
    assert r.vector_store.count() == 2


def test_list_kb_ids_includes_default():
    """公共白名单里的 kb 即使没在 registry 登记, 也必须被自检覆盖到."""
    from app.core.kb_registry import list_kb_ids

    assert "default" in list_kb_ids()


async def test_lifespan_reports_mismatch(monkeypatch):
    """把自检接到启动流程上: 不一致时必须在日志里出现 error (而不是静默启动)."""
    import app.main as main_mod
    from app.rag.retriever import get_retriever

    _seed("default", 2)
    get_retriever("default").bm25_store.docs = []  # BM25 全丢, 向量库仍有 2 条

    records: list[tuple[str, str]] = []

    class _RecordingLogger:
        def __getattr__(self, level):
            return lambda msg, *a, **k: records.append((level, str(msg)))

    monkeypatch.setattr(main_mod, "logger", _RecordingLogger())

    async with main_mod.lifespan(None):
        pass

    assert any(
        level == "error" and "不一致" in msg and "bm25=0" in msg and "vector=2" in msg
        for level, msg in records
    ), f"启动日志里没有出现不一致告警: {records}"
