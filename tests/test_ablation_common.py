"""_ablation_common 单测: 改写回放 / 单路补丁 / settings 切换 / 延迟统计.

全程 fakes (install_all_fakes), 不碰真实模型与磁盘库.
"""
from __future__ import annotations

import pytest

from tests import _ablation_common as ab
from tests._fakes import FakeBM25Store, FakeVectorStore, install_all_fakes

GEN = {
    "保修多久": {"rewrite": ["保修多久", "免费维修期限", "质保期多长"], "hyde": "本产品保修一年。"},
}


@pytest.fixture()
def fakes(monkeypatch):
    install_all_fakes(monkeypatch)
    from app.rag import retriever as rt_mod

    # install_all_fakes patch 的是 vector_store/bm25_store 源模块, 但
    # HybridRetriever 顶层 `from app.rag.vector_store import ChromaStore` 把类名
    # 绑进了 retriever 命名空间 —— 不补这两刀, conftest 强制的 chroma 分支会构建
    # 真实 ChromaStore, seed 时的空 metadata 直接抛错, 单测就撞上真磁盘库了.
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    rt_mod.reset_retriever_cache()
    yield
    rt_mod.reset_retriever_cache()
    ab.uninstall_rewrite_replay()  # 未安装时是 no-op, 统一收尾防泄漏


def test_replay_rewriter_uses_cache_and_respects_enabled(fakes):
    ab.install_rewrite_replay(GEN)
    from app.rag.pipeline import QueryRewriter

    assert QueryRewriter(enabled=True).rewrite("保修多久") == ["保修多久", "免费维修期限", "质保期多长"]
    # 单 query 配置 (enabled=False): 原样返回, 不查缓存
    assert QueryRewriter(enabled=False).rewrite("保修多久") == ["保修多久"]
    # 缓存缺失回退原句, 不抛错
    assert QueryRewriter(enabled=True).rewrite("没有缓存的问题") == ["没有缓存的问题"]

    ab.uninstall_rewrite_replay()
    # 卸载后必须恢复生产实现, 否则污染同进程的其他测试
    from app.rag.query_rewrite import QueryRewriter as RealRewriter

    from app.rag import pipeline as pl

    assert pl.QueryRewriter is RealRewriter


def test_replay_enhancer_returns_cached_hyde(fakes):
    ab.install_rewrite_replay(GEN)
    from app.rag.pipeline import MetadataEnhancer

    assert MetadataEnhancer.hyde_query("保修多久") == "本产品保修一年。"
    # 缓存缺失回退原问题 (与生产 HyDE 回退行为一致)
    assert MetadataEnhancer.hyde_query("没有缓存的问题") == "没有缓存的问题"


def _seed_disjoint(kb: str = "default") -> None:
    """vector 只进 v1, bm25 只进 b1 —— 用行为差异证明单路补丁真的关掉了另一路."""
    from app.rag.retriever import get_retriever

    r = get_retriever(kb)
    r.vector_store.add(
        ids=["v1"],
        documents=["支持微信支付宝银联支付"],
        embeddings=r.embedding.embed(["支持微信支付宝银联支付"]),
        metadatas=[{}],
    )
    r.bm25_store.add(ids=["b1"], documents=["发票需在订单完成后30天内申请开具"], metadatas=[{}])


# 检索词取文档原文整句: 无论分词粒度如何, 两条 token 序列完全一致,
# BM25 必然命中, 不把单测耦合到分词器实现上.
_QUERY = "发票需在订单完成后30天内申请开具"


def test_single_path_vector_only_drops_bm25_docs(fakes):
    _seed_disjoint()
    from app.rag.retriever import get_retriever

    saved = ab.patch_single_path("default", "vector")
    try:
        hits = get_retriever("default").retrieve(_QUERY)
    finally:
        ab.restore_stores(saved)
    # BM25 路打空后, 只剩向量路里的 v1
    assert [h["id"] for h in hits] == ["v1"]


def test_single_path_bm25_only_drops_vector_docs(fakes):
    _seed_disjoint()
    from app.rag.retriever import get_retriever

    saved = ab.patch_single_path("default", "bm25")
    try:
        hits = get_retriever("default").retrieve(_QUERY)
    finally:
        ab.restore_stores(saved)
    # 向量路打空 (含 embedding 桩) 后, 只剩 BM25 路里的 b1
    assert [h["id"] for h in hits] == ["b1"]


def test_restore_stores_brings_back_hybrid(fakes):
    _seed_disjoint()
    from app.rag.retriever import get_retriever

    saved = ab.patch_single_path("default", "vector")
    ab.restore_stores(saved)
    hits = get_retriever("default").retrieve(_QUERY)
    # 恢复后双路都在: 两路的文档都该出现
    assert {h["id"] for h in hits} == {"v1", "b1"}


def test_settings_apply_and_restore_roundtrip():
    from app.core.config import settings

    keys = ("query_rewrite_enabled", "hyde_query_enabled")
    # 两个字段各自记原值: 它们的默认值不同 (True / False), 共用一个 orig 会错判
    origs = {k: getattr(settings, k) for k in keys}
    saved = ab.apply_settings({k: not v for k, v in origs.items()})
    try:
        for k, v in origs.items():
            assert getattr(settings, k) == (not v)
    finally:
        ab.restore_settings(saved)
    for k, v in origs.items():
        assert getattr(settings, k) == v


def test_latency_stats_and_percentile():
    stats = ab.latency_stats([100.0, 200.0, 300.0, 400.0])
    assert stats["mean"] == 250.0
    assert stats["p50"] == 300.0  # n=4 时 percentile(0.5) 取 idx=2
    assert stats["min"] == 100.0
    assert stats["max"] == 400.0
    assert stats["n"] == 4
    assert ab.latency_stats([]) == {"n": 0, "mean": 0.0, "p50": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}


def test_config_matrix_shapes():
    m = ab.config_matrix("bm25", True)
    assert m["single-single"]["path"] == "bm25"
    assert m["single-single"]["settings"]["query_rewrite_enabled"] is False
    assert m["dual-multi"]["path"] == "hybrid"
    assert m["dual-multi"]["settings"]["query_rewrite_enabled"] is True
    assert m["dual-multi"]["settings"]["hyde_query_enabled"] is True


# ---------------------------------------------------------------------------
# LLM judge (--judge llm): 纯函数部分, verdict 预填不触 LLM
# ---------------------------------------------------------------------------


def test_parse_verdict_variants():
    from tests.run_retrieval_ablation import _parse_verdict

    assert _parse_verdict("是") is True
    assert _parse_verdict(" 否") is False
    assert _parse_verdict("Yes.") is True
    assert _parse_verdict("NO") is False
    assert _parse_verdict("1") is True
    assert _parse_verdict("") is None
    assert _parse_verdict("无法判断") is None


def test_llm_eval_math_with_prefilled_cache():
    from tests.run_retrieval_ablation import _llm_eval

    dataset = [{"question": "q1"}, {"question": "q2"}]
    retrievals = {
        "q1": [{"id": "a", "document": "x"}, {"id": "b", "document": "y"}],
        "q2": [{"id": "c", "document": "z"}],
    }
    # q1: 第 1 个相关; q2: 唯一候选不相关. cache 预填满 -> 不会真的调 LLM
    cache = {("q1", "a"): True, ("q1", "b"): False, ("q2", "c"): False}

    rep = _llm_eval(dataset, retrievals, cache)

    assert rep["n"] == 2
    assert rep["recall"]["@3"] == 0.5  # hit-rate: q1 命中, q2 无
    assert rep["precision"]["@3"] == 0.25  # (1/2 + 0/1) / 2
    assert rep["mrr"]["@3"] == 0.5
    assert rep["per_question"][1]["mrr"]["@5"] == 0.0
