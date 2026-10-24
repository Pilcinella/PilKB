"""测试: APIReranker (HTTP rerank 后端). 用 httpx.MockTransport 打桩, 不联网.

重点覆盖两类**静默**失败 —— 它们都不会抛异常, 只会让检索结果莫名变差:
  1) **分数错位**: 服务端返回的 results 是**按分数降序**排列的, 必须按 `index`
     回填. 若按顺序 zip, 高分会被挂到错误的候选上.
  2) **降级不可见**: 重试耗尽后要退回原始顺序并打 warning, 不能悄悄用未重排的结果.
"""
from __future__ import annotations

import json

import httpx
import pytest

from app.rag import reranker as rk_mod
from app.rag.reranker import APIReranker, RerankOptimizer

BASE = "https://fake.rerank.invalid/v1"


def _make(handler, **kwargs) -> APIReranker:
    """构造 APIReranker, 并把内部 client 换成 MockTransport 版本 (不联网).

    直接构造不会发请求, 换掉 `_client` 即可注入桩 —— 生产代码不需要为测试开口子.
    必须把原 client 的 headers 带过去, 否则测不到 Authorization 头.
    """
    r = APIReranker(
        model="BAAI/bge-reranker-v2-m3", base_url=BASE, api_key="test-key", **kwargs
    )
    r._client = httpx.Client(
        base_url=BASE,
        headers=r._client.headers,
        transport=httpx.MockTransport(handler),
    )
    return r


def _ok(results):
    def handler(request):
        return httpx.Response(200, json={"results": results})

    return handler


def _no_sleep(monkeypatch):
    """去掉退避等待, 否则每个重试用例要白等 1.5s."""
    monkeypatch.setattr(rk_mod.time, "sleep", lambda _s: None)


# ---------------------------------------------------------------------------
# 分数对齐: 最容易错且不会报错的地方
# ---------------------------------------------------------------------------


def test_scores_are_mapped_by_index_not_by_order():
    """results 按分数降序, index 是乱的 —— 必须按 index 回填."""
    handler = _ok([
        {"index": 2, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.5},
        {"index": 3, "relevance_score": 0.1},
        {"index": 1, "relevance_score": 0.05},
    ])
    r = _make(handler)
    # 输入顺序 doc0..doc3 对应的分数应当是 0.5 / 0.05 / 0.9 / 0.1
    assert r._score("q", ["doc0", "doc1", "doc2", "doc3"]) == [0.5, 0.05, 0.9, 0.1]


def test_missing_results_default_to_zero():
    """服务端少返回了几条时, 缺失的候选得 0 分而不是让整个列表错位."""
    handler = _ok([{"index": 0, "relevance_score": 0.7}])
    r = _make(handler)
    assert r._score("q", ["a", "b"]) == [0.7, 0.0]


def test_out_of_range_index_is_ignored():
    handler = _ok([{"index": 99, "relevance_score": 1.0}])
    r = _make(handler)
    assert r._score("q", ["a"]) == [0.0]


def test_accepts_score_field_as_fallback_key():
    """部分 provider 用 `score` 而不是 `relevance_score`."""
    handler = _ok([{"index": 0, "score": 0.42}])
    r = _make(handler)
    assert r._score("q", ["a"]) == [0.42]


# ---------------------------------------------------------------------------
# 契约: 与本地版一致 (下游 RerankOptimizer / pipeline / chat 都不改)
# ---------------------------------------------------------------------------


def test_rerank_sorts_desc_and_truncates():
    handler = _ok([
        {"index": 1, "relevance_score": 0.9},
        {"index": 0, "relevance_score": 0.2},
        {"index": 2, "relevance_score": 0.5},
    ])
    out = _make(handler).rerank("q", [{"document": t} for t in "abc"], top_k=2)
    assert [c["document"] for c in out] == ["b", "c"]
    assert [c["rerank_score"] for c in out] == [0.9, 0.5]


def test_empty_candidates_short_circuits():
    """空候选直接返回, 不该发请求."""
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(200, json={"results": []})

    assert _make(handler).rerank("q", [], top_k=5) == []
    assert calls["n"] == 0


def test_optimizer_works_with_api_reranker():
    """RerankOptimizer 必须能直接吃 APIReranker (契约兼容性回归护栏)."""
    handler = _ok([{"index": 0, "relevance_score": 0.9}, {"index": 1, "relevance_score": 0.1}])
    opt = RerankOptimizer()
    candidates = [
        {"document": "a", "rrf_score": 0.5},
        {"document": "b", "rrf_score": 0.4},
    ]
    out = opt.optimize("q", candidates, pool=10, top_n=2, reranker=_make(handler))
    assert [c["document"] for c in out] == ["a", "b"]
    assert out[0]["rerank_score"] == 0.9


# ---------------------------------------------------------------------------
# 请求构造
# ---------------------------------------------------------------------------


def test_request_payload_shape():
    seen = {}

    def handler(request):
        seen["path"] = request.url.path
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    _make(handler).rerank("保修期多久", [{"document": "x"}], top_k=1)
    assert seen["path"] == "/v1/rerank"
    assert seen["auth"] == "Bearer test-key"
    body = seen["body"]
    assert body["model"] == "BAAI/bge-reranker-v2-m3"
    assert body["query"] == "保修期多久"
    assert body["documents"] == ["x"]
    # 显式要回全部候选的分数, 否则缺分数的候选会退化成 0 分排到末尾.
    assert body["top_n"] == 1
    assert body["return_documents"] is False


def test_documents_truncated_to_max_chars():
    """本地 CrossEncoder 也按 max_length 截断, api 侧保持一致以免按 token 全额计费."""
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 1.0}]})

    _make(handler, max_chars=10).rerank("q", [{"document": "x" * 100}], top_k=1)
    assert seen["body"]["documents"] == ["x" * 10]


# ---------------------------------------------------------------------------
# 重试与降级
# ---------------------------------------------------------------------------


def test_retries_on_429_then_succeeds(monkeypatch):
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, json={"message": "rate limited"})
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

    out = _make(handler, max_retries=3).rerank("q", [{"document": "a"}], top_k=1)
    assert calls["n"] == 2
    assert out[0]["rerank_score"] == 0.8


def test_retries_on_5xx(monkeypatch):
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(503 if calls["n"] == 1 else 200,
                              json={"results": [{"index": 0, "relevance_score": 0.3}]})

    _make(handler, max_retries=3).rerank("q", [{"document": "a"}], top_k=1)
    assert calls["n"] == 2


def test_does_not_retry_on_4xx(monkeypatch):
    """400/401 是请求本身的问题 (模型名错、key 失效), 重试只会把同一个错误再犯一遍."""
    _no_sleep(monkeypatch)
    calls = {"n": 0}

    def handler(request):
        calls["n"] += 1
        return httpx.Response(400, json={"message": "bad request"})

    _make(handler, max_retries=3).rerank("q", [{"document": "a"}], top_k=1)
    assert calls["n"] == 1


def test_falls_back_to_original_order_and_warns(monkeypatch):
    """重试耗尽后退回原始顺序 —— 但必须打 warning, 不能静默."""
    _no_sleep(monkeypatch)
    warnings: list[str] = []
    monkeypatch.setattr(rk_mod.logger, "warning", lambda msg, *a, **k: warnings.append(str(msg)))

    def handler(request):
        return httpx.Response(500, json={})

    docs = [{"document": "a"}, {"document": "b"}]
    out = _make(handler, max_retries=2).rerank("q", docs, top_k=2)
    assert [c["document"] for c in out] == ["a", "b"]      # 原序
    assert all("rerank_score" not in c for c in out)       # 不编造分数
    assert warnings and "fallback" in warnings[0]


def test_network_error_falls_back(monkeypatch):
    _no_sleep(monkeypatch)
    monkeypatch.setattr(rk_mod.logger, "warning", lambda *a, **k: None)

    def handler(request):
        raise httpx.ConnectError("connection refused")

    out = _make(handler, max_retries=2).rerank("q", [{"document": "a"}], top_k=1)
    assert [c["document"] for c in out] == ["a"]


# ---------------------------------------------------------------------------
# 后端分派
# ---------------------------------------------------------------------------


def test_get_reranker_dispatches_to_api(monkeypatch):
    monkeypatch.setattr(rk_mod.settings, "rerank_backend", "api")
    monkeypatch.setattr(rk_mod.settings, "rerank_api_key", "k")
    rk_mod.reset_reranker_cache()
    try:
        r = rk_mod.get_reranker()
        assert isinstance(r, APIReranker)
        assert r.model == "BAAI/bge-reranker-v2-m3"
    finally:
        rk_mod.reset_reranker_cache()


def test_get_reranker_defaults_to_local(monkeypatch):
    monkeypatch.setattr(rk_mod.settings, "rerank_backend", "local")
    rk_mod.reset_reranker_cache()
    try:
        assert isinstance(rk_mod.get_reranker(), rk_mod.CrossEncoderReranker)
    finally:
        rk_mod.reset_reranker_cache()


def test_get_reranker_api_without_key_raises(monkeypatch):
    """缺 key 必须显式报错, 不能悄悄退回本地模型 (两者分数量纲不同)."""
    monkeypatch.setattr(rk_mod.settings, "rerank_backend", "api")
    monkeypatch.setattr(rk_mod.settings, "rerank_api_key", "")
    monkeypatch.delenv("RERANK_API_KEY", raising=False)
    rk_mod.reset_reranker_cache()
    try:
        with pytest.raises(RuntimeError, match="RERANK_API_KEY"):
            rk_mod.get_reranker()
    finally:
        rk_mod.reset_reranker_cache()


def test_get_reranker_is_singleton(monkeypatch):
    monkeypatch.setattr(rk_mod.settings, "rerank_backend", "api")
    monkeypatch.setattr(rk_mod.settings, "rerank_api_key", "k")
    rk_mod.reset_reranker_cache()
    try:
        assert rk_mod.get_reranker() is rk_mod.get_reranker()
    finally:
        rk_mod.reset_reranker_cache()
