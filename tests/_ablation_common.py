"""消融评估共用件: 改写缓存回放 / 单路补丁 / settings 配置切换 / 延迟统计.

被 run_retrieval_ablation.py 与 run_ragas_ablation.py 共用, 保证两个脚本的
"单路单query vs 双路多query" 是**同一套口径** —— 同一份缓存回放、同一套单路补丁、
同一组 settings 字段, 两张表才能并排比较.

为什么不改生产代码: 单路开关的理想形态是 settings.retrieval_paths (config.py +
retriever.py 各几行), 但那要动生产链路; 评估侧用 retriever **实例级**补丁同样能
得到单路行为 —— RRF 少一路输入时退化为该路自身的排序, 融合公式不用分叉.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_jsonl(path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def load_gen_cache(path) -> dict[str, dict]:
    """读 query 改写缓存: {问题: {"rewrite": [原句+变体...], "hyde": 假设答案}}."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 改写回放
#
# pipeline.retrieve_and_rerank 内部做 "QueryRewriter.rewrite -> 多 query 并发
# 检索 -> HyDE 追加" (见 pipeline.py), 原始实现都要调 LLM. 评估把这两个入口换成
# 查表: 多 query 的内容来自 data/eval 的缓存, 检索链路全程不调改写 LLM ——
# 可复现、零成本, 也让"检索延迟"里不含改写 LLM 的秒级抖动.
# ---------------------------------------------------------------------------
_REPLAY_SAVED: dict[str, Any] = {}


def install_rewrite_replay(gen_cache: dict[str, dict]) -> None:
    """把 pipeline 的 QueryRewriter / MetadataEnhancer.hyde_query 换成缓存回放."""
    from app.agents import llm_router as lr_mod
    from app.rag import pipeline as pl

    class _ReplayRewriter:
        """接口对齐 query_rewrite.QueryRewriter (pipeline 只用这两个方法签名)."""

        def __init__(self, enabled: bool = True, max_variants: int = 3):
            self.enabled = enabled
            self.max_variants = max_variants

        def rewrite(self, query, llm=None, history=None):
            q = (query or "").strip()
            if self.enabled:
                qs = [
                    x
                    for x in (gen_cache.get(query) or {}).get("rewrite", [])
                    if x and x.strip()
                ]
                if qs:
                    return qs
            # 单 query 配置 (enabled=False) 或缓存缺失: 原样返回
            return [q] if q else []

    class _ReplayEnhancer:
        @staticmethod
        def hyde_query(question, llm=None):
            return (gen_cache.get(question) or {}).get("hyde") or question

    _REPLAY_SAVED["QueryRewriter"] = pl.QueryRewriter
    _REPLAY_SAVED["MetadataEnhancer"] = pl.MetadataEnhancer
    pl.QueryRewriter = _ReplayRewriter
    pl.MetadataEnhancer = _ReplayEnhancer
    # pipeline 的 HyDE 分支要求 LLM 适配器非 None 才进 (见 retrieve_and_rerank).
    # 只有拿不到真实适配器 (无 key 环境) 才空壳占位, 让 --with-hyde 的回放能进;
    # 有真实适配器时**绝不能**打桩 —— run_retrieval_ablation --judge llm 的相关性
    # 判定同样走这个入口, 打桩会让 judge 拿到空串、全部判成不相关.
    try:
        lr_mod.get_llm_text_callable()
        _REPLAY_SAVED["get_llm_text_callable"] = None
    except Exception:
        _REPLAY_SAVED["get_llm_text_callable"] = lr_mod.get_llm_text_callable
        lr_mod.get_llm_text_callable = lambda: (lambda prompt: "")


def uninstall_rewrite_replay() -> None:
    if not _REPLAY_SAVED:
        return
    from app.agents import llm_router as lr_mod
    from app.rag import pipeline as pl

    pl.QueryRewriter = _REPLAY_SAVED["QueryRewriter"]
    pl.MetadataEnhancer = _REPLAY_SAVED["MetadataEnhancer"]
    if _REPLAY_SAVED.get("get_llm_text_callable") is not None:
        lr_mod.get_llm_text_callable = _REPLAY_SAVED["get_llm_text_callable"]
    _REPLAY_SAVED.clear()


# ---------------------------------------------------------------------------
# 单路补丁 (retriever 实例级)
#
# vector: bm25.query -> []          embedding 照算, 向量路本来就需要
# bm25:   embed_query -> None 且 vector.query -> []
#         真实的纯 BM25 实现同样不该做 embedding, 桩一起打掉, 延迟口径才是纯 BM25.
# ---------------------------------------------------------------------------
def patch_single_path(kb_id: str, path: str) -> list[tuple[Any, str, Any]]:
    """把一路召回打空, 返回恢复所需的原属性列表 (交给 restore_stores)."""
    from app.rag.retriever import get_retriever

    r = get_retriever(kb_id)
    saved: list[tuple[Any, str, Any]] = []
    if path == "vector":
        saved.append((r.bm25_store, "query", r.bm25_store.query))
        r.bm25_store.query = lambda text, top_k=20: []
    elif path == "bm25":
        saved.append((r.embedding, "embed_query", r.embedding.embed_query))
        r.embedding.embed_query = lambda text: None
        saved.append((r.vector_store, "query", r.vector_store.query))
        r.vector_store.query = lambda qv, top_k=20, where=None: []
    return saved


def restore_stores(saved: list[tuple[Any, str, Any]]) -> None:
    for obj, name, orig in saved:
        setattr(obj, name, orig)


def apply_settings(overrides: dict) -> dict:
    """临时覆盖 settings 字段, 返回原值供 restore_settings. 字段全为现有配置项."""
    from app.core.config import settings

    saved = {k: getattr(settings, k) for k in overrides}
    for k, v in overrides.items():
        setattr(settings, k, v)
    return saved


def restore_settings(saved: dict) -> None:
    from app.core.config import settings

    for k, v in saved.items():
        setattr(settings, k, v)


def config_matrix(single_path: str, with_hyde: bool) -> dict[str, dict]:
    """消融配置矩阵. settings 字段都是现有项, 不需要新增生产开关.

    single-single: 一路召回 + 原始问题 (query_rewrite_enabled=False 时
                   回放 rewriter 自然退化为 [原句], 不需要额外分支)
    dual-multi:    向量+BM25+RRF + 缓存回放的多条改写 (生产默认形态;
                   hyde_query_enabled 生产默认 False, 要加 HyDE 传 with_hyde)
    """
    return {
        "single-single": {
            "path": single_path,
            "settings": {"query_rewrite_enabled": False, "hyde_query_enabled": False},
        },
        "dual-multi": {
            "path": "hybrid",
            "settings": {"query_rewrite_enabled": True, "hyde_query_enabled": with_hyde},
        },
    }


def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals:
        return 0.0
    return sorted_vals[min(len(sorted_vals) - 1, int(p * len(sorted_vals)))]


def latency_stats(values_ms: list[float]) -> dict:
    s = sorted(values_ms)
    return {
        "n": len(s),
        "mean": round(sum(s) / len(s), 1) if s else 0.0,
        "p50": round(percentile(s, 0.50), 1),
        "p95": round(percentile(s, 0.95), 1),
        "min": round(s[0], 1) if s else 0.0,
        "max": round(s[-1], 1) if s else 0.0,
    }
