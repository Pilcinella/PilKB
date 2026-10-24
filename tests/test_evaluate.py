"""测试: 离线检索评估 Recall@K / Precision@K / MRR@K + 评估器."""
from __future__ import annotations

import pytest

from app.rag.evaluate import (
    ContextRecallEvaluator,
    RecallEvaluator,
    paired_bootstrap_ci,
    precision_at_k_by_context,
    recall_at_k,
    reciprocal_rank_by_context,
)


def test_recall_at_k_perfect():
    assert recall_at_k(["a", "b"], ["a", "b", "c"], k=3) == 1.0


def test_recall_at_k_partial():
    # 相关集 2 个, 前 3 只命中 1 个 -> 0.5
    assert recall_at_k(["a", "b"], ["a", "x", "y"], k=3) == 0.5


def test_recall_at_k_truncated_by_k():
    # 相关集 2 个, 但 k=1 只取第一, 命中 1 -> 0.5
    assert recall_at_k(["a", "b"], ["a", "b"], k=1) == 0.5


def test_recall_at_k_empty_relevant():
    # 无相关项, 定义 Recall=1.0
    assert recall_at_k([], ["a"], k=5) == 1.0


def test_evaluator_runs_over_dataset():
    dataset = [
        {"question": "q1", "relevant_ids": ["d1", "d2"]},
        {"question": "q2", "relevant_ids": ["d3"]},
    ]

    def fake_retrieve(q: str) -> list[dict]:
        # 模拟检索: q1 召回 d1,d2 ; q2 召回 d3,x
        if q == "q1":
            return [{"id": "d1"}, {"id": "d2"}, {"id": "x"}]
        return [{"id": "d3"}, {"id": "x"}]

    ev = RecallEvaluator(ks=(1, 3, 5))
    res = ev.evaluate(dataset, fake_retrieve)
    assert res["n"] == 2
    # q1 相关 2 项, 但 top1 只命中 d1 -> recall@1 = 0.5; q2 top1 命中 -> 1.0; 平均 0.75
    assert res["recall"]["@1"] == 0.75
    assert res["recall"]["@3"] == 1.0
    assert len(res["per_question"]) == 2


def test_evaluator_handles_misses():
    dataset = [{"question": "q", "relevant_ids": ["missing"]}]

    def fake_retrieve(q: str) -> list[dict]:
        return [{"id": "other"}]

    res = RecallEvaluator(ks=(1, 5)).evaluate(dataset, fake_retrieve)
    assert res["recall"]["@1"] == 0.0
    assert res["recall"]["@5"] == 0.0


# ---------------------------------------------------------------------------
# 文本口径: Precision@K / RR@K
# ---------------------------------------------------------------------------


def test_precision_at_k_perfect_and_partial():
    refs = ["保修一年"]
    texts = ["本产品保修一年", "配送时效说明"]
    assert precision_at_k_by_context(refs, texts, k=1) == 1.0
    # top2 里 1 个命中 1 个无关 -> 0.5
    assert precision_at_k_by_context(refs, texts, k=2) == 0.5


def test_precision_at_k_no_refs():
    # 无 reference: 检索到什么都不算相关 (与 recall 的 1.0 相反)
    assert precision_at_k_by_context([], ["任何文本"], k=3) == 0.0


def test_precision_at_k_truncated_by_k():
    refs = ["支持微信支付"]
    texts = ["配送说明", "支付: 支持微信支付"]
    assert precision_at_k_by_context(refs, texts, k=1) == 0.0


def test_reciprocal_rank_first_hit_position():
    refs = ["支持微信支付"]
    texts = ["配送时效说明", "发票规则", "支付方式: 支持微信支付", "售后条款"]
    assert reciprocal_rank_by_context(refs, texts, k=10) == pytest.approx(1 / 3)
    # 截止 k=2 内未命中 -> 0
    assert reciprocal_rank_by_context(refs, texts, k=2) == 0.0


def test_reciprocal_rank_no_hit():
    assert reciprocal_rank_by_context(["退款政策"], ["配送说明", "保修条款"], k=5) == 0.0


def test_context_evaluator_reports_all_metric_families():
    dataset = [
        {"question": "q1", "reference_context": ["保修一年"]},
        {"question": "q2", "reference_context": ["支持微信支付"]},
    ]
    docs = {
        # q1 命中第 1 位 (且只召回 1 条); q2 命中第 2 位
        "q1": [{"document": "本产品保修一年"}],
        "q2": [{"document": "保修条款"}, {"document": "支付: 支持微信支付"}],
    }

    res = ContextRecallEvaluator(ks=(1, 2, 10)).evaluate(dataset, lambda q: docs[q])

    assert res["n"] == 2
    assert res["recall"]["@1"] == 0.5
    assert res["recall"]["@2"] == 1.0
    # q1: P@1=1/1; q2: P@1=0/1 -> 0.5. q1 库里只有 1 条, 分母取 min(k, len)=1
    assert res["precision"]["@1"] == 0.5
    assert res["precision"]["@2"] == 0.75
    # MRR@10 = (1/1 + 1/2) / 2
    assert res["mrr"]["@10"] == 0.75
    assert res["per_question"][1]["mrr"]["@2"] == 0.5


def test_context_evaluator_no_answer_excluded():
    dataset = [
        {"question": "q1", "reference_context": ["保修一年"]},
        {"question": "q-noanswer", "reference_context": []},
    ]
    docs = {
        "q1": [{"document": "本产品保修一年"}],
        "q-noanswer": [{"document": "无关文本"}],
    }

    res = ContextRecallEvaluator(ks=(1,)).evaluate(dataset, lambda q: docs[q])

    # 无答案样本不计入任何一族指标的平均
    assert res["n"] == 1
    assert res["n_no_answer"] == 1
    assert res["recall"]["@1"] == 1.0
    assert res["precision"]["@1"] == 1.0


# ---------------------------------------------------------------------------
# 配对 bootstrap 显著性
# ---------------------------------------------------------------------------


def test_paired_bootstrap_identical_series():
    a = [1.0, 0.0, 0.5, 1.0, 0.0]
    diff, lo, hi = paired_bootstrap_ci(a, a)
    assert diff == 0.0
    # 差分全 0 -> CI 收缩到 [0, 0], 跨 0 即不显著
    assert lo == 0.0 and hi == 0.0


def test_paired_bootstrap_separated_series():
    diff, lo, hi = paired_bootstrap_ci([1.0] * 10, [0.0] * 10)
    assert diff == 1.0
    assert lo > 0  # CI 不跨 0 -> 显著


def test_paired_bootstrap_rejects_mismatch():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 0.0])
