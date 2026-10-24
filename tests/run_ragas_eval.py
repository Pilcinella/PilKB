"""实际运行 RAGAS 评估, 产出 faithfulness / answer_relevancy / context_precision / context_recall 四项指标.

两种运行模式:
  - 默认 (--llm-mode fake_llm): 使用内置 FakeLLM, 无需 API key, 供 CI / 沙箱环境验证脚本可跑通.
  - 真 LLM (--llm-mode real_llm): 使用 settings.llm_provider 所配置的真实 provider, 需在 .env 中备好 API key.

输出:
  data/eval/ragas_report.json  (含 scores 及每条样本的 question/answer/contexts/ground_truth)
  data/eval/baseline.json      (更精简, 仅保留 4 项 scores 与元信息)

用法 (在仓库根目录执行):
  python -m tests.run_ragas_eval --kb default \
      --dataset data/eval/qa.jsonl \
      --out data/eval/ragas_report.json \
      --baseline-out data/eval/baseline.json \
      --llm-mode fake_llm
"""
from __future__ import annotations

import argparse
import json
import sys
import types
from datetime import datetime
from pathlib import Path

from loguru import logger


class _StubVertexAI:  # noqa: D401 - stub
    """空壳 vertexai, 仅服务于 ragas 0.1.10 的 import 链."""

    def __init__(self, *args, **kwargs):
        raise RuntimeError("VertexAI stub: not available in fake mode")


def _stub_langchain_community_vertexai() -> None:
    """ragas 0.1.10 import 时会强拉 langchain_community.chat_models.vertexai, 而它在
    sandbox 中会因 pydantic/langchain-core 大版本错位而触发元类冲突.

    因此在 import ragas 之前, 先向 sys.modules 塞入空壳 vertexai 模块,
    让 ragas 的 import 链顺利走通; 真正评估用的是 FakeLLM, 并不依赖 vertexai.
    """
    pkg_lc = types.ModuleType("langchain_community")
    pkg_lc.__path__ = []  # 标明这是 namespace package
    sys.modules.setdefault("langchain_community", pkg_lc)

    pkg_llms = types.ModuleType("langchain_community.llms")
    # ragas 会直接 from langchain_community.llms import VertexAI
    pkg_llms.VertexAI = _StubVertexAI
    pkg_llms._VertexAICommon = _StubVertexAI
    sys.modules.setdefault("langchain_community.llms", pkg_llms)

    pkg_chat = types.ModuleType("langchain_community.chat_models")
    sys.modules.setdefault("langchain_community.chat_models", pkg_chat)

    fake_llms_module = types.ModuleType("langchain_community.llms.vertexai")
    fake_llms_module.VertexAI = _StubVertexAI
    fake_llms_module._VertexAICommon = _StubVertexAI
    sys.modules["langchain_community.llms.vertexai"] = fake_llms_module

    fake_chat_module = types.ModuleType("langchain_community.chat_models.vertexai")
    fake_chat_module.ChatVertexAI = _StubVertexAI
    sys.modules["langchain_community.chat_models.vertexai"] = fake_chat_module
    logger.debug("Stubbed langchain_community.vertexai for ragas import compatibility")


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# 预先 stub 一遍: 后面 import ragas 会立刻触发 langchain_community.vertexai.
_stub_langchain_community_vertexai()


# 复用 tests/_fakes.py 的假实现, 与 test_integration_retrieval.py 共享同一套 fake,
# 免得两边各维护一份、日久漂移.
from tests._fakes import (  # noqa: E402
    FakeBM25Store,
    FakeLLM,
    FakeRagasEmbeddings,
    FakeRagasJudge,
    FakeReranker,
    FakeVectorStore,
    tokenize as _tokenize,  # noqa: F401  (留作备用)
)


# 以下 chunks 与 data/samples/产品手册.md 内容对应, 供 fake 检索链路使用,
# 使 eval 在不依赖 chromadb / rank_bm25 的前提下获得接近真实链路的表现.
FAKE_DOCS_BY_KB: dict[str, list[dict]] = {
    "default": [
        {"id": "c1", "document": "本产品自购买日起享受一年免费保修。保修范围包括非人为损坏的硬件故障。", "metadata": {"source": "产品手册-售后"}},
        {"id": "c2", "document": "用户可在 7 天内无理由退货，15 天内因质量问题可换货。", "metadata": {"source": "产品手册-售后"}},
        {"id": "c3", "document": "登录官网 → 我的订单 → 申请售后 → 提交故障描述与图片。", "metadata": {"source": "产品手册-售后流程"}},
        {"id": "c4", "document": "支持微信、支付宝、银联信用卡、企业月结。", "metadata": {"source": "产品手册-支付"}},
        {"id": "c5", "document": "一线城市 24h 达，二线城市 48h 达，偏远地区 3-5 个工作日。", "metadata": {"source": "产品手册-配送"}},
        {"id": "c6", "document": "发票需在订单完成后 30 天内申请开具，逾期不再补开。", "metadata": {"source": "产品手册-发票"}},
    ],
}


def install_fakes() -> None:
    """将 Embedding / Vector Store / BM25 / Reranker / LLM 全部替换为 fake, 使检索/重排/生成均可运行.

    顺序不可乱: 设环境变量 -> 清 lru_cache -> 令 retriever 重新实例化 -> monkey-patch HybridRetriever.
    """
    import os
    from app.rag import embedding as emb_mod
    from app.rag import vector_store as vs_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.agents import llm_router as lr_mod

    os.environ["PILKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()

    # 将 Chroma / BM25 / CrossEncoderReranker 全数短路, 摆脱外部库依赖
    vs_mod.ChromaStore = FakeVectorStore
    bm_mod.BM25Store = FakeBM25Store
    rk_mod.CrossEncoderReranker = FakeReranker

    # HybridRetriever.retrieve 仍走真实逻辑 (FAKE_DOCS_BY_KB 已备好数据),
    # 在 __init__ 阶段把 FAKE_DOCS 灌入 vector / bm25.
    def _fake_init(self, kb_id: str = "default"):
        self.kb_id = kb_id
        # Fake store 构造时灌入 FAKE_DOCS; Vector 依赖 embedding, 这里直接
        # 给 c-dim 的 hash vector, 与 FakeEmbeddingService._vec 对齐.
        from tests._fakes import FakeEmbeddingService as _FES
        _emb = _FES()
        emb = _emb.embed([c["document"] for c in FAKE_DOCS_BY_KB[kb_id]])
        self.vector_store = FakeVectorStore(collection_name=f"kb_{kb_id}")
        self.vector_store.add(
            ids=[c["id"] for c in FAKE_DOCS_BY_KB[kb_id]],
            documents=[c["document"] for c in FAKE_DOCS_BY_KB[kb_id]],
            embeddings=emb,
            metadatas=[c["metadata"] for c in FAKE_DOCS_BY_KB[kb_id]],
        )
        self.bm25_store = FakeBM25Store(persist_path=f"./data/bm25_{kb_id}.pkl")
        self.bm25_store.add(
            ids=[c["id"] for c in FAKE_DOCS_BY_KB[kb_id]],
            documents=[c["document"] for c in FAKE_DOCS_BY_KB[kb_id]],
            metadatas=[c["metadata"] for c in FAKE_DOCS_BY_KB[kb_id]],
        )
        self.embedding = _emb

    rt_mod.HybridRetriever.__init__ = _fake_init

    lr_mod.reset_llm_cache()
    lr_mod.get_llm = lambda: FakeLLM()
    logger.info("Fake mode installed: embedding/vector/bm25/rerank/llm all stubbed")





def _run_records(records: list[dict], kb: str, mode: str, llm_mode: str) -> dict:
    """ragas 0.1.10 的 evaluate() 内部使用 asyncio.run, 无法在 async 上下文中执行, 故此处为同步函数.

    每个样本同步调用 run_rag (其内部没有 asyncio, 顶层虽是 await 但实际兼容 sync).
    """
    from app.rag.pipeline import run_rag
    import asyncio as _asyncio

    questions, answers, contexts, ground_truths = [], [], [], []
    per_sample = []

    for i, r in enumerate(records, 1):
        q = r["question"]
        gt = r.get("ground_truth", "")
        ref_ctx = r.get("reference_context") or []

        try:
            result = _asyncio.run(
                run_rag(question=q, kb_id=kb, top_k=3, use_agent=(mode == "agent"))
            )
        except Exception as e:
            logger.warning(f"RAG failed for q='{q[:30]}...': {e}; skipping.")
            continue

        questions.append(q)
        answers.append(result["answer"])
        contexts.append(result["contexts"])
        # ragas 0.1.10 要求 ground_truth 为 str, 不接受 list[str]
        gt_str = gt if isinstance(gt, str) else (gt[0] if gt else "")
        ground_truths.append(gt_str)
        per_sample.append(
            {
                "question": q,
                "answer": result["answer"],
                "contexts": result["contexts"],
                "ground_truth": gt_str,
                "ref_context_for_inspection": ref_ctx,
            }
        )
        logger.info(f"[{i}/{len(records)}] {q[:40]} -> {result['answer'][:60]}")

    from datasets import Dataset
    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    eval_kwargs = {
        "metrics": [faithfulness, answer_relevancy, context_precision, context_recall],
    }
    if llm_mode == "fake_llm":
        # 指标自行运行, 不让 RAGAS 默认去 OpenAI 拉 chat model.
        # 1) 把 llm + embeddings 直接传给 evaluate().
        # 2) 并 monkey-patch ragas.llms.base.llm_factory 与 ragas.embeddings.factory,
        #    以防某个 metric 内部 None 检查时再度触发远端初始化.
        import ragas.llms.base as ragas_llm_base
        import ragas.embeddings as ragas_emb_base
        ragas_llm_base.llm_factory = lambda *a, **kw: FakeRagasJudge()
        if hasattr(ragas_emb_base, "embedding_factory"):
            ragas_emb_base.embedding_factory = lambda *a, **kw: FakeRagasEmbeddings()
        # 双保险: 逐个 metric 实例也做覆盖
        for m in (faithfulness, answer_relevancy, context_precision, context_recall):
            try:
                m.llm = FakeRagasJudge()
            except Exception:
                pass
            try:
                m.embeddings = FakeRagasEmbeddings()
            except Exception:
                pass
        eval_kwargs["llm"] = FakeRagasJudge()
        eval_kwargs["embeddings"] = FakeRagasEmbeddings()
    scores = evaluate(ds, **eval_kwargs)
    try:
        score_dict = dict(scores)
    except Exception:
        score_dict = {
            m.name: float(scores[m.name])
            for m in (faithfulness, answer_relevancy, context_precision, context_recall)
        }

    return {"scores": score_dict, "samples": per_sample, "n": len(per_sample)}


def _fmt(val) -> str:
    try:
        return f"{float(val):.4f}"
    except Exception:
        return str(val)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kb", default="default")
    parser.add_argument(
        "--dataset", type=Path,
        default=Path(__file__).parents[2] / "data" / "eval" / "qa.jsonl",
    )
    parser.add_argument("--mode", choices=["rag", "agent"], default="rag")
    parser.add_argument(
        "--llm-mode", choices=["fake_llm", "real_llm"], default="fake_llm",
        help="fake_llm 以内置 fake 跑通评估脚本; real_llm 使用真实 provider (需 .env 配好 API key)",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--baseline-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.dataset.exists():
        raise SystemExit(f"数据集文件不存在: {args.dataset}")

    if args.llm_mode == "fake_llm":
        install_fakes()
    else:
        logger.info("Real LLM mode: using settings.llm_provider (需在 .env 配置 API key)")

    records = load_jsonl(args.dataset)
    print(f"[PilKB RAGAS] llm_mode={args.llm_mode} | dataset={args.dataset} | n={len(records)}")

    report = _run_records(records, args.kb, args.mode, args.llm_mode)

    print("\n=== RAGAS Scores ===")
    for name in ("faithfulness", "answer_relevancy", "context_precision", "context_recall"):
        val = report["scores"].get(name)
        print(f"  {name:>20s}: {_fmt(val)}")
    print(f"  n = {report['n']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        # 同样做 NaN -> 0 的清理
        scores_clean: dict = {}
        for k, v in report["scores"].items():
            try:
                fv = float(v)
                if fv != fv:
                    scores_clean[k] = 0.0
                else:
                    scores_clean[k] = fv
            except Exception:
                scores_clean[k] = 0.0
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "kb_id": args.kb,
            "mode": args.mode,
            "llm_mode": args.llm_mode,
            "dataset": str(args.dataset),
            "scores": scores_clean,
            "samples": report["samples"],
            "note": (
                "fake_llm 路径: 供 CI / 沙箱验证脚本可跑通, 数值不代表真实模型水平."
                if args.llm_mode == "fake_llm"
                else "real_llm 路径: 凭真实 API key 跑出的分数, 可充当 baseline."
            ),
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已落盘: {args.out}")

    if args.baseline_out:
        args.baseline_out.parent.mkdir(parents=True, exist_ok=True)
        # NaN 归 0: ragas 在个别 metric 的 parser 全部失败时会给出 nan, 而 JSON 标准
        # 不允许 NaN, 这里统一置 0, 并在 baseline 中记录哪些 metric 发生了退化.
        cleaned = {}
        nan_metrics = []
        for k, v in report["scores"].items():
            try:
                fv = float(v)
                if fv != fv:  # NaN 判断
                    nan_metrics.append(k)
                    cleaned[k] = 0.0
                else:
                    cleaned[k] = fv
            except Exception:
                nan_metrics.append(k)
                cleaned[k] = 0.0
        compact = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "llm_mode": args.llm_mode,
            "kb_id": args.kb,
            "mode": args.mode,
            "n_samples": report["n"],
            "scores": cleaned,
            "nan_metrics": nan_metrics,
            "note": (
                "fake LLM 产出的分数, 仅用于脚本通断校验, 不充当 baseline."
                "nan_metrics 记录退化的指标 (parser 全部失败, 已置 0)."
                if args.llm_mode == "fake_llm"
                else "请将此处数值作为 baseline 提交, 供后续优化对比."
            ),
        }
        args.baseline_out.write_text(
            json.dumps(compact, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"baseline 已落盘: {args.baseline_out}")


if __name__ == "__main__":
    main()