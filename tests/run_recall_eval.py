"""离线检索评估: 在评估语料上量化 Recall@K / Precision@K / MRR@K, 不需要 LLM.

与 run_ragas_eval.py 的分工:
  - run_ragas_eval.py 评**生成质量** (faithfulness / answer_relevancy 等), 必须调 LLM.
  - 本脚本评**检索质量** (Recall / Precision / MRR), 完全不碰 LLM —— 既能跑在没 API key
    的环境, 指标也不受生成模型波动影响. 调切分参数 / 检索策略时看这个.

为什么要单独的语料: 仓库原有的 data/samples 文档都太小 (几百字符), 在
adaptive_chunk_size=5500 下只能切出 1 个 chunk, Recall@K 恒等于 1.0, 测不出任何
差异. data/eval/corpus/ 下的语料专为评估准备: 章节间有**相似但不同**的条款
(三条产品线的保修期、四个地区的配送时效、六档折扣), 检索必须真正分得清才能得高分.

数据集格式: {"question": ..., "reference_context": ["答案所在的原文片段", ...]}
  reference_context 为空表示"答案不在语料内" —— 这类样本不计入召回统计,
  否则会把指标刷虚高.

用法:
  # 单配置
  python -m tests.run_recall_eval --out data/eval/recall_baseline.json
  # 对比多种切分粒度 (回答"切多细对召回最好")
  python -m tests.run_recall_eval --compare

注意: 本脚本需要真实的 embedding 模型与向量库, 请在完整依赖环境 (Docker) 里跑.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

from loguru import logger

# 项目根 / data 目录 (backend/ 拍平到根后, tests 的上级即项目根)
_BACKEND = Path(__file__).resolve().parents[1]
_DATA = _BACKEND / "data"


def _load_corpus(path: Path) -> str:
    """读语料并走与上传链路相同的清洗, 保证评估文本与线上入库文本一致."""
    from app.rag.cleaner import DocumentCleaner
    from app.core.config import settings

    raw = path.read_text(encoding="utf-8")
    return DocumentCleaner(enabled=settings.cleaner_enabled).clean(raw)


def _build_index(corpus_path: Path, kb_id: str, chunk_kwargs: dict) -> tuple[int, list[str]]:
    """把语料切分并灌入指定 kb, 返回 (chunk 数, 每个 chunk 的文本).

    kb_id 由调用方保证是临时的, 跑完清理.
    """
    from app.rag.chunker import TextChunker
    from app.rag.retriever import get_retriever

    text = _load_corpus(corpus_path)
    chunks = TextChunker(**chunk_kwargs).split(text, doc_id=f"eval{uuid.uuid4().hex[:8]}")
    if not chunks:
        raise SystemExit(f"语料切分后为空: {corpus_path}")

    retriever = get_retriever(kb_id)
    ids = [c.metadata["chunk_id"] for c in chunks]
    docs = [c.text for c in chunks]
    metas = [dict(c.metadata, filename=corpus_path.name) for c in chunks]
    retriever.add_documents(ids=ids, documents=docs, metadatas=metas)
    logger.info(f"已入库 {len(chunks)} 个 chunk -> kb={kb_id}")
    return len(chunks), docs


def _cleanup(kb_id: str) -> None:
    """删掉临时 kb 的向量 collection 与 BM25 文件, 不给真实库留垃圾.

    评估用的是独立 kb_id (evaltmp_*), 与真实知识库天然隔离; 这里只负责把
    这一次跑出来的残留清掉. 任何一步失败都只告警 —— 清理失败不该让评估结果作废.
    """
    from app.core.config import settings
    from app.rag.retriever import get_retriever, reset_retriever_cache

    try:
        vs = get_retriever(kb_id).vector_store
        # Chroma 后端: 直接删 collection
        client = getattr(vs, "_client", None)
        if client is not None:
            try:
                client.delete_collection(f"kb_{kb_id}")
            except Exception as e:  # noqa: BLE001
                logger.warning(f"删除临时 collection 失败: {e}")
        # NumPy 后端: collection 名即文件名
        persist = getattr(vs, "persist_path", None) or getattr(vs, "_path", None)
        if persist and Path(persist).is_file():
            Path(persist).unlink()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"清理临时向量库失败: {e}")

    try:
        p = Path(settings.bm25_persist_dir) / f"bm25_{kb_id}.pkl"
        if p.is_file():
            p.unlink()
    except Exception as e:  # noqa: BLE001
        logger.warning(f"清理临时 BM25 文件失败: {e}")

    reset_retriever_cache()


def _run_one(
    corpus_path: Path,
    dataset: list[dict],
    chunk_kwargs: dict,
    label: str,
    top_k: int,
) -> dict:
    """跑单个切分配置: 建库 -> 逐条检索 -> 算 Recall@K -> 清理."""
    from app.rag.evaluate import ContextRecallEvaluator
    from app.rag.retriever import get_retriever

    kb_id = f"evaltmp_{uuid.uuid4().hex[:6]}"
    n_chunks, _ = _build_index(corpus_path, kb_id, chunk_kwargs)
    retriever = get_retriever(kb_id)

    try:
        evaluator = ContextRecallEvaluator()
        # 用 retriever.retrieve 而不是 pipeline: 隔离检索本身的效果,
        # 不让查询改写 / HyDE 的 LLM 波动混进"切分粒度"的对比里.
        report = evaluator.evaluate(dataset, lambda q: retriever.retrieve(q, top_k=top_k))
    finally:
        _cleanup(kb_id)

    report["label"] = label
    report["chunk_count"] = n_chunks
    report["chunk_kwargs"] = {k: v for k, v in chunk_kwargs.items()}
    return report


_COLS = ("R@1", "R@3", "R@5", "R@10", "P@1", "P@3", "P@5", "P@10")


def _print_table_header() -> None:
    # 行前缀是 "  " + 26 宽 label + " chunks=" + 4 宽数字 + 两空格, 表头对齐同一布局
    head = "".join(f"{c:>7}" for c in _COLS) + f"{'MRR@10':>9}"
    print(f"  {'':<26} {'chunks':>11}  {head}")


def _fmt_row(label: str, r: dict) -> str:
    cells = [f"{r[f][k]:.3f}" for f in ("recall", "precision") for k in ("@1", "@3", "@5", "@10")]
    body = "".join(f"{c:>7}" for c in cells)
    return f"  {label:<26} chunks={r['chunk_count']:>4}  {body}  {r['mrr']['@10']:>7.3f}"


def _series(report: dict, family: str, k: str) -> list[float]:
    """抽某指标的逐题数组 (无答案样本不计入), 供配对 bootstrap 用."""
    return [row[family][k] for row in report["per_question"] if not row.get("no_answer")]


def _print_paired_bootstrap(results: list[dict]) -> None:
    """逐指标: 其余配置 vs 该指标最优, 配对 bootstrap 给均值差的 95% CI.

    几十条样本上 0.05 量级的分差很可能只是噪音, CI 跨不跨 0 才是判断标准.
    注意 vs-best 有 winner's curse: "最优"本身是从同一批样本里挑出来的,
    这里的 CI 系统性偏乐观 —— 打 * 的结论可信, 没打 * 未必真无差异.
    """
    from app.rag.evaluate import paired_bootstrap_ci

    print("\n=== 配对 bootstrap: 其余配置 vs 各指标最优 (95% CI, * = CI 不跨 0) ===")
    for family, k in (("recall", "@5"), ("precision", "@5"), ("mrr", "@10")):
        best = max(results, key=lambda r: r[family][k])
        print(f"  {family}{k:<3} 最优: {best['label']} ({best[family][k]:.3f})")
        for r in results:
            if r is best:
                continue
            diff, lo, hi = paired_bootstrap_ci(_series(best, family, k), _series(r, family, k))
            sig = "*" if (lo > 0 or hi < 0) else " "
            # Δ 口径 = 最优 - 该配置, 正值即最优确实更好
            print(f"    vs {r['label']:<26} Δ={diff:+.3f}  CI[{lo:+.3f}, {hi:+.3f}] {sig}")


# 5500 是调优前的历史值, 保留作为反例: 它切出的 chunk 少、看似"召回率高",
# 但 Recall@1 反而最低, 且超出 bge-small-zh 的 512 token 上下文窗口.
CONFIGS = [
    ("adaptive(5500) 历史值", dict(adaptive=True, adaptive_chunk_size=5500, adaptive_short_doc_chars=500)),
    ("chunk_size=500", dict(chunk_size=500, chunk_overlap=80)),
    ("chunk_size=300", dict(chunk_size=300, chunk_overlap=60)),
    ("chunk_size=800", dict(chunk_size=800, chunk_overlap=120)),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=_DATA / "eval" / "corpus" / "企业知识库演示语料.md")
    ap.add_argument("--dataset", type=Path, default=_DATA / "eval" / "recall_dataset.jsonl")
    ap.add_argument("--top-k", type=int, default=10, help="每条问题召回的候选数 (需 >= 最大 K)")
    ap.add_argument("--out", type=Path, default=None, help="把结果写到 JSON")
    ap.add_argument("--compare", action="store_true", help="对比 CONFIGS 里的多种切分粒度")
    ap.add_argument("--label", default="custom", help="单配置模式下的结果标签")
    ap.add_argument("--chunk-size", type=int, default=None)
    ap.add_argument("--adaptive", action="store_true")
    args = ap.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"语料不存在: {args.corpus}")
    if not args.dataset.exists():
        raise SystemExit(f"数据集不存在: {args.dataset}")

    dataset = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(f"[PilKB Recall] 语料={args.corpus.name} | 问题={len(dataset)} 条 | top_k={args.top_k}")

    if args.compare:
        configs = CONFIGS
    else:
        if args.chunk_size:
            kw = dict(chunk_size=args.chunk_size, chunk_overlap=max(0, args.chunk_size // 6))
        elif args.adaptive:
            # 强制走自适应切分, 粒度取**当前配置** (便于验证改完 adaptive_chunk_size 的效果)
            from app.core.config import settings

            kw = dict(
                adaptive=True,
                adaptive_chunk_size=settings.adaptive_chunk_size,
                adaptive_short_doc_chars=settings.adaptive_short_doc_chars,
            )
        else:
            from app.core.config import settings

            kw = dict(
                adaptive=settings.adaptive_chunking_enabled,
                adaptive_chunk_size=settings.adaptive_chunk_size,
                adaptive_short_doc_chars=settings.adaptive_short_doc_chars,
            )
        configs = [(args.label, kw)]

    results = []
    for label, kw in configs:
        logger.info(f"--- {label} ---")
        results.append(_run_one(args.corpus, dataset, kw, label, args.top_k))

    print("\n=== Recall@K / Precision@K / MRR@10 (文本口径) ===")
    _print_table_header()
    for r in results:
        print(_fmt_row(r["label"], r))
    print(f"\n  参与统计 {results[0]['n']} 条; 无答案样本 {results[0]['n_no_answer']} 条 (不计入)")

    if len(results) > 1:
        best = max(results, key=lambda r: r["recall"]["@5"])
        print(f"\n  Recall@5 最高: {best['label']} ({best['recall']['@5']:.3f})")
        _print_paired_bootstrap(results)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "corpus": str(args.corpus),
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "results": results,
            "note": (
                "Recall/Precision/MRR 按 reference_context 文本匹配计算 (阈值 0.7), "
                "不依赖 chunk_id, 因此重新入库后评估集依然可用. "
                "数字越高越好, Recall 的随机基线约为 1/chunk_count."
            ),
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
