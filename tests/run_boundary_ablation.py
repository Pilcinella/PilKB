"""跨 chunk 边界合成题消融: 兄弟连坐 (parent_recall) off vs on.

动机与口径 (跑之前先读):
  - 现有消融 (run_retrieval_ablation / run_ragas_ablation) 的数据集几乎全是
    单 chunk 可答的短事实题, 测的是"答案 chunk 是否进 top-k" —— 对兄弟连坐
    天然不敏感 (2026-10-27 全量重跑已证实主链路指标无差异). 连坐的价值前提是
    **答案由切点两侧的事实合成**, 本脚本用一册专门锚定 kb-eval 真实 chunk 边界
    出题的数据集 (boundary_span_dataset.jsonl) 补上这个盲区. 每题关键词按切点
    两侧标注 side=A/B, 出题时已逐条验证关键词落在标注侧 chunk 的 overlap 窗口
    之外 (语料切分带 ~70 字 overlap, 切点句子两侧都有, 不避开就测不出差异).

两层指标:
  L1 检索层 (确定性, 无 LLM 噪声): target 两侧 chunk 是否都进入最终 top-k 上下文
     (run_rag 返回的 context_ids). 这是连坐机制的直接观测 —— off 时靠两块各自
     被独立召回, on 时弱匹配侧还可由兄弟连坐带入.
  L2 生成层: 答案 keyword coverage (口径同 run_agent_ablation: 去全部空白后子串
     匹配), 整体 + 分侧. 分侧才是判据: 连坐生效应主要体现为弱匹配侧 coverage
     上升, 而不是两侧同涨.

单一变量原则: 查询改写 / HyDE 一律关闭 (多 query 会放大无关差异, 且新题没有改写
缓存), 其余生产链路 (混合检索 → RRF → 连坐 → CrossEncoder 重排 → 生成) 原样保留.
三个配置:
  parent-off  基线: 候选连坐与拼接都关;
  parent-on   路线 A: 候选阶段连坐 (生产现状) —— 兄弟要与全库竞争 rerank 相关性;
  stitch-on   路线 B: 重排选定 top-k 后在**上下文组装阶段**拼 ±1 兄弟正文
              (small-to-big 模式; **评测侧补丁, 未进生产代码**, 见 install_context_stitch)
              —— 兄弟不经过竞争, 连续性直接兑现.

指标判读: L1 的 both_in (两侧 chunk 都在 context_ids) 对路线 B 天然失真 (兄弟不以
自身 id 进上下文), 所以主 L1 用 ctx_kw_B —— 弱侧关键词是否出现在**喂给 LLM 的正文**
里 (确定性, 不依赖生成); 事实在库内多处重复 (FAQ 回声等) 时 ctx_kw_B 会高估机制,
端到端结论仍以 L2 (cov_B) 为准.

用法:
  python -m tests.run_boundary_ablation --kb kb-eval -n 3   # 先小跑看机制
  python -m tests.run_boundary_ablation --kb kb-eval --out data/eval/boundary_ablation.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

from tests import _ablation_common as ab
from tests.run_agent_ablation import _norm

_DATA = Path(__file__).resolve().parents[1] / "data"


def _side_hits(norm_text: str, keywords: list[dict], side: str) -> float | None:
    """某一侧关键词在 (归一化后的) 文本中的出现比例. 该侧无关键词返回 None."""
    kws = [k["kw"] for k in keywords if k.get("side") == side]
    if not kws:
        return None
    return sum(1 for k in kws if _norm(k) in norm_text) / len(kws)


def _side_coverage(answer: str, keywords: list[dict], side: str) -> float | None:
    """某一侧关键词的命中率. 该侧没有关键词时返回 None (不参与均值)."""
    kws = [k["kw"] for k in keywords if k.get("side") == side]
    if not kws:
        return None
    a = _norm(answer)
    return sum(1 for k in kws if _norm(k) in a) / len(kws)


def _stitch_sources(retriever, sources: list[dict], window: int = 1) -> list[dict]:
    """路线 B 拼接 (**评测侧实现, 未进生产代码**): 给每个 source 正文拼 ±window 兄弟.

    引用溯源字段 (id/metadata/score) 原样保留, 仅 document 替换为 [c-window .. c+window]
    升序拼接文本; 已是 source 或已被前面窗口认领的兄弟不重复拼. 相邻 chunk 自带
    ~70 字 overlap, 拼接处小段重复对 LLM 无害, 不做文本级去重.
    """
    selected: dict[str, tuple[str, int]] = {}
    for c in sources:
        md = c.get("metadata", {}) or {}
        doc_id, idx = md.get("doc_id"), md.get("chunk_index")
        if doc_id and isinstance(idx, int) and c.get("id"):
            selected[c["id"]] = (doc_id, idx)
    if not selected:
        return list(sources)

    index = retriever._index_by_chunk_index({doc_id for doc_id, _ in selected.values()})
    used: set = set(selected)
    out: list[dict] = []
    for c in sources:
        cid = c.get("id")
        key = selected.get(cid) if cid else None
        if key is None:
            out.append(dict(c))
            continue
        bucket = index.get(key[0], {})
        parts: list[str] = []
        for i in range(key[1] - window, key[1] + window + 1):
            nb = bucket.get(i)
            if nb is None:
                continue
            if i != key[1] and nb["id"] in used:
                continue
            parts.append(nb["document"])
            used.add(nb["id"])
        nc = dict(c)
        if parts:
            nc["document"] = "\n".join(parts)
        out.append(nc)
    return out


def install_context_stitch(kb: str):
    """包一层 pipeline._retrieve_sources: 原检索不动, sources 拼接后重建 contexts.

    graph.py 走模块属性 pl._retrieve_sources 调用, 模块级替换可生效 (与
    _ablation_common.patch_single_path 同一评估侧补丁模式). 返回恢复函数, 必须调用.
    """
    import app.rag.pipeline as pl
    from app.rag.retriever import get_retriever

    orig = pl._retrieve_sources
    retriever = get_retriever(kb)

    async def _stitched(question, kb_id, top_k, trace, history=None):
        sources, _ = await orig(
            question, kb_id=kb_id, top_k=top_k, trace=trace, history=history
        )
        stitched = _stitch_sources(retriever, sources)
        contexts, _, _ = pl._format_sources(stitched)
        return stitched, contexts

    pl._retrieve_sources = _stitched
    return lambda: setattr(pl, "_retrieve_sources", orig)


async def _run_config(
    name: str,
    dataset: list[dict],
    kb: str,
    top_k: int,
    parent_on: bool,
    stitch: bool = False,
) -> list[dict]:
    from app.rag.pipeline import run_rag

    saved = ab.apply_settings(
        {
            "query_rewrite_enabled": False,
            "hyde_query_enabled": False,
            "parent_recall_enabled": parent_on,
        }
    )
    undo_stitch = install_context_stitch(kb) if stitch else None
    rows: list[dict] = []
    try:
        for q in dataset:
            ta, tb = q["target_chunks"]
            t0 = time.perf_counter()
            err = None
            out: dict = {}
            try:
                out = await run_rag(q["question"], kb_id=kb, top_k=top_k)
            except Exception as e:  # noqa: BLE001 - 单题失败记录后继续
                err = repr(e)
            dt = (time.perf_counter() - t0) * 1000
            answer = out.get("answer", "")
            ids = list(out.get("context_ids") or [])
            ctx_texts = list(out.get("contexts") or [])
            ctx_norm = _norm("\n".join(ctx_texts))
            kws = q["keywords"]
            row = {
                "question": q["question"],
                "boundary": q["boundary"],
                "in_A": ta in ids,
                "in_B": tb in ids,
                "both_in": ta in ids and tb in ids,
                "ctx_kw_A": _side_hits(ctx_norm, kws, "A"),
                "ctx_kw_B": _side_hits(ctx_norm, kws, "B"),
                "ctx_chars": sum(len(c) for c in ctx_texts),
                "cov_A": _side_coverage(answer, kws, "A"),
                "cov_B": _side_coverage(answer, kws, "B"),
                "latency_ms": round(dt, 1),
                "answer_head": answer[:100],
                "error": err,
            }
            # 整体 coverage 对齐 run_agent_ablation.keyword_coverage 口径:
            # 全部关键词的命中比例 (两侧合并计).
            all_kws = [k["kw"] for k in kws]
            a = _norm(answer)
            row["cov"] = (
                sum(1 for k in all_kws if _norm(k) in a) / len(all_kws) if all_kws else 0.0
            )
            rows.append(row)
            both = "AB" if row["both_in"] else ("A" if row["in_A"] else ("B" if row["in_B"] else "-"))
            ckb = row["ctx_kw_B"]
            flag = " ERR" if err else ""
            print(
                f"  [{name:<11}] hit={both:<2} ctxB={ckb if ckb is None else round(ckb, 2)} "
                f"cov={row['cov']:.2f} (B={row['cov_B'] if row['cov_B'] is None else round(row['cov_B'], 2)}) "
                f"ctx={row['ctx_chars']:>5}c {row['latency_ms']:>6.0f}ms  {q['question'][:24]}{flag}",
                flush=True,
            )
    finally:
        if undo_stitch:
            undo_stitch()
        ab.restore_settings(saved)
    return rows


def _mean(vals: list) -> float:
    xs = [v for v in vals if v is not None]
    return sum(xs) / len(xs) if xs else 0.0


def _paired(off: list[float], on: list[float]) -> tuple:
    from app.rag.evaluate import paired_bootstrap_ci

    return paired_bootstrap_ci(on, off)  # Δ = on - off


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb-eval")
    ap.add_argument(
        "--dataset", type=Path, default=_DATA / "eval" / "boundary_span_dataset.jsonl"
    )
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("-n", "--num", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dataset = ab.load_jsonl(args.dataset)
    if args.num:
        dataset = dataset[: args.num]
    print(
        f"[边界消融] kb={args.kb} | 数据集={args.dataset.name} ({len(dataset)} 题) "
        f"| top_k={args.top_k} | 改写/HyDE 强制关闭 | 三配置: off / 路线A(候选连坐) / 路线B(组装拼接)"
    )

    configs = [
        ("parent-off", False, False),  # 基线: 两种连坐都关
        ("parent-on", True, False),    # 路线 A: 候选阶段连坐 (生产现状)
        ("stitch-on", False, True),    # 路线 B: 上下文组装阶段拼接 (评测侧补丁)
    ]
    results = {}
    for name, parent_on, stitch in configs:
        results[name] = asyncio.run(
            _run_config(name, dataset, args.kb, args.top_k, parent_on, stitch)
        )

    off = results["parent-off"]
    print("\n=== 汇总 (三配置) ===")
    for name, _, _ in configs:
        rows = results[name]
        cells = "  ".join(
            f"{key}={_mean([r[key] for r in rows]):{fmt}}"
            for key, fmt in (
                ("both_in", ".1%"),
                ("ctx_kw_B", ".4f"),
                ("cov", ".4f"),
                ("cov_B", ".4f"),
                ("ctx_chars", ".0f"),
            )
        )
        print(f"  {name:<11}{cells}")

    print("\n=== 配对 bootstrap vs parent-off (Δ = 配置 - off, 95% CI, * = 不跨 0) ===")
    for name in ("parent-on", "stitch-on"):
        rows = results[name]
        print(f"  [{name}]")
        for key in ("ctx_kw_B", "cov", "cov_B"):
            sa = [float(r[key]) for r in off]
            sb = [float(r[key]) for r in rows]
            diff, lo, hi = _paired(sa, sb)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"    {key:<8} Δ={diff:+.4f}  CI[{lo:+.4f}, {hi:+.4f}] {sig}")

    # 子集: off 时弱侧事实未全部进上下文的题 —— 两种连坐机制的目标人群.
    # "补全"= ctx_kw_B 达到 1.0 (弱侧关键词全部出现在正文里), cov_B 是端到端兑现.
    weak = [i for i, r in enumerate(off) if (r["ctx_kw_B"] or 0.0) < 1.0]
    print(f"\n=== 子集: off 弱侧事实未全部进上下文 (n={len(weak)}) ===")
    if weak:
        print(f"  {'parent-off':<11} 子集 ctx_kw_B={_mean([off[i]['ctx_kw_B'] for i in weak]):.4f}"
              f"  cov_B={_mean([off[i]['cov_B'] for i in weak]):.4f}")
        for name in ("parent-on", "stitch-on"):
            rows = results[name]
            fixed = sum(1 for i in weak if (rows[i]["ctx_kw_B"] or 0.0) >= 1.0)
            print(
                f"  {name:<11} ctx_kw_B 补全 {fixed}/{len(weak)}"
                f"  子集 ctx_kw_B={_mean([rows[i]['ctx_kw_B'] for i in weak]):.4f}"
                f"  cov_B={_mean([rows[i]['cov_B'] for i in weak]):.4f}"
            )
    else:
        print("  (无 —— 库内事实冗余已让弱侧事实进正文, 数据集对两种机制都不敏感)")

    if args.out:
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "kb": args.kb,
            "dataset": str(args.dataset),
            "top_k": args.top_k,
            "configs": {
                name: {
                    "parent_recall_enabled": parent_on,
                    "context_stitch": stitch,
                    "query_rewrite_enabled": False,
                    "hyde_query_enabled": False,
                }
                for name, parent_on, stitch in configs
            },
            "summary": {
                name: {
                    "both_in": round(_mean([r["both_in"] for r in rows]), 4),
                    "ctx_kw_B": round(_mean([r["ctx_kw_B"] for r in rows]), 4),
                    "cov": round(_mean([r["cov"] for r in rows]), 4),
                    "cov_A": round(_mean([r["cov_A"] for r in rows]), 4),
                    "cov_B": round(_mean([r["cov_B"] for r in rows]), 4),
                    "ctx_chars": round(_mean([r["ctx_chars"] for r in rows]), 1),
                }
                for name, rows in results.items()
            },
            "results": results,
            "note": (
                "跨边界合成题: 答案需组合切点两侧事实 (关键词按 side 标注, 出题时已验证"
                "落在标注侧 overlap 窗口外). 三配置: parent-off 基线 / parent-on 路线A"
                "候选阶段连坐 / stitch-on 路线B 上下文组装阶段拼接 ±1 兄弟 (评测侧补丁,"
                "未进生产代码). 主 L1=ctx_kw_B (弱侧关键词是否在喂给 LLM 的正文里, "
                "确定性), L2=cov_B (端到端)."
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
