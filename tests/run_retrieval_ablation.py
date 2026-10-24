"""检索链路消融: 单路单query vs 双路多query (Recall/Precision/MRR @3/5/10 + 检索延迟).

指标与口径:
  - Recall / Precision / MRR @3/@5/@10: 相关性判定是确定性的文本匹配
    (reference_context 覆盖率 >= 0.7, 与 run_recall_eval 同口径), 不需要 LLM.
    与 run_recall_eval 的分工: 那个走 retriever.retrieve 隔离切分参数; 本脚本走
    pipeline.retrieve_and_rerank —— 与线上聊天**同一条链路** (改写 -> 多 query
    并发检索 -> RRF -> 父召回 -> 重排), 测的是部署形态的检索质量.
  - 检索延迟: 收到 query 到 "检索+重排完成" 的墙钟, 逐条测量取 mean/p50/p95.
    不含 LLM 改写耗时 —— 多 query 回放自 data/eval 缓存, 整个脚本不调改写 LLM
    (生产里改写是检索前的独立 LLM 步骤, 要量单独量, 别混进检索口径).

数据 (全部现成, 不现场生成, 不烧 LLM):
  - 问题与相关性锚点: data/eval/recall_dataset.jsonl (130 条, 119 条有
    reference_context; 无答案样本不计入指标但照测延迟).
  - query 改写: data/eval/hyde_gen_cache.json, {问题: {"rewrite": [原句+3 变体],
    "hyde": 假设答案}}, 与数据集 62 条全对齐. --with-hyde 可把 HyDE 加进多 query.
  - 知识库: 直接查 --kb 指向的现成 kb (语料与数据集需配套), 不建临时库.
    注意: 查询侧 embedding 后端/模型必须与当时入库的一致, 否则向量路失真
    (fake embedding 跑出来只有链路通断意义, 数字不作数).

相关性判定两种口径 (--judge):
  text (默认): reference_context 文本覆盖 >= 0.7, 确定性, 与历史轮次可比.
  llm: LLM 逐对判断 (question, chunk) 是否相关 (走 settings.llm_provider, verdict
       缓存到 data/eval/llm_judge_cache.json: 两配置重叠的候选只判一次, 跨运行复用).
       口径差异: llm 模式下 Recall@K 是 hit-rate (top-k 内存在相关 chunk), 不锚定
       reference 片段; 单 reference 问题与文本口径基本等价.

用法 (完整依赖环境: 真实 embedding + CrossEncoder; --judge llm 还需 LLM key):
  python -m tests.run_retrieval_ablation --kb kb-eval --out data/eval/ablation_retrieval.json
  # LLM 判定 + 随机抽样 60 条 (seed 固定, 抽样可复现):
  python -m tests.run_retrieval_ablation --kb kb-eval --judge llm --sample 60 \
      --out data/eval/ablation_retrieval_llm.json
  # 变体: --single-path bm25 | --with-hyde | -n 20 (先小跑验证)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

from loguru import logger

from tests import _ablation_common as ab

_DATA = Path(__file__).resolve().parents[1] / "data"
KS = ("@3", "@5", "@10")


# ---------------------------------------------------------------------------
# LLM 相关性判定 (--judge llm)
#
# 文本口径的盲区是 paraphrase 与模板句; LLM 判定换掉的是"命中"这一步, 指标定义
# 不变. verdict 按 (question, chunk_id) 缓存并落盘 —— chunk 相关性与配置无关,
# 两配置重叠的候选只判一次, 跨运行也不再付费.
# ---------------------------------------------------------------------------
_JUDGE_PROMPT = (
    "你是检索质量评估器。判断【检索片段】是否包含回答【用户问题】所需的相关信息：\n"
    "只覆盖答案的一部分也算相关；与问题主题无关的片段算不相关。\n"
    "只输出一个字：是 或 否。\n\n"
    "【用户问题】{q}\n\n【检索片段】{chunk}"
)


def _parse_verdict(resp: str) -> bool | None:
    s = (resp or "").strip().lower()
    if not s:
        return None
    if s[0] in ("是", "y", "1"):
        return True
    if s[0] in ("否", "n", "0"):
        return False
    if "yes" in s[:10]:
        return True
    if "no" in s[:10]:
        return False
    return None


def _judge_pairs(
    pairs: list[tuple[str, dict]], cache: dict, max_workers: int = 6
) -> None:
    """LLM 判断 (question, chunk) 相关性, 写入 cache; 已有 verdict 的直接复用."""
    from app.agents import llm_router

    todo = [(q, c) for q, c in pairs if (q, c["id"]) not in cache]
    if not todo:
        logger.info(f"LLM judge: {len(pairs)} 对全部命中缓存, 0 次调用")
        return
    logger.info(f"LLM judge: 待判 {len(todo)} 对 (缓存命中 {len(pairs) - len(todo)})")
    llm = llm_router.get_llm_text_callable()
    n_bad = 0

    def one(item):
        q, c = item
        try:
            resp = llm(_JUDGE_PROMPT.format(q=q, chunk=(c.get("document") or "")[:800]))
        except Exception as e:  # noqa: BLE001 - 单对失败不炸整轮
            logger.warning(f"judge 调用失败 ({q[:20]}...): {e}")
            return q, c["id"], False, True
        v = _parse_verdict(resp)
        if v is None:
            return q, c["id"], False, True
        return q, c["id"], v, False

    # 并发压 6: judge 是独立短请求, 深并发会撞 provider 限流 (ragas 那次的教训)
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for q, cid, v, bad in ex.map(one, todo):
            cache[(q, cid)] = v
            n_bad += bad
    if n_bad:
        # 解析/调用失败按"不相关"计: 对两个配置同等对待, 对比结论仍然公平
        logger.warning(f"LLM judge: {n_bad} 对解析失败或调用失败, 已按不相关计")


def _llm_eval(dataset: list[dict], retrievals: dict, cache: dict) -> dict:
    """先判完所有 (question, chunk) 对, 再从 verdict 推 R(hit-rate)/P/MRR.

    行结构与文本口径完全一致, 下游表格 / bootstrap / JSON 落盘零改动.
    """
    pairs = [
        (r["question"], c) for r in dataset for c in retrievals.get(r["question"], [])
    ]
    _judge_pairs(pairs, cache)

    ks = (3, 5, 10)
    sums = {f: {k: 0.0 for k in ks} for f in ("recall", "precision", "mrr")}
    per_question = []
    for r in dataset:
        q = r["question"]
        flags = [bool(cache.get((q, c["id"]))) for c in retrievals.get(q, [])]
        first = next((i + 1 for i, f in enumerate(flags) if f), None)
        row: dict = {"question": q, "recall": {}, "precision": {}, "mrr": {}}
        for k in ks:
            top = flags[:k]
            vals = {
                "recall": 1.0 if any(top) else 0.0,
                "precision": (sum(top) / min(k, len(top))) if top else 0.0,
                "mrr": (1.0 / first) if first and first <= k else 0.0,
            }
            for f, v in vals.items():
                sums[f][k] += v
                row[f][f"@{k}"] = round(v, 4)
        per_question.append(row)
    n = len(dataset) or 1
    return {
        "n": len(dataset),
        "n_no_answer": 0,
        **{
            f: {f"@{k}": round(sums[f][k] / n, 4) for k in ks}
            for f in ("recall", "precision", "mrr")
        },
        "per_question": per_question,
    }


def _run_config(
    name: str,
    cfg: dict,
    kb: str,
    dataset: list[dict],
    judge: str = "text",
    judge_cache: dict | None = None,
) -> dict:
    """跑一个配置: 切 settings + 单路补丁 -> 逐题测检索 (指标+延迟) -> 恢复现场."""
    from app.rag.evaluate import ContextRecallEvaluator
    from app.rag.pipeline import retrieve_and_rerank

    saved_cfg = ab.apply_settings(cfg["settings"])
    saved_stores = ab.patch_single_path(kb, cfg["path"]) if cfg["path"] != "hybrid" else []
    try:
        # 预热 1 条 (不计入): embedding 与 CrossEncoder 都是懒加载, 不预热的话
        # 首条延迟会被模型加载拖高数秒, 均值直接失真.
        asyncio.run(retrieve_and_rerank(dataset[0]["question"], kb_id=kb))
        latencies: list[float] = []

        if judge == "llm":
            # 先取回候选再统一判: ks 最大是 10, 第 10 名之后进不了任何指标,
            # 只判 top-10 省掉一半以上的 judge 调用.
            retrievals: dict[str, list[dict]] = {}
            for r in dataset:
                q = r["question"]
                t0 = time.perf_counter()
                out = asyncio.run(retrieve_and_rerank(q, kb_id=kb))
                latencies.append((time.perf_counter() - t0) * 1000)
                retrievals[q] = out[:10]
            # 不能写 judge_cache or {}: 空 dict 是 falsy, or 会换出一个临时 dict,
            # 两配置各判各的、跨配置去重和落盘缓存全都悄悄失效 (实测暴露).
            report = _llm_eval(
                dataset, retrievals, judge_cache if judge_cache is not None else {}
            )
        else:

            def retrieve(q: str) -> list[dict]:
                t0 = time.perf_counter()
                out = asyncio.run(retrieve_and_rerank(q, kb_id=kb))
                latencies.append((time.perf_counter() - t0) * 1000)
                return out

            report = ContextRecallEvaluator(ks=(3, 5, 10)).evaluate(dataset, retrieve)
    finally:
        ab.restore_stores(saved_stores)
        ab.restore_settings(saved_cfg)

    report["per_question_latency_ms"] = [round(x, 1) for x in latencies]
    report["latency_ms"] = ab.latency_stats(latencies)
    logger.info(
        f"[{name}] R@5={report['recall']['@5']} P@5={report['precision']['@5']} "
        f"MRR@5={report['mrr']['@5']} | 延迟 mean={report['latency_ms']['mean']}ms"
    )
    return report


def _fmt_row(name: str, r: dict) -> str:
    cells = "".join(f"{r[f][k]:>8.3f}" for f in ("recall", "precision", "mrr") for k in KS)
    lat = r["latency_ms"]
    return f"  {name:<18}{cells}  {lat['mean']:>10.0f}{lat['p50']:>8.0f}{lat['p95']:>8.0f}"


def _print_bootstrap(results: dict[str, dict]) -> None:
    """两配置逐题差分的 95% CI: 几十条样本上 0.05 量级的分差必须过显著性才算数."""
    if len(results) != 2:
        return
    from app.rag.evaluate import paired_bootstrap_ci

    (na, ra), (nb, rb) = results.items()
    # 两配置跑同一批问题, 按问题对齐 (无答案行两边都无指标, 自然被跳过).
    qa = {r["question"]: r for r in ra["per_question"] if not r.get("no_answer")}
    qb = {r["question"]: r for r in rb["per_question"] if not r.get("no_answer")}
    common = [q for q in qa if q in qb]
    print(f"\n=== 配对 bootstrap (Δ = {nb} - {na}, n={len(common)}, 95% CI, * = 不跨 0) ===")
    for family in ("recall", "precision", "mrr"):
        for k in KS:
            sa = [qa[q][family][k] for q in common]
            sb = [qb[q][family][k] for q in common]
            # 标签口径是 Δ = nb - na, 参数顺序必须与之对应 (sa, sb 算的是 na - nb,
            # 传反了数字对但符号全反, 2026-10-11 第二轮消融实跑时暴露).
            diff, lo, hi = paired_bootstrap_ci(sb, sa)
            sig = "*" if (lo > 0 or hi < 0) else " "
            print(f"  {family}{k:<3} Δ={diff:+.4f}  CI[{lo:+.4f}, {hi:+.4f}] {sig}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb-eval", help="评测语料所在的 kb (需先建库并上传 data/eval/corpus/)")
    ap.add_argument("--dataset", type=Path, default=_DATA / "eval" / "recall_dataset.jsonl")
    ap.add_argument("--gen-cache", type=Path, default=_DATA / "eval" / "hyde_gen_cache.json")
    ap.add_argument("--single-path", choices=["vector", "bm25"], default="vector")
    ap.add_argument("--with-hyde", action="store_true", help="多 query 追加缓存的 HyDE 假设答案")
    ap.add_argument("-n", "--num", type=int, default=None, help="只跑前 n 条 (小跑验证用)")
    ap.add_argument("--judge", choices=["text", "llm"], default="text", help="相关性判定: 文本覆盖 or LLM")
    ap.add_argument("--sample", type=int, default=None, help="从有参考的问题中随机抽 N 条")
    ap.add_argument("--seed", type=int, default=0, help="--sample 的随机种子 (固定可复现)")
    ap.add_argument("--judge-cache", type=Path, default=_DATA / "eval" / "llm_judge_cache.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dataset = ab.load_jsonl(args.dataset)
    if not dataset:
        raise SystemExit(f"数据集为空: {args.dataset}")
    if args.num:
        dataset = dataset[: args.num]
    if args.sample:
        # 只从有 reference 的池子抽: 无答案样本不进指标, 抽进来只会白测延迟
        scored = [r for r in dataset if r.get("reference_context")]
        dataset = random.Random(args.seed).sample(scored, min(args.sample, len(scored)))
        print(f"[抽样] seed={args.seed} -> {len(dataset)}/{len(scored)} 条 (有参考池)")
    gen_cache = ab.load_gen_cache(args.gen_cache)
    missing = [r["question"] for r in dataset if r["question"] not in gen_cache]
    if missing:
        # 不悄悄降级: 双路多query 缺改写缓存的那几题会退化为单 query, 必须显式告知.
        logger.warning(
            f"{len(missing)} 条问题没有缓存改写, 这些题的双路多query 将退化为单 query: "
            f"{missing[:3]}..."
        )

    n_ref = sum(1 for r in dataset if r.get("reference_context"))
    print(
        f"[检索消融] kb={args.kb} | 数据集={args.dataset.name} ({len(dataset)} 条, {n_ref} 条有参考) "
        f"| 改写缓存={args.gen_cache.name} ({len(gen_cache)} 条) | hyde={args.with_hyde}"
    )

    # LLM judge 的 verdict 缓存: 跨配置去重 + 跨运行复用 (key 含 \x1f 分隔符,
    # question 与 chunk_id 里都不会出现)
    judge_cache: dict = {}
    if args.judge == "llm" and args.judge_cache.exists():
        raw = json.loads(args.judge_cache.read_text(encoding="utf-8"))
        judge_cache = {tuple(k.split("\x1f")): bool(v) for k, v in raw.items()}
        print(f"[judge] 载入缓存 {len(judge_cache)} 条 verdict")

    # 回放装一次, 两个配置共用; single 配置靠 query_rewrite_enabled=False 自然走原句.
    ab.install_rewrite_replay(gen_cache)
    try:
        configs = ab.config_matrix(args.single_path, args.with_hyde)
        results = {
            name: _run_config(
                name, cfg, args.kb, dataset, judge=args.judge, judge_cache=judge_cache
            )
            for name, cfg in configs.items()
        }
    finally:
        ab.uninstall_rewrite_replay()

    if args.judge == "llm":
        args.judge_cache.parent.mkdir(parents=True, exist_ok=True)
        args.judge_cache.write_text(
            json.dumps({f"{q}\x1f{c}": v for (q, c), v in judge_cache.items()}),
            encoding="utf-8",
        )
        print(f"[judge] verdict 缓存已写入: {args.judge_cache} ({len(judge_cache)} 条)")

    head = "".join(f"{f + k:>8}" for f in ("R", "P", "MRR") for k in KS)
    print(f"\n=== Recall / Precision / MRR (@3/@5/@10) + 检索延迟 (ms) ===")
    print(f"  {'':<18}{head}{'mean_ms':>12}{'p50':>8}{'p95':>8}")
    for name, r in results.items():
        print(_fmt_row(name, r))
    first = next(iter(results.values()))
    print(f"\n  参与统计 {first['n']} 条; 无答案样本 {first['n_no_answer']} 条 (不计入指标, 延迟照测)")
    _print_bootstrap(results)

    if args.out:
        judge_note = (
            "相关性判定: LLM 逐对判断 (question, chunk); Recall@K 为 hit-rate 口径 "
            "(top-k 内存在相关 chunk), 不锚定 reference 片段."
            if args.judge == "llm"
            else "相关性判定: reference_context 文本覆盖 >= 0.7 (确定性, 无 LLM)."
        )
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "kb": args.kb,
            "dataset": str(args.dataset),
            "gen_cache": str(args.gen_cache),
            "with_hyde": args.with_hyde,
            "judge": args.judge,
            "sample": args.sample,
            "seed": args.seed,
            "results": {
                name: {
                    "config": configs[name],
                    "recall": r["recall"],
                    "precision": r["precision"],
                    "mrr": r["mrr"],
                    "latency_ms": r["latency_ms"],
                    # 与 per_question 同序 (数据集顺序), 一一对应
                    "per_question_latency_ms": r["per_question_latency_ms"],
                    "per_question": r["per_question"],
                }
                for name, r in results.items()
            },
            "note": (
                f"{judge_note} "
                "延迟口径: 收到 query -> 检索+重排完成, 不含 LLM 改写 (改写走缓存回放); "
                "single-single 的 path 字段是评估侧实例补丁 (非生产开关)."
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
