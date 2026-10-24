# -*- coding: utf-8 -*-
"""Agent 引擎消融: react (工具循环) vs router (一次性路由) vs rag (无 agent).

同一批问题逐个跑指定引擎, 输出口径可比的汇总与分层统计:
- 命中率: 参考关键词在答案中的覆盖率 (文本包含判定, 与 recall 消融的
  ``--judge text`` 同口径 —— 不追求语义等价, 追求可复现零成本);
- 步数: react 的实际工具轮数 (metadata.trace 里带 tool_calls 的 reason 条数);
- 延迟: 端到端 wall time 的 mean / p50 / p95;
- 分层: 按 type 分题型统计 —— 单跳题往往三引擎全对, 只看总分会把差异摊平;
- 显著性: 以 rag 为基线做配对 bootstrap, CI 不跨 0 的分差才值得当结论。

与 run_retrieval_ablation 的分工: 那个消融的是**检索链路配置** (单路/多query/
HyDE), 这个消融的是**编排层引擎** —— 两层各自的收益不混在一张表里。

用法 (需要真实 LLM key, .env 照常加载; 逐题串行, 延迟才可比):
    python tests/run_agent_ablation.py --kb kb-eval
    python tests/run_agent_ablation.py --kb kb-eval --engines rag,router,react -n 5

数据集: ``data/eval/agent_ablation_dataset_v2.jsonl`` (60 题), 每行
``{"question": ..., "keywords": [...], "type": ...}``。keywords 是答案应包含的
字面片段 (数字写 "23040" 不带千分位), 覆盖率 = 命中数 / 总数; type 供分层用
(single-hop / multi-hop / cross-doc, 计算题加 +calc 后缀)。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows 控制台默认 GBK, 中文问题会炸 UnicodeEncodeError
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 项目根目录

from app.core.config import settings  # noqa: E402
from app.rag.pipeline import run_rag  # noqa: E402

# tests/ 的上一级就是项目根 (路径解析只看文件自身位置, 与 CWD 无关)
_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET = _REPO / "data" / "eval" / "agent_ablation_dataset_v2.jsonl"


def load_jsonl(path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _norm(s: str) -> str:
    """去掉所有空白再匹配.

    语料与模型的空格习惯不一致: 语料写 "11 - 20 个 9 折", 模型写 "适用 9 折",
    而关键词若写成 "9折" 就永远匹配不上 —— 2026-10-16 首跑三引擎在折扣题上被
    集体误判 (0.67), 查答案才发现是口径 bug 而非 agent 失败。
    """
    return re.sub(r"\s+", "", s or "")


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    a = _norm(answer)
    hit = sum(1 for k in keywords if k and _norm(k) in a)
    return hit / len(keywords)


def count_tool_rounds(metadata: dict) -> int:
    """react 引擎实际发起工具调用的轮数 (decision=done 的那一轮不算)."""
    trace = (metadata or {}).get("trace") or []
    return sum(
        1
        for t in trace
        if t.get("role") == "reason"
        and isinstance(t.get("content"), dict)
        and t["content"].get("tool_calls")
    )


async def run_engine(name: str, engine: str, dataset: list[dict], kb: str, top_k: int):
    use_agent = engine != "rag"
    if use_agent:
        settings.agent_engine = engine
    rows: list[dict] = []
    for q in dataset:
        t0 = time.perf_counter()
        err = None
        out: dict = {}
        try:
            out = await run_rag(
                q["question"], kb_id=kb, top_k=top_k, use_agent=use_agent
            )
        except Exception as e:  # noqa: BLE001 - 单题失败记下来, 不拖垮整轮
            err = repr(e)
        dt = (time.perf_counter() - t0) * 1000
        answer = out.get("answer", "")
        row = {
            "question": q["question"],
            "coverage": keyword_coverage(answer, q.get("keywords", [])),
            "steps": count_tool_rounds(out.get("metadata") or {}) if use_agent else 0,
            "contexts": len(out.get("contexts") or []),
            "latency_ms": round(dt, 1),
            "answer_chars": len(answer),
            "answer_head": answer[:120],
            "error": err,
        }
        rows.append(row)
        flag = " ERR" if err else ""
        print(
            f"  [{name:<6}] cov={row['coverage']:.2f} steps={row['steps']} "
            f"ctx={row['contexts']} {row['latency_ms']:>7.0f}ms  "
            f"{q['question'][:28]}{flag}",
            flush=True,
        )
    return rows


def _lat(lats: list[float]) -> dict:
    if not lats:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    s = sorted(lats)
    p95 = s[min(len(s) - 1, max(0, round(0.95 * len(s)) - 1))]
    return {
        "mean": round(statistics.mean(s), 1),
        "p50": round(statistics.median(s), 1),
        "p95": round(p95, 1),
    }


def _by_type(rows: list[dict], dataset: list[dict]) -> dict:
    """分题型汇总. 题型来自数据集的 type 字段, 没有则归入 unknown.

    answer_chars 一并统计: react 的覆盖率若靠"话多"刷高, 在这里能被直接看出来.
    """
    type_of = {q["question"]: q.get("type", "unknown") for q in dataset}
    groups: dict[str, list[dict]] = {}
    for r in rows:
        groups.setdefault(type_of.get(r["question"], "unknown"), []).append(r)
    return {
        t: {
            "n": len(rs),
            "coverage_mean": round(statistics.mean([r["coverage"] for r in rs]), 4),
            "steps_mean": round(statistics.mean([r["steps"] for r in rs]), 2),
            "answer_chars_mean": round(
                statistics.mean([r.get("answer_chars", 0) for r in rs]), 1
            ),
            "latency_ms": _lat([r["latency_ms"] for r in rs]),
        }
        for t, rs in groups.items()
    }


def _paired_stats(results: dict, dataset: list[dict]) -> dict:
    """coverage 与端到端延迟都以 rag 为基线做配对 bootstrap (整体 + 分题型).

    同一批问题内部作差, 题目难度被自然消掉; CI 跨 0 的分差不能当结论写.
    """
    from app.rag.evaluate import paired_bootstrap_ci

    if "rag" not in results:
        return {}
    qorder = [q["question"] for q in dataset]
    type_of = {q["question"]: q.get("type", "unknown") for q in dataset}
    base = {r["question"]: r for r in results["rag"]["per_question"]}

    def one(cur: dict, qs: list[str]) -> dict:
        cov_a = [base[q]["coverage"] for q in qs]
        cov_b = [cur[q]["coverage"] for q in qs]
        lat_a = [base[q]["latency_ms"] for q in qs]
        lat_b = [cur[q]["latency_ms"] for q in qs]
        dc, lc, hc = paired_bootstrap_ci(cov_b, cov_a)
        dl, ll, hl = paired_bootstrap_ci(lat_b, lat_a)
        return {
            "n": len(qs),
            "coverage_delta": round(dc, 4),
            "coverage_ci": [round(lc, 4), round(hc, 4)],
            "coverage_significant": bool(lc > 0 or hc < 0),
            "latency_delta_ms": round(dl, 1),
            "latency_ci_ms": [round(ll, 1), round(hl, 1)],
            "latency_significant": bool(ll > 0 or hl < 0),
        }

    out: dict = {"overall": {}, "by_type": {}}
    for eng, res in results.items():
        if eng == "rag":
            continue
        cur = {r["question"]: r for r in res["per_question"]}
        common = [q for q in qorder if q in base and q in cur]
        if not common:
            continue
        out["overall"][f"{eng}_vs_rag"] = one(cur, common)
        for t in sorted(set(type_of.values())):
            qs = [q for q in common if type_of[q] == t]
            if qs:
                out["by_type"].setdefault(t, {})[f"{eng}_vs_rag"] = one(cur, qs)
    return out


def _print_by_type(results: dict) -> None:
    print("\n=== 分题型 ===")
    print(
        f"  {'type':<18}{'engine':<10}{'n':>4}{'coverage':>10}"
        f"{'steps':>7}{'chars':>8}{'mean_ms':>10}"
    )
    for eng, r in results.items():
        for t, v in r["by_type"].items():
            print(
                f"  {t:<18}{eng:<10}{v['n']:>4}{v['coverage_mean']:>10.3f}"
                f"{v['steps_mean']:>7.2f}{v['answer_chars_mean']:>8.0f}"
                f"{v['latency_ms']['mean']:>10.0f}"
            )


def _print_significance(sig: dict) -> None:
    if not sig:
        return
    print("\n=== 配对 bootstrap (Δ = engine - rag, 95% CI, * = CI 不跨 0) ===")
    rows = [("overall", k, v) for k, v in sig["overall"].items()]
    for t, d in sig["by_type"].items():
        rows.extend((t, k, v) for k, v in d.items())
    for scope, name, v in rows:
        cs = "*" if v["coverage_significant"] else " "
        ls = "*" if v["latency_significant"] else " "
        print(
            f"  [{scope:<16}] {name:<14} n={v['n']:>2} "
            f"covΔ={v['coverage_delta']:+.3f} "
            f"CI[{v['coverage_ci'][0]:+.3f},{v['coverage_ci'][1]:+.3f}]{cs} "
            f"latΔ={v['latency_delta_ms']:+.0f}ms "
            f"CI[{v['latency_ci_ms'][0]:+.0f},{v['latency_ci_ms'][1]:+.0f}]{ls}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb-eval", help="评测语料所在的 kb (需先建库并上传 data/eval/corpus/)")
    ap.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    ap.add_argument("--engines", default="rag,router,react", help="逗号分隔: rag,router,react")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("-n", "--num", type=int, default=None, help="只跑前 n 题")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dataset = load_jsonl(args.dataset)
    if not dataset:
        raise SystemExit(f"数据集为空: {args.dataset}")
    if args.num:
        dataset = dataset[: args.num]

    engines = [e.strip() for e in args.engines.split(",") if e.strip()]
    original_engine = settings.agent_engine
    results: dict[str, dict] = {}
    try:
        for engine in engines:
            name = engine if engine != "rag" else f"{engine}(无agent)"
            print(f"\n=== {name} ({len(dataset)} 题) ===", flush=True)
            rows = asyncio.run(run_engine(name, engine, dataset, args.kb, args.top_k))
            lats = [r["latency_ms"] for r in rows]
            results[engine] = {
                "n": len(rows),
                "coverage_mean": round(statistics.mean([r["coverage"] for r in rows]), 4),
                "steps_mean": round(statistics.mean([r["steps"] for r in rows]), 2),
                "steps_max": max((r["steps"] for r in rows), default=0),
                "errors": sum(1 for r in rows if r["error"]),
                "latency_ms": _lat(lats),
                "answer_chars_mean": round(
                    statistics.mean([r.get("answer_chars", 0) for r in rows]), 1
                ),
                "by_type": _by_type(rows, dataset),
                "per_question": rows,
            }
    finally:
        settings.agent_engine = original_engine

    significance = _paired_stats(results, dataset)
    for eng, res in results.items():
        if eng == "rag":
            continue
        key = f"{eng}_vs_rag"
        if key in significance.get("overall", {}):
            res["significance_vs_rag"] = significance["overall"][key]
        res["significance_by_type_vs_rag"] = {
            t: d[key]
            for t, d in significance.get("by_type", {}).items()
            if key in d
        }

    print("\n=== 汇总 ===")
    print(
        f"  {'engine':<12}{'coverage':>10}{'steps':>8}{'err':>5}"
        f"{'chars':>8}{'mean_ms':>10}{'p50':>8}{'p95':>8}"
    )
    for engine, r in results.items():
        lat = r["latency_ms"]
        print(
            f"  {engine:<12}{r['coverage_mean']:>10.3f}{r['steps_mean']:>8.2f}"
            f"{r['errors']:>5}{r['answer_chars_mean']:>8.0f}"
            f"{lat['mean']:>10.0f}{lat['p50']:>8.0f}{lat['p95']:>8.0f}"
        )
    _print_by_type(results)
    _print_significance(significance)

    if args.out is None:
        args.out = _REPO / "data" / "eval" / (
            "agent_ablation_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
