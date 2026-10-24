# -*- coding: utf-8 -*-
"""Agent 引擎消融: react (工具循环) vs router (一次性路由) vs rag (无 agent).

对同一批问题逐个跑指定引擎, 输出三张口径可比的表:
- 命中率: 参考关键词在答案中的覆盖率 (文本包含判定, 与 recall 消融的
  ``--judge text`` 同口径 —— 不追求语义等价, 追求可复现零成本);
- 步数: react 的实际工具轮数 (metadata.trace 里带 tool_calls 的 reason 条数);
- 延迟: 端到端 wall time 的 mean / p50 / p95。

与 run_retrieval_ablation 的分工: 那个消融的是**检索链路配置** (单路/多query/
HyDE), 这个消融的是**编排层引擎** —— 两层各自的收益不混在一张表里。

用法 (需要真实 LLM key, .env 照常加载; 逐题串行, 延迟才可比):
    python tests/run_agent_ablation.py --kb kb-demo
    python tests/run_agent_ablation.py --kb kb-demo --engines rag,router,react -n 5

数据集: ``{repo}/data/eval/agent_ablation_dataset.jsonl``, 每行
``{"question": ..., "keywords": [...]}``。keywords 是答案应包含的字面片段
(数字写 "23040" 不带千分位), 覆盖率 = 命中数 / 总数。
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

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET = _REPO / "data" / "eval" / "agent_ablation_dataset.jsonl"


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb-demo", help="演示语料所在的 kb")
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
                "per_question": rows,
            }
    finally:
        settings.agent_engine = original_engine

    print("\n=== 汇总 ===")
    print(f"  {'engine':<12}{'coverage':>10}{'steps':>8}{'err':>5}{'mean_ms':>10}{'p50':>8}{'p95':>8}")
    for engine, r in results.items():
        lat = r["latency_ms"]
        print(
            f"  {engine:<12}{r['coverage_mean']:>10.3f}{r['steps_mean']:>8.2f}"
            f"{r['errors']:>5}{lat['mean']:>10.0f}{lat['p50']:>8.0f}{lat['p95']:>8.0f}"
        )

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
