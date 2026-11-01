# -*- coding: utf-8 -*-
"""工具描述优化 A/B: chunk_id 显式化 (工作区新版) vs HEAD 旧版.

被测改动 (app/agents/tools.py, 未提交, 逐字对照 ``git show HEAD``):
1. format_hits 每行候选新增 ``(chunk_id=xxx)`` 字段;
2. hybrid_search docstring 写明 "[n] 仅是展示序号, 不是任何工具的参数,
   需要完整原文/上下文时用 chunk_id 调 read_chunk/read_parent";
3. read_chunk / read_parent docstring 写明 "chunk_id 不是 [n] 展示序号, 也不是纯数字".

动机: 旧观测里 [n] 是模型唯一看得到的"编号", 模型可能把它当 chunk_id 传给
read_chunk / read_parent -> "chunk 不存在", 白耗一轮工具预算 (agent_max_steps
默认只有几轮, 一次 miss 就是实打实的证据损失)。

A/B 方式 (进程内切换, 不动工作区文件, 不需要重建编译好的图):
- old 臂: format_hits / build_tools 替换成 HEAD 等价实现;
- new 臂: 工作区原样。
reason_node 每轮按模块名查找 ``build_tools``, handler 运行时按模块名查找
``format_hits`` (见 graph.reset_agent_graph_cache 注释: patch 即时生效)。

两个 regime:
- default: 生产口径 (context_snippet_chars 不动)。上次消融 (agent_ablation_v2)
  react steps_mean 1.27, read_* 几乎不被触发 —— 该口径预期测不出差异, 跑它是
  为了证明"默认口径无回归";
- stress: context_snippet_chars=300, 摘要被切断, 逼模型调 read_chunk /
  read_parent 才能看到全文, [n] 误用才会高频出现 —— 测机制收益的口径。
  两臂同压, 对比仍然公平。

指标 (全部取自 metadata.trace, coverage/steps 与 run_agent_ablation 同口径):
- coverage: 参考关键词覆盖率;
- steps: 带工具调用的决策轮数;
- searches / reads: hybrid_search 与 read_* 调用次数 (行为基线);
- bad_reads: read_* 传**纯数字** chunk_id 的调用数 ([n] 误用的直接证据);
- miss_reads: 观测以 "chunk 不存在" 开头的次数 (含模型瞎猜 id, 是 bad_reads
  的超集 —— bad_reads 看的是"传了什么", miss_reads 看的是"结果如何");
- latency: 端到端 wall time。
显著性: 逐题配对 bootstrap (Δ = new - old), CI 不跨 0 才值得当结论。

用法 (需要真实 LLM key; 逐题串行且 old/new 背靠背, 时间上配对):
    python tests/run_tooldesc_ab.py --kb kb-eval                          # 全量
    python tests/run_tooldesc_ab.py --kb kb-eval --num 3 --regimes stress # 冒烟
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
import app.agents.graph as graph_mod  # noqa: E402
import app.agents.tools as tools_mod  # noqa: E402

_REPO = Path(__file__).resolve().parents[1]
_DEFAULT_DATASET = _REPO / "data" / "eval" / "agent_ablation_dataset_v2.jsonl"

# ---------------------------------------------------------------------------
# old 臂: HEAD 版实现 (docstring / 输出格式逐字对照 git show HEAD:app/agents/tools.py)
# ---------------------------------------------------------------------------

_OLD_HYBRID_DOC = (
    "从企业知识库混合检索（BM25+向量+RRF 重排）。自行组织检索词, 不要照抄用户原话"
    "（口语/指代需展开）；结果不相关时换关键词重试。返回 [n] (score=..) 正文 格式的候选。"
)
_OLD_READ_CHUNK_DOC = "按 chunk_id 读取一条检索结果的完整原文。"
_OLD_READ_PARENT_DOC = "读取某 chunk 及其在原文档中紧邻的前后各一个 chunk, 补全被切断的上下文。"


def _old_format_hits(hits: list) -> str:
    limit = getattr(settings, "context_snippet_chars", 2000)
    lines = []
    for i, r in enumerate(hits, 1):
        doc = r.get("document", "")
        snippet = doc[:limit].replace("\n", " ")
        score = r.get("rerank_score", 0)
        lines.append("[" + str(i) + "] (score=" + format(score, ".3f") + ") " + snippet)
    return "\n".join(lines)


def _old_build_tools(kb_id="default"):
    from langchain_core.tools import tool

    H = tools_mod.HANDLERS

    @tool
    def hybrid_search(query: str, top_k: int = 5) -> str:
        "从企业知识库混合检索（BM25+向量+RRF 重排）。自行组织检索词, 不要照抄用户原话（口语/指代需展开）；结果不相关时换关键词重试。返回 [n] (score=..) 正文 格式的候选。"
        obs, _ = H["hybrid_search"]({"query": query, "top_k": top_k}, kb_id)
        return obs

    @tool
    def read_chunk(chunk_id: str) -> str:
        "按 chunk_id 读取一条检索结果的完整原文。"
        obs, _ = H["read_chunk"]({"chunk_id": chunk_id}, kb_id)
        return obs

    @tool
    def read_parent(chunk_id: str) -> str:
        "读取某 chunk 及其在原文档中紧邻的前后各一个 chunk, 补全被切断的上下文。"
        obs, _ = H["read_parent"]({"chunk_id": chunk_id}, kb_id)
        return obs

    @tool
    def list_documents() -> str:
        "列出当前知识库的全部文档 (doc_id / 文件名 / chunk 数), 适合先了解库里有什么再决定怎么检索。"
        obs, _ = H["list_documents"]({}, kb_id)
        return obs

    @tool
    def calculator(expression: str) -> str:
        "安全计算数学表达式: 支持 + - * / // % **, 一元正负号, math 模块常用函数."
        obs, _ = H["calculator"]({"expression": expression}, kb_id)
        return obs

    @tool
    def current_date() -> str:
        "获取当前日期时间。"
        obs, _ = H["current_date"]({}, kb_id)
        return obs

    return [hybrid_search, read_chunk, read_parent, list_documents, calculator, current_date]


_NEW_FORMAT_HITS = tools_mod.format_hits
_NEW_BUILD_TOOLS = tools_mod.build_tools


def apply_arm(arm: str) -> None:
    if arm == "old":
        tools_mod.format_hits = _old_format_hits
        graph_mod.build_tools = _old_build_tools
        tools_mod.build_tools = _old_build_tools
    else:
        tools_mod.format_hits = _NEW_FORMAT_HITS
        graph_mod.build_tools = _NEW_BUILD_TOOLS
        tools_mod.build_tools = _NEW_BUILD_TOOLS


def selfcheck() -> None:
    """跑之前先证明两臂真的不一样, 否则 A/B 白跑."""
    fake = [{"id": "doc_0", "document": "正文内容", "rerank_score": 0.5}]
    old_line = _old_format_hits(fake)
    new_line = _NEW_FORMAT_HITS(fake)
    assert old_line == "[1] (score=0.500) 正文内容", f"old 臂输出与 HEAD 不符: {old_line!r}"
    assert "chunk_id=doc_0" in new_line, f"new 臂输出未带 chunk_id: {new_line!r}"

    old_tools = {t.name: t for t in _old_build_tools("kb-selfcheck")}
    new_tools = {t.name: t for t in _NEW_BUILD_TOOLS("kb-selfcheck")}
    assert old_tools["hybrid_search"].description == _OLD_HYBRID_DOC, "old hybrid docstring 漂移"
    assert old_tools["read_chunk"].description == _OLD_READ_CHUNK_DOC, "old read_chunk docstring 漂移"
    assert old_tools["read_parent"].description == _OLD_READ_PARENT_DOC, "old read_parent docstring 漂移"
    for name in ("hybrid_search", "read_chunk", "read_parent"):
        assert old_tools[name].description != new_tools[name].description, f"{name} 两臂相同, A/B 无意义"
    # 未被测的三个工具两臂必须一致 (它们若有差异, 说明工作区改动超出本 A/B 的范围)
    for name in ("list_documents", "calculator", "current_date"):
        assert old_tools[name].description == new_tools[name].description, name


# ---------------------------------------------------------------------------
# 指标 (coverage / count_tool_rounds 与 run_agent_ablation 同口径, 此处复制
# 而非 import: tests 不是包, 且两脚本要能独立运行)
# ---------------------------------------------------------------------------


def load_jsonl(path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def keyword_coverage(answer: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    a = _norm(answer)
    hit = sum(1 for k in keywords if k and _norm(k) in a)
    return hit / len(keywords)


def count_tool_rounds(metadata: dict) -> int:
    trace = (metadata or {}).get("trace") or []
    return sum(
        1
        for t in trace
        if t.get("role") == "reason"
        and isinstance(t.get("content"), dict)
        and t["content"].get("tool_calls")
    )


_PURE_NUM = re.compile(r"\d+")


def analyze_trace(metadata: dict) -> dict:
    """从 trace 抽数: 调用量 / 纯数字 chunk_id ([n] 误用) / miss 观测.

    read_ids / missed_ids 是明细 (不进 bootstrap, 只落盘供报告点名失败形态):
    miss 观测文本本身带 id ("chunk 不存在: X"), 直接从观测侧提取。
    """
    trace = (metadata or {}).get("trace") or []
    searches = reads = bad_reads = 0
    read_ids: list[str] = []
    for t in trace:
        if t.get("role") != "reason":
            continue
        c = t.get("content")
        if not isinstance(c, dict):
            continue
        for call in c.get("tool_calls") or []:
            name = str(call.get("name") or "")
            args = call.get("args") or {}
            if name == "hybrid_search":
                searches += 1
            elif name in ("read_chunk", "read_parent"):
                reads += 1
                cid = str(args.get("chunk_id") or "").strip()
                read_ids.append(cid)
                if _PURE_NUM.fullmatch(cid):
                    bad_reads += 1
    missed_ids = [
        str(t.get("content") or "").split(":", 1)[-1].strip()
        for t in trace
        if str(t.get("role") or "").startswith("tool:read_")
        and str(t.get("content") or "").startswith("chunk 不存在")
    ]
    return {
        "searches": searches,
        "reads": reads,
        "bad_reads": bad_reads,
        "miss_reads": len(missed_ids),
        "read_ids": read_ids,
        "missed_ids": missed_ids,
    }


async def run_one(arm: str, question: str, keywords: list[str], kb: str, top_k: int) -> dict:
    apply_arm(arm)
    t0 = time.perf_counter()
    err = None
    out: dict = {}
    try:
        out = await run_rag(question, kb_id=kb, top_k=top_k, use_agent=True)
    except Exception as e:  # noqa: BLE001 - 单题失败记下来, 不拖垮整轮
        err = repr(e)
    dt = (time.perf_counter() - t0) * 1000
    md = out.get("metadata") or {}
    answer = out.get("answer", "")
    row = {
        "question": question,
        "coverage": keyword_coverage(answer, keywords),
        "steps": count_tool_rounds(md),
        **analyze_trace(md),
        "latency_ms": round(dt, 1),
        "answer_head": answer[:100],
        "error": err,
    }
    print(
        f"    [{arm:<3}] cov={row['coverage']:.2f} steps={row['steps']} "
        f"srch={row['searches']} rd={row['reads']} bad={row['bad_reads']} "
        f"miss={row['miss_reads']} {row['latency_ms']:>6.0f}ms{(' ERR:' + err[:60]) if err else ''}",
        flush=True,
    )
    return row


def _lat(lats: list[float]) -> dict:
    if not lats:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    s = sorted(lats)
    return {
        "mean": round(statistics.mean(s), 1),
        "p50": round(statistics.median(s), 1),
        "p95": round(s[min(len(s) - 1, max(0, round(0.95 * len(s)) - 1))], 1),
    }


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        "n": n,
        "coverage_mean": round(statistics.mean([r["coverage"] for r in rows]), 4),
        "steps_mean": round(statistics.mean([r["steps"] for r in rows]), 2),
        "searches_mean": round(statistics.mean([r["searches"] for r in rows]), 2),
        "reads_mean": round(statistics.mean([r["reads"] for r in rows]), 2),
        "reads_total": sum(r["reads"] for r in rows),
        "bad_reads_total": sum(r["bad_reads"] for r in rows),
        "miss_reads_total": sum(r["miss_reads"] for r in rows),
        "questions_with_reads": sum(1 for r in rows if r["reads"] > 0),
        "questions_with_miss": sum(1 for r in rows if r["miss_reads"] > 0),
        "errors": sum(1 for r in rows if r["error"]),
        "latency_ms": _lat([r["latency_ms"] for r in rows]),
    }


_PAIRED_METRICS = ("coverage", "steps", "searches", "reads", "bad_reads", "miss_reads", "latency_ms")


def paired(new_rows: list[dict], old_rows: list[dict]) -> dict:
    """逐题配对 bootstrap, Δ = new - old (顺序与两臂行序一致)."""
    from app.rag.evaluate import paired_bootstrap_ci

    out = {}
    for m in _PAIRED_METRICS:
        a = [r[m] for r in new_rows]
        b = [r[m] for r in old_rows]
        d, lo, hi = paired_bootstrap_ci(a, b)
        out[m] = {
            "delta": round(d, 4),
            "ci": [round(lo, 4), round(hi, 4)],
            "significant": bool(lo > 0 or hi < 0),
        }
    return out


def print_regime(name: str, res: dict) -> None:
    print(f"\n=== [{name}] 汇总 ===")
    print(
        f"  {'arm':<5}{'coverage':>9}{'steps':>7}{'srch':>6}{'rd':>5}{'bad':>5}"
        f"{'miss':>5}{'触read题':>9}{'err':>5}{'mean_ms':>10}"
    )
    for arm in ("old", "new"):
        s = res[arm]["summary"]
        print(
            f"  {arm:<5}{s['coverage_mean']:>9.3f}{s['steps_mean']:>7.2f}"
            f"{s['searches_mean']:>6.2f}{s['reads_mean']:>5.2f}"
            f"{s['bad_reads_total']:>5}{s['miss_reads_total']:>5}"
            f"{s['questions_with_reads']:>4}/{s['n']:<4}{s['errors']:>5}"
            f"{s['latency_ms']['mean']:>10.0f}"
        )
    print("  --- 配对 bootstrap (Δ = new - old, 95% CI, * = CI 不跨 0) ---")
    for m, v in res["paired"].items():
        star = "*" if v["significant"] else " "
        print(
            f"  {m:<12} Δ={v['delta']:>9.3f}  CI[{v['ci'][0]:>9.3f},{v['ci'][1]:>9.3f}] {star}"
        )


async def run_regime(name: str, snippet: int | None, dataset: list[dict], kb: str, top_k: int) -> dict:
    original_snippet = settings.context_snippet_chars
    if snippet is not None:
        settings.context_snippet_chars = snippet
        print(f"\n=== regime [{name}]: context_snippet_chars {original_snippet} -> {snippet} ===", flush=True)
    else:
        print(f"\n=== regime [{name}]: context_snippet_chars={original_snippet} (生产口径) ===", flush=True)
    old_rows: list[dict] = []
    new_rows: list[dict] = []
    try:
        for q in dataset:
            print(f"  Q: {q['question'][:40]}", flush=True)
            # old/new 背靠背: 同一题的两臂在时间上紧邻, provider 抖动对配对差分的
            # 污染最小 (先跑哪臂固定为 old, 顺序偏差两臂同担)。
            old_rows.append(await run_one("old", q["question"], q.get("keywords", []), kb, top_k))
            new_rows.append(await run_one("new", q["question"], q.get("keywords", []), kb, top_k))
    finally:
        if snippet is not None:
            settings.context_snippet_chars = original_snippet
    res = {
        "snippet_chars": snippet if snippet is not None else int(original_snippet),
        "old": {"summary": summarize(old_rows), "per_question": old_rows},
        "new": {"summary": summarize(new_rows), "per_question": new_rows},
    }
    res["paired"] = paired(res["new"]["per_question"], res["old"]["per_question"])
    print_regime(name, res)
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="kb-eval")
    ap.add_argument("--dataset", type=Path, default=_DEFAULT_DATASET)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("-n", "--num", type=int, default=None, help="只跑前 n 题")
    ap.add_argument(
        "--regimes", default="default,stress", help="逗号分隔: default,stress"
    )
    ap.add_argument("--stress-chars", type=int, default=300, help="stress 口径的摘要截断长度")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    dataset = load_jsonl(args.dataset)
    if not dataset:
        raise SystemExit(f"数据集为空: {args.dataset}")
    if args.num:
        dataset = dataset[: args.num]

    selfcheck()
    print(
        f"provider={getattr(settings, 'llm_provider', '?')} "
        f"engine=react kb={args.kb} n={len(dataset)} "
        f"agent_max_steps={getattr(settings, 'agent_max_steps', '?')}",
        flush=True,
    )

    regime_map = {
        "default": ("default", None),
        "stress": ("stress", args.stress_chars),
    }
    wanted = [r.strip() for r in args.regimes.split(",") if r.strip()]
    unknown = [r for r in wanted if r not in regime_map]
    if unknown:
        raise SystemExit(f"未知 regime: {unknown}")

    original_engine = settings.agent_engine
    settings.agent_engine = "react"
    results: dict = {}
    try:
        for key in wanted:
            name, snippet = regime_map[key]
            results[name] = asyncio.run(
                run_regime(name, snippet, dataset, args.kb, args.top_k)
            )
    finally:
        settings.agent_engine = original_engine
        apply_arm("new")  # 还原工作区行为, 不把 old 臂留在进程状态里

    meta = {
        "kb": args.kb,
        "n": len(dataset),
        "dataset": str(args.dataset),
        "provider": getattr(settings, "llm_provider", None),
        "agent_max_steps": getattr(settings, "agent_max_steps", None),
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "change": "format_hits 带 chunk_id + hybrid_search/read_chunk docstring 区分 [n] 与 chunk_id",
    }
    payload = {"meta": meta, **results}
    if args.out is None:
        args.out = _REPO / "data" / "eval" / (
            "tooldesc_ab_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".json"
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n结果已写入: {args.out}")


if __name__ == "__main__":
    main()
