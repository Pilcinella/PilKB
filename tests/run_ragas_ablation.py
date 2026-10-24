"""生成侧消融: 单路单query vs 双路多query 的 Context Precision / Faithfulness.

RAGAS (LLM judge), 需要真实 LLM key —— 生成与 judge 都用 settings.llm_provider
配置的模型. 检索与 run_retrieval_ablation **完全同口径**: 同一 kb、同一份改写
缓存回放、同一套单路实例补丁, 两张表可以并排放进同一页对比.

ground_truth 口径 (必读): 当前数据集 (recall_dataset.jsonl) 只有 reference_context
(答案所在的原文片段), 没有独立撰写的标准答案; 而 RAGAS context_precision 必需
ground_truth. 这里把多条 reference 拼接后当 ground_truth 用 —— 对"检索上下文是否
支撑得出参考内容"这个判据成立, 但与"对标准答案的精确率"有差别, 写结论时要注明.
Faithfulness (答案是否忠于检索上下文) 不依赖 ground_truth, 不受此影响.

为什么 judge 要显式接线: ragas 0.1.10 默认 llm_factory 拉 OpenAI, 配 deepseek/qwen
key 时 judge 初始化直接失败 (run_ragas_eval real 模式踩过的坑). 这里显式传
LangchainLLMWrapper(langchain_llm=llm_router.get_llm()), judge 与生成共用同一 provider.

用法 (完整依赖环境 + .env 配好 LLM key):
  cd backend
  python -m tests.run_ragas_ablation --out ../data/eval/ablation_ragas.json
  # 先小跑: -n 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import math
from datetime import datetime
from pathlib import Path

from loguru import logger

from tests import _ablation_common as ab

_DATA = Path(__file__).resolve().parents[2] / "data"


def _dump_gen_cache(cache: dict, path: Path) -> None:
    """落盘生成缓存. 只存 answer/contexts —— RAGAS 只用这两项, ground_truth 由
    items 现算 (确定性, 不需要缓存)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")


def _run_config(
    name: str,
    cfg: dict,
    kb: str,
    items: list[dict],
    gen_cache: dict | None = None,
    gen_cache_path: Path | None = None,
) -> dict:
    """生成阶段 (走 run_rag 全链路, 检索配置生效) -> RAGAS 打分 (配置已恢复).

    gen_cache: {(config, question): {answer, contexts}} 形式的中断续跑缓存.
    生成一次约 5 分钟且烧 LLM, 被进程退出打断就全丢 —— 每 10 题落一次盘,
    重跑时命中的题直接复用. **缓存键不含代码版本**: 改过 graph / 检索链路后
    必须删掉缓存文件, 否则续跑用的是旧代码生成的结果.
    """
    from app.rag.pipeline import run_rag

    saved_cfg = ab.apply_settings(cfg["settings"])
    saved_stores = ab.patch_single_path(kb, cfg["path"]) if cfg["path"] != "hybrid" else []
    questions, answers, contexts, gts = [], [], [], []
    n_cached = 0
    try:
        for i, r in enumerate(items, 1):
            q = r["question"]
            key = f"{name}\x1f{q}"
            hit = (gen_cache or {}).get(key)
            if hit is not None:
                n_cached += 1
                res = hit
            else:
                res = asyncio.run(run_rag(question=q, kb_id=kb))
                if gen_cache is not None:
                    gen_cache[key] = {"answer": res["answer"], "contexts": res["contexts"]}
                    if gen_cache_path is not None and i % 10 == 0:
                        _dump_gen_cache(gen_cache, gen_cache_path)
            questions.append(q)
            answers.append(res["answer"])
            contexts.append(res["contexts"])
            gts.append("\n".join(r["reference_context"]))
            logger.info(f"[{name} {i}/{len(items)}] {q[:32]} -> {res['answer'][:48]}")
    finally:
        # 生成阶段结束就恢复配置: judge 阶段不该再受单路补丁影响.
        ab.restore_stores(saved_stores)
        ab.restore_settings(saved_cfg)
        if gen_cache is not None and gen_cache_path is not None:
            _dump_gen_cache(gen_cache, gen_cache_path)
    if n_cached:
        logger.info(f"[{name}] 生成阶段命中缓存 {n_cached}/{len(items)} 题")

    from datasets import Dataset

    ds = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": gts,
        }
    )

    from app.agents import llm_router
    from ragas import evaluate
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import context_precision, faithfulness
    from ragas.run_config import RunConfig

    # ragas 0.1.10 的 executor 走 asyncio.get_event_loop() 旧范式: Python 3.12+
    # 主线程没有"当前 loop"时直接 RuntimeError (Docker 里的老版本 Python 从没暴露过).
    # 显式建一个再进 evaluate.
    asyncio.set_event_loop(asyncio.new_event_loop())
    # 并发必须压: 默认 max_workers=16 会把 provider 打到限流, 429 -> SDK 重试 ->
    # tenacity 再重试, 单任务累计轻松超过默认 180s 的 wait_for, 整轮被一个
    # TimeoutError 炸掉 (实测: 57 行 x 2 指标在默认配置下必挂, 1 行最小复现正常).
    scores = evaluate(
        ds,
        metrics=[faithfulness, context_precision],
        llm=LangchainLLMWrapper(langchain_llm=llm_router.get_llm()),
        run_config=RunConfig(timeout=240, max_workers=4, max_retries=5, max_wait=60),
        raise_exceptions=False,  # 单行失败转 NaN, 由下面的 nan_metrics 暴露, 不炸整轮
    )
    # ragas parser 全失败时给 NaN, JSON 不允许 NaN: 置 0 并记进 nan_metrics,
    # 读报告时先看它 —— 0 分可能是退化, 不一定是真差.
    cleaned: dict = {}
    nan_metrics: list[str] = []
    for k, v in dict(scores).items():
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = math.nan
        if fv != fv:
            nan_metrics.append(k)
            fv = 0.0
        cleaned[k] = fv

    return {
        "scores": cleaned,
        "nan_metrics": nan_metrics,
        "n": len(questions),
        "samples": [
            {"question": q, "answer": a, "contexts": c, "ground_truth": g}
            for q, a, c, g in zip(questions, answers, contexts, gts)
        ],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kb", default="default", help="当前存了演示语料的 kb")
    ap.add_argument("--dataset", type=Path, default=_DATA / "eval" / "recall_dataset.jsonl")
    ap.add_argument("--gen-cache", type=Path, default=_DATA / "eval" / "hyde_gen_cache.json")
    ap.add_argument("--single-path", choices=["vector", "bm25"], default="vector")
    ap.add_argument("--with-hyde", action="store_true")
    ap.add_argument("-n", "--num", type=int, default=None, help="只跑前 n 条 (小跑验证用)")
    ap.add_argument(
        "--run-cache", type=Path, default=_DATA / "eval" / "ragas_gen_cache.json",
        help="生成结果的中断续跑缓存 (每 10 题落盘)",
    )
    ap.add_argument("--no-run-cache", action="store_true", help="禁用生成缓存, 强制重新生成")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    # ragas 0.1.10 import 链硬拉 langchain_community.vertexai (沙箱里元类冲突),
    # 必须在任何 import ragas 之前注入空壳.
    from tests.run_ragas_eval import _stub_langchain_community_vertexai

    _stub_langchain_community_vertexai()

    dataset = ab.load_jsonl(args.dataset)
    # context_precision 需 ground_truth, 无 reference 的样本进不来 (qa.jsonl 那种
    # 带标准答案的数据集换语料时也可以直接 --dataset 换过来).
    items = [
        {"question": r["question"], "reference_context": r["reference_context"]}
        for r in dataset
        if r.get("question") and r.get("reference_context")
    ]
    if not items:
        raise SystemExit(f"数据集里没有带 reference_context 的问题: {args.dataset}")
    if args.num:
        items = items[: args.num]
    rewrite_cache = ab.load_gen_cache(args.gen_cache)

    # 生成结果缓存: 被打断后重跑不必重新生成+重新烧 LLM.
    # 键不含代码版本 —— 改过 graph / 检索链路必须 --no-run-cache 或删文件,
    # 否则续跑的是旧代码生成的结果, 消融对比直接失真.
    run_cache: dict | None = None
    if not args.no_run_cache:
        if args.run_cache.exists():
            run_cache = json.loads(args.run_cache.read_text(encoding="utf-8"))
            logger.warning(
                f"载入生成缓存 {len(run_cache)} 条 ({args.run_cache.name}); "
                f"若本轮改过 graph/检索代码, 请用 --no-run-cache 重新生成"
            )
        else:
            run_cache = {}

    print(
        f"[RAGAS 消融] kb={args.kb} | 样本 {len(items)} 条 "
        f"| 改写缓存={args.gen_cache.name} ({len(rewrite_cache)} 条) | hyde={args.with_hyde} "
        f"| 生成缓存={'off' if run_cache is None else len(run_cache)}"
    )

    ab.install_rewrite_replay(rewrite_cache)
    try:
        configs = ab.config_matrix(args.single_path, args.with_hyde)
        results = {
            name: _run_config(
                name, cfg, args.kb, items,
                gen_cache=run_cache,
                gen_cache_path=None if run_cache is None else args.run_cache,
            )
            for name, cfg in configs.items()
        }
    finally:
        ab.uninstall_rewrite_replay()

    print(f"\n=== RAGAS 消融 (n={len(items)}) ===")
    names = list(results)
    print(f"  {'指标':<20}" + "".join(f"{n:>16}" for n in names))
    for metric in ("context_precision", "faithfulness"):
        cells = [f"{r['scores'].get(metric, float('nan')):>16.4f}" for r in results.values()]
        print(f"  {metric:<20}" + "".join(cells))
    for name, r in results.items():
        if r["nan_metrics"]:
            print(f"  [{name}] 退化指标 (parser 失败已置 0): {r['nan_metrics']}")

    if args.out:
        payload = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "kb": args.kb,
            "dataset": str(args.dataset),
            "gen_cache": str(args.gen_cache),
            "with_hyde": args.with_hyde,
            "results": {name: {"config": configs[name], **r} for name, r in results.items()},
            "note": (
                "ground_truth = reference_context 拼接 (数据集无独立标准答案, 见脚本"
                "docstring 的口径说明); judge = llm_provider 同款模型 (LangchainLLM 接线); "
                "检索阶段与 run_retrieval_ablation 同口径 (缓存回放改写)."
            ),
        }
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
