"""_tokenize 三代实现的离线消融: 固定语料与切分, 只替换分词函数, 复现迭代数据。

背景
----
`app/rag/bm25_store.py::_tokenize` 经历过三代实现, 模块 docstring 里记了每代的
实测结论. 本脚本把三代实现**并排放在同一份语料 + 同一份评估集**上重跑, 把
"为什么必须迭代"从注释里的说法变成可复现、可重跑的数字.

    v1  连续汉字串整段作为一个 token   (re.findall 的结果直接当 token 列表)
    v2  中文段 2-gram                 (现为 jieba 缺失时的回退分支)
    v3  jieba 词级切分                (当前生产实现, 直接 import 不复制)

为什么这个脚本能秒级跑完
------------------------
与 run_recall_eval.py 不同, 本脚本**只测 BM25 单路**: 不加载 embedding 模型、
不碰向量库、不需要 API key, jieba 与 rank_bm25 都是纯 CPU. 因此它能在任何环境
里跑, 适合当"动分词代码之前先看一眼"的快检.
刻意不走 HybridRetriever: RRF 融合会把 BM25 这一路的退化掩盖掉 —— v1 交白卷时
向量路仍能召回, 融合后指标看着还行, 差异就看不见了.

三个观察口径
------------
1) 词表交集: query 的 token 集合与语料词表有没有交集. 无交集意味着这条 query 在
   BM25 下恒为 0 分, 这一路召回等于不存在. docstring 里 v1 的 "0/47" 即此口径;
   47 是评估集里不含 ASCII 字母/数字的纯中文问题数.
2) BM25 单路 Recall@K: 按 reference_context 文本匹配计算 (复用 evaluate.py 的
   recall_at_k_by_context, 阈值 0.7), 不依赖 chunk_id, 重新入库后依然可比.
3) 分词开销: 语料 token 总数 / 词表大小 / 平均每 chunk token 数, 直接对应索引
   体积与打分耗时.

复现结论 (62 条评估集, chunk_size=300, 本轮实测)
----------------------------------------------
  - chunk_size=300 是那批数据的基准粒度 (生产默认是 500), 因此本脚本默认取 300.
    离开这个粒度 v1 就对不上 —— 见下面第三条.
  - v2 的两个数字能精确对上: 词表交集 98.4%、Recall@5 0.9825 (docstring 记 98%、0.983).
  - v1 对不齐, 且两个口径指向**不同的 v1 变体**, 如实记在这里:
        * 词表交集 24%: 带 lower 得 15/62 = 24.2% ✓; 不带 lower 得 13/62 = 21.0%
        * Recall@5 0.228: **不带 lower** 才得 0.2281 ✓; 带 lower 得 0.2807
    也就是说 docstring 那两个数字并非同一次跑出来的 (交集疑似后测, Recall 抄的是
    旧值, 或反之). 本脚本的 v1 取**带 lower** —— 它与 _SEG_RE 在 v2/v3 里的调用
    方式一致, 是更可信的历史形态; 代价是表 2 的 v1@5 比 docstring 记的略高.
  - "token 数减半" 这个说法要打个折扣: v2 -> v3 语料 token 总数 7060 -> 4636
    (减 34%), 而**词表** 3182 -> 1322 (减 58%). 减半的是词表不是 token 数, 而索引
    体积与打分耗时是按 token 总数走的 —— 收益比"减半"听起来的小.
  - v1 的召回对切分粒度**极度敏感**: chunk_size 从 200 涨到 1200, @5 从 0.211 爬到
    0.912; 而 v2/v3 几乎不动 (稳定在 0.95~1.00). 看表 2 的"零分占比"列就明白了:
    v1 恒在 ~90% —— 它**从不真的匹配**, 只是 chunk 越大覆盖越广, top-5 蒙中答案
    的概率越高. 分数涨了, 检索质量没涨.
  - 于是一个很容易踩的坑: 用大 chunk 测 v1 会得到"v1 也还行"的错觉 (@5=0.912),
    进而误判分词改造没必要. --sweep 就是把这个陷阱摆到台面上.

用法
----
    cd backend
    python -m tests.run_tokenizer_ablation
    python -m tests.run_tokenizer_ablation --out ../data/eval/tokenizer_ablation.json
    python -m tests.run_tokenizer_ablation --sweep        # 扫切分粒度, 看 v1 的假象怎么来的
    python -m tests.run_tokenizer_ablation --chunk-size 300
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from loguru import logger

_BACKEND = Path(__file__).resolve().parents[1]
_DATA = _BACKEND.parent / "data"

# 与生产共用同一个正则: 重新定义会漂移, 而漂移了实验结论就不作数了.
from app.rag.bm25_store import _SEG_RE, _tokenize  # noqa: E402

TokenizeFn = Callable[[str], List[str]]

# 判断"纯中文问题"用: 不含任何 ASCII 字母/数字的问题, 在 v1 下必然整句成 1 个 token.
_ASCII_ALNUM = re.compile(r"[a-zA-Z0-9]")


# ---------------------------------------------------------------------------
# 三代实现
# ---------------------------------------------------------------------------

def tokenize_v1(text: str) -> List[str]:
    """第一代: 连续汉字串整段作为一个 token.

    docstring 把这一代记作 `re.findall(r"[一-龥]+|[a-z0-9]+")`, 那个写法省略了
    `.lower()`. 这里取带 lower 的形态 —— 它与 _SEG_RE 在 v2/v3 里的调用方式一致,
    也对得上 docstring 记的 24% 交集率; 大小写这两个变体的取舍, 详见模块 docstring
    的复现结论 (另一口径 Recall@5 指向的恰好是**不带** lower 的变体).

    真正的病灶不在大小写, 在 `[一-龥]+`: 中文句内没有分隔符, 整句只产出 1 个
    token, 与任何 chunk 都对不上 —— 这才是 0/47 的来源.
    """
    return _SEG_RE.findall((text or "").lower())


def tokenize_v2(text: str) -> List[str]:
    """第二代: 中文段 2-gram, 英文/数字整词.

    与当前生产代码里 jieba 缺失时的回退分支逐行一致 —— 回退分支就是 v2 的化石.
    """
    out: List[str] = []
    for seg in _SEG_RE.findall((text or "").lower()):
        if seg[0].isascii():
            out.append(seg)  # 英文/数字: 整词
        elif len(seg) == 1:
            out.append(seg)
        else:
            out.extend(seg[i : i + 2] for i in range(len(seg) - 1))
    return out


def tokenize_v3(text: str) -> List[str]:
    """第三代 = 当前生产实现.

    直接 import 而不是复制代码: 复制一份就等于埋了一份会过期的"生产实现",
    改了 bm25_store 却忘了改这里, 实验结论会悄悄失真. 生产实现一变, 本脚本
    的 v3 列跟着变, 这正是想要的行为.
    """
    return _tokenize(text)


VERSIONS: List[tuple[str, str, TokenizeFn]] = [
    ("v1", "汉字整段", tokenize_v1),
    ("v2", "中文2-gram", tokenize_v2),
    ("v3", "jieba词级", tokenize_v3),
]


# ---------------------------------------------------------------------------
# 语料 / 数据集
# ---------------------------------------------------------------------------

def _load_corpus(path: Path) -> str:
    """读语料并走与上传链路相同的清洗, 保证评估文本与线上入库文本一致."""
    from app.core.config import settings
    from app.rag.cleaner import DocumentCleaner

    return DocumentCleaner(enabled=settings.cleaner_enabled).clean(path.read_text(encoding="utf-8"))


def _load_dataset(path: Path) -> List[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build_docs(text: str, chunk_kwargs: dict) -> List[str]:
    """按给定粒度切分语料, 返回 chunk 文本列表 (三版本共用同一份, 变量只有 tokenizer)."""
    from app.rag.chunker import TextChunker

    return [c.text for c in TextChunker(**chunk_kwargs).split(text, doc_id="tokenizer_ablation")]


# ---------------------------------------------------------------------------
# 口径 1: 词表交集
# ---------------------------------------------------------------------------

def vocab_probe(
    queries: Sequence[str],
    corpus_tokens: Sequence[Sequence[str]],
    fn: TokenizeFn,
) -> Dict[str, object]:
    """query 的 token 集合与语料词表是否有交集.

    无交集 => BM25 对这条 query 恒为 0 分. 这是 v1 最致命的地方: 不是"排得不好",
    而是"根本没进入候选". 分纯中文 / 全部两个子集统计 —— 纯中文那一档最能说明
    "连续汉字串"这个设计错在哪.
    """
    vocab = set(chain.from_iterable(corpus_tokens))
    rows = []
    for q in queries:
        rows.append((bool(set(fn(q)) & vocab), bool(_ASCII_ALNUM.search(q))))

    def _rate(subset: List[tuple[bool, bool]]) -> Dict[str, object]:
        hit = sum(1 for h, _ in subset if h)
        return {"hit": hit, "total": len(subset), "rate": round(hit / len(subset), 4) if subset else 0.0}

    return {
        "all": _rate(rows),
        "pure_zh": _rate([r for r in rows if not r[1]]),
        "with_ascii": _rate([r for r in rows if r[1]]),
    }


# ---------------------------------------------------------------------------
# 口径 2: BM25 单路 Recall@K
# ---------------------------------------------------------------------------

def bm25_probe(
    docs: Sequence[str],
    dataset: Sequence[dict],
    fn: TokenizeFn,
    top_k: int,
    ks: Sequence[int],
) -> Dict[str, object]:
    """用 fn 建 BM25 索引, 逐条 query 取 top_k, 算 Recall@K 与零分统计.

    零分统计存在的理由: docstring 记的是 "打分为 0 后被 `if sc[i] > 0` 过滤掉,
    这一路召回等于不存在". 该过滤在当前 BM25Store.query 里**已经没有了**, 所以
    这里把两种口径都报出来 —— 不过滤时 v1 会靠 0 分候选凑满 top_k, 看着像"召回了
    东西"; 一旦按历史口径过滤, 这些 query 的返回直接是空列表. 差异有多大, 看
    `zero_ratio` 与 `empty_after_filter` 两列.
    """
    from rank_bm25 import BM25Okapi
    from app.rag.evaluate import recall_at_k_by_context

    bm25 = BM25Okapi([fn(d) for d in docs])
    n = len(docs)

    sum_recall = {k: 0.0 for k in ks}
    n_scored = 0
    n_no_answer = 0
    zero_ratio_sum = 0.0
    empty_after_filter = 0
    per_question: List[dict] = []

    for item in dataset:
        q = item.get("question", "")
        refs = [r for r in (item.get("reference_context") or []) if r and r.strip()]

        scores = bm25.get_scores(fn(q))
        order = sorted(range(n), key=lambda i: scores[i], reverse=True)[:top_k]
        texts = [docs[i] for i in order]

        # top_k 候选里有多少条是"零分/负分" —— 负分来自 BM25Okapi 的 IDF 修正,
        # 同样意味着"这个词在本语料里没有区分度", 跟 0 分归为一类.
        top_scores = [scores[i] for i in order]
        zero_cnt = sum(1 for s in top_scores if s <= 0)
        zero_ratio_sum += zero_cnt / max(len(top_scores), 1)
        if zero_cnt == len(top_scores):
            empty_after_filter += 1

        if not refs:
            # 答案本就不在语料内: 不计入召回统计, 否则指标被刷虚高.
            n_no_answer += 1
            per_question.append({"question": q, "no_answer": True})
            continue

        row: dict = {"question": q, "recall": {}}
        for k in ks:
            r = recall_at_k_by_context(refs, texts, k)
            sum_recall[k] += r
            row["recall"][f"@{k}"] = round(r, 4)
        per_question.append(row)
        n_scored += 1

    denom = n_scored or 1
    return {
        "recall": {f"@{k}": round(sum_recall[k] / denom, 4) for k in ks},
        "n_scored": n_scored,
        "n_no_answer": n_no_answer,
        "zero_ratio": round(zero_ratio_sum / max(len(dataset), 1), 4),
        "empty_after_filter": empty_after_filter,
        "per_question": per_question,
    }


# ---------------------------------------------------------------------------
# 口径 3: 分词开销
# ---------------------------------------------------------------------------

def cost_probe(
    corpus_tokens: Sequence[Sequence[str]],
    queries: Sequence[str],
    fn: TokenizeFn,
) -> Dict[str, object]:
    total = sum(len(t) for t in corpus_tokens)
    unseen = sum(1 for t in chain.from_iterable(corpus_tokens) if not t.strip())
    return {
        "corpus_tokens": total,
        "vocab_size": len(set(chain.from_iterable(corpus_tokens))),
        "avg_doc_tokens": round(total / max(len(corpus_tokens), 1), 1),
        "avg_query_tokens": round(sum(len(fn(q)) for q in queries) / max(len(queries), 1), 1),
        "blank_tokens": unseen,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def _fmt_int(v) -> str:
    return f"{v:,}"


def _print_tables(results: List[dict], meta: dict, ks: Sequence[int]) -> None:
    print(f"\n语料: {meta['corpus']}  |  切分: {meta['chunk_kwargs']}  |  "
          f"chunk 数: {meta['chunk_count']}  |  问题: {meta['n_questions']} 条 "
          f"(纯中文 {meta['n_pure_zh']} / 含ASCII {meta['n_questions'] - meta['n_pure_zh']})  |  "
          f"top_k={meta['top_k']}")

    print("\n=== 表 1: 词表交集 (query token 与语料词表是否有交集) ===")
    print(f"  {'版本':<4}{'策略':<14}{'纯中文命中':>12}{'纯中文率':>10}{'含ASCII命中':>13}{'全部命中':>10}")
    for r in results:
        v = r["vocab"]
        pz, asc, allv = v["pure_zh"], v["with_ascii"], v["all"]
        print(f"  {r['version']:<4}{r['strategy']:<14}"
              f"{str(pz['hit']) + '/' + str(pz['total']):>12}"
              f"{pz['rate'] * 100:>9.1f}%"
              f"{str(asc['hit']) + '/' + str(asc['total']):>13}"
              f"{allv['rate'] * 100:>9.1f}%")

    kcols = "".join(f"{'Recall@' + str(k):>12}" for k in ks)
    print("\n=== 表 2: BM25 单路召回 ===")
    print(f"  {'版本':<4}{'策略':<14}{kcols}{'零分占比':>11}{'过滤后全空':>12}")
    for r in results:
        b = r["bm25"]
        cells = "".join(f"{b['recall'][f'@{k}']:>12.3f}" for k in ks)
        print(f"  {r['version']:<4}{r['strategy']:<14}{cells}"
              f"{b['zero_ratio'] * 100:>10.1f}%"
              f"{str(b['empty_after_filter']) + '/' + str(meta['n_questions']):>12}")

    print("\n=== 表 3: 分词开销 ===")
    print(f"  {'版本':<4}{'策略':<14}{'语料tokens':>12}{'词表大小':>10}{'均/chunk':>10}{'均/query':>10}")
    for r in results:
        c = r["cost"]
        print(f"  {r['version']:<4}{r['strategy']:<14}"
              f"{_fmt_int(c['corpus_tokens']):>12}{_fmt_int(c['vocab_size']):>10}"
              f"{c['avg_doc_tokens']:>10}{c['avg_query_tokens']:>10}")

    scored = results[0]["bm25"]["n_scored"] if results else 0
    no_ans = results[0]["bm25"]["n_no_answer"] if results else 0
    print(f"\n  参与召回统计 {scored} 条; 无答案样本 {no_ans} 条 (不计入, 否则刷虚高)")
    print("  注: '过滤后全空' = top_k 候选全部为 0/负分, 即按历史 `if sc[i] > 0` 口径该 query 返回空列表.")


# ---------------------------------------------------------------------------
# 切分粒度扫描
# ---------------------------------------------------------------------------

# 200~1200 覆盖"v1 看着还行"到"v1 明显崩"的全区间, 专门用来看指标对粒度的敏感度.
# 步长不取整百是为了避开 500/1000 这类"看着像默认值"的巧合, 让基准值 300 更醒目.
SWEEP_SIZES = (200, 300, 500, 800, 1200)


def _run_sweep(text: str, dataset: Sequence[dict], top_k: int, ks: Sequence[int]) -> List[dict]:
    """同一份语料, 只改 chunk_size, 看三版本的 Recall@5 各自怎么变."""
    k_key = "@5" if 5 in ks else f"@{ks[len(ks) // 2]}"
    rows: List[dict] = []
    for cs in SWEEP_SIZES:
        docs = _build_docs(text, dict(chunk_size=cs, chunk_overlap=cs // 6))
        if not docs:
            logger.warning(f"chunk_size={cs} 切分后为空, 跳过")
            continue
        row: dict = {"chunk_size": cs, "chunk_count": len(docs), "n_questions": len(dataset)}
        for version, _, fn in VERSIONS:
            r = bm25_probe(docs, dataset, fn, top_k, ks)
            row[version] = {
                "recall5": r["recall"][k_key],
                "zero_ratio": r["zero_ratio"],
                "empty_after_filter": r["empty_after_filter"],
            }
        rows.append(row)
    return rows


def _print_sweep(rows: Sequence[dict]) -> None:
    print("\n=== 表 4: 切分粒度敏感性 (同一份语料, 只改 chunk_size) ===")
    print(f"  {'chunk':>6}{'chunks':>8}{'v1@5':>9}{'v2@5':>9}{'v3@5':>9}{'v1零分':>9}{'v1过滤后全空':>14}")
    for r in rows:
        print(f"  {r['chunk_size']:>6}{r['chunk_count']:>8}"
              f"{r['v1']['recall5']:>9.3f}{r['v2']['recall5']:>9.3f}{r['v3']['recall5']:>9.3f}"
              f"{r['v1']['zero_ratio'] * 100:>8.1f}%"
              f"{str(r['v1']['empty_after_filter']) + '/' + str(r['n_questions']):>14}")
    print("  v1@5 随 chunk 变大而涨, 但 v1 的零分占比恒在 90% 上下 —— 涨的不是检索质量,")
    print("  是 chunk 覆盖面: chunk 越大, top-5 越容易蒙中答案所在的那一段.")
    print("  所以评测分词改动时, 粒度必须钉死; 拿大 chunk 的结果说'v1 也还行'是错的.")


def main() -> None:
    ap = argparse.ArgumentParser(description="_tokenize 三代实现消融对比")
    ap.add_argument("--corpus", type=Path, default=_DATA / "eval" / "corpus" / "企业知识库演示语料.md")
    ap.add_argument("--dataset", type=Path, default=_DATA / "eval" / "recall_dataset.jsonl")
    ap.add_argument("--chunk-size", type=int, default=300,
                    help="切分粒度. 默认 300 = docstring 那批数据的复现基准 (生产默认是 500)")
    ap.add_argument("--chunk-overlap", type=int, default=None, help="默认 chunk_size//6, 与 run_recall_eval 一致")
    ap.add_argument("--top-k", type=int, default=20, help="每条 query 取多少候选 (需 >= 最大 K)")
    ap.add_argument("--sweep", action="store_true", help="额外汇总表 4: 扫多个切分粒度下三版本的 @5")
    ap.add_argument("--out", type=Path, default=None, help="把结果写成 JSON")
    args = ap.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"语料不存在: {args.corpus}")
    if not args.dataset.exists():
        raise SystemExit(f"数据集不存在: {args.dataset}")

    ks = (1, 3, 5, 10)
    if args.top_k < max(ks):
        raise SystemExit(f"--top-k={args.top_k} 小于最大 K={max(ks)}, 指标会被截断")

    text = _load_corpus(args.corpus)
    chunk_kwargs = dict(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap if args.chunk_overlap is not None else args.chunk_size // 6,
    )
    docs = _build_docs(text, chunk_kwargs)
    if not docs:
        raise SystemExit(f"语料切分后为空: {args.corpus}")

    dataset = _load_dataset(args.dataset)
    queries = [d.get("question", "") for d in dataset]
    n_pure_zh = sum(1 for q in queries if not _ASCII_ALNUM.search(q))

    logger.info(f"语料 {len(text)} 字符 -> {len(docs)} chunks; 数据集 {len(dataset)} 条")

    results = []
    for version, strategy, fn in VERSIONS:
        logger.info(f"--- {version} ({strategy}) ---")
        corpus_tokens = [fn(d) for d in docs]
        results.append({
            "version": version,
            "strategy": strategy,
            "vocab": vocab_probe(queries, corpus_tokens, fn),
            "bm25": bm25_probe(docs, dataset, fn, args.top_k, ks),
            "cost": cost_probe(corpus_tokens, queries, fn),
        })

    meta = {
        "corpus": args.corpus.name,
        "corpus_chars": len(text),
        "chunk_count": len(docs),
        "chunk_kwargs": chunk_kwargs,
        "dataset": args.dataset.name,
        "n_questions": len(dataset),
        "n_pure_zh": n_pure_zh,
        "top_k": args.top_k,
        "ks": list(ks),
    }
    _print_tables(results, meta, ks)

    sweep = None
    if args.sweep:
        logger.info("--- 切分粒度扫描 ---")
        sweep = _run_sweep(text, dataset, args.top_k, ks)
        _print_sweep(sweep)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "meta": meta,
            "results": results,
            "sweep": sweep,
            "note": (
                "v1=连续汉字串整段, v2=中文2-gram(现 jieba 缺失回退分支), v3=当前生产实现. "
                "只测 BM25 单路, 不含 embedding / RRF. Recall@K 按 reference_context 文本匹配 "
                "(阈值 0.7) 计算, 不依赖 chunk_id."
            ),
        }
        args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n报告已写入: {args.out}")


if __name__ == "__main__":
    main()
