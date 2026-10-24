"""离线检索评估 (Recall@K / Precision@K / MRR@K).

RAG 链路调优离不开**可量化的检索指标**. 这里提供:
  - `recall_at_k(relevant, retrieved, k)`: 单条问答的 Recall@K.
  - `RecallEvaluator`: 在一份 (question, relevant_chunk_ids) 数据集上,
    跑 retriever 统计 Recall@1 / @3 / @5 / @10, 用来对比
    "加/不加清洗、自适应切分、父文档召回、重排优化" 前后的差异.
  - `precision_at_k_by_context` / `reciprocal_rank_by_context`: 文本口径的
    Precision@K 与 RR@K. 光有 Recall 会把调优往"多召回"带偏 —— Precision
    管 top-k 里的垃圾率, MRR 管首个命中排在多前.
  - `paired_bootstrap_ci`: 两配置逐题指标的配对 bootstrap 置信区间,
    用来判断对比表里的分差是真实差异还是几十条样本上的噪音.

`retriever.retrieve` 返回的是 dict (含 id), 这里统一按 `id` 对齐.
数据集格式 (JSONL / 列表):
    {"question": "...", "relevant_ids": ["doc_x_c0", "doc_x_c3"]}
"""
from __future__ import annotations

import json
import random
import re
from difflib import SequenceMatcher
from typing import Callable, Dict, List, Optional, Sequence


def recall_at_k(relevant: Sequence[str], retrieved: Sequence[str], k: int) -> float:
    """Recall@K = |relevant ∩ retrieved[:k]| / |relevant|.

    relevant 为空时定义 Recall=1.0 (无相关项可漏).
    """
    if not relevant:
        return 1.0
    if k <= 0:
        return 0.0
    rel_set = set(relevant)
    # 必须按**集合交集**算命中, 不能逐条累加:
    # 父文档召回 (连坐) 会让同一个 chunk id 在候选里出现多次, 逐条累加会把
    # Recall 算到 > 1.0, 直接把评估指标刷虚高.
    top = list(retrieved)[:k]
    hits = len(rel_set & set(top))
    return hits / len(rel_set)


class RecallEvaluator:
    """在数据集上评估 retriever 的 Recall@K."""

    DEFAULT_KS = (1, 3, 5, 10)

    def __init__(self, ks: Sequence[int] = DEFAULT_KS):
        self.ks = tuple(ks)

    def evaluate(
        self,
        dataset: List[dict],
        retrieve_fn: Callable[[str], Sequence[dict]],
        id_key: str = "id",
    ) -> Dict[str, object]:
        """retrieve_fn(question) -> list of dict (含 id_key).

        返回 {recall@k: float, per_question: [...], n: int}.
        """
        per_question: List[dict] = []
        sum_recall = {k: 0.0 for k in self.ks}
        for item in dataset:
            q = item.get("question", "")
            relevant = item.get("relevant_ids", []) or item.get("relevant", [])
            retrieved = retrieve_fn(q)
            ids = [r.get(id_key) for r in retrieved if r.get(id_key)]
            row = {"question": q, "recall": {}}
            for k in self.ks:
                r = recall_at_k(relevant, ids, k)
                sum_recall[k] += r
                row["recall"][f"@{k}"] = round(r, 4)
            per_question.append(row)
        n = len(dataset) or 1
        return {
            "n": len(dataset),
            "recall": {f"@{k}": round(sum_recall[k] / n, 4) for k in self.ks},
            "per_question": per_question,
        }

    @staticmethod
    def load_dataset(path: str) -> List[dict]:
        """读取 JSONL 或 JSON 数组格式的评估集."""
        with open(path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)
        out = []
        for line in text.splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out


# ---------------------------------------------------------------------------
# 按文本匹配的召回评估
#
# 上面那套按 chunk_id 对齐, 前提是 id 稳定. 但上传时 doc_id = f"doc_{uuid4().hex[:12]}"
# (见 api/documents.py), 每次重新入库都会变, chunk_id 随之变化 —— 写死 id 的评估集
# 换一次库就全失效. 这里改成按**内容文本**对齐: 评估集只记答案所在的原文片段,
# 召回结果里只要有 chunk 覆盖了该片段就算命中.
# ---------------------------------------------------------------------------

_MD_MARKS = re.compile(r"\*\*|__|`{1,3}")
_HEAD_MARK = re.compile(r"^#{1,6}\s*", re.M)
_LIST_MARK = re.compile(r"^\s*[-*+]\s+", re.M)


def normalize_for_match(text: str) -> str:
    """归一化文本用于比对: 去 markdown 标记 / 表格竖线 / 全部空白.

    为什么要去标记: 标注写的是"人眼看到的文本", 语料里却可能带 **加粗** 与表格 |,
    清洗器入库时同样会去掉它们. 两边归一化到同一形态才能稳定比对.
    """
    s = _MD_MARKS.sub("", text or "")
    s = _HEAD_MARK.sub("", s)
    s = _LIST_MARK.sub("", s)
    s = s.replace("|", " ")
    return re.sub(r"\s+", "", s)


def context_coverage(reference: str, chunk_text: str) -> float:
    """reference 的内容有多大比例被 chunk_text 覆盖 (最长公共子串口径).

    不用"完整包含"是因为切分边界可能把片段截到相邻 chunk; 也不看字符集重合度,
    那对语序完全不敏感, 会把不相干的 chunk 也判成命中.
    """
    ref = normalize_for_match(reference)
    if not ref:
        return 0.0
    chunk = normalize_for_match(chunk_text)
    if not chunk:
        return 0.0
    if ref in chunk:
        return 1.0
    sm = SequenceMatcher(None, ref, chunk, autojunk=False)
    return sm.find_longest_match(0, len(ref), 0, len(chunk)).size / len(ref)


def recall_at_k_by_context(
    reference_context: Sequence[str],
    retrieved_texts: Sequence[str],
    k: int,
    threshold: float = 0.7,
) -> float:
    """Recall@K (文本口径) = 被前 k 个召回覆盖的 reference 片段占比.

    与 recall_at_k 同理, 必须按**集合覆盖**算而非逐条累加: 父文档召回会让同一段
    文本在候选里出现多次, 逐条累加会把指标刷到 > 1.0.
    """
    refs = [r for r in (reference_context or []) if r and r.strip()]
    if not refs:
        return 1.0  # 无标注 (答案本就不在语料内) 的样本不拉低召回
    if k <= 0:
        return 0.0
    top = list(retrieved_texts)[:k]
    if not top:
        return 0.0
    hit = sum(1 for r in refs if any(context_coverage(r, t) >= threshold for t in top))
    return hit / len(refs)


def precision_at_k_by_context(
    reference_context: Sequence[str],
    retrieved_texts: Sequence[str],
    k: int,
    threshold: float = 0.7,
) -> float:
    """Precision@K (文本口径) = 前 k 个召回里覆盖了某条 reference 的 chunk 占比.

    覆盖判定与 recall_at_k_by_context 完全一致, 方向相反: recall 问"该召回的
    都召回了吗", precision 问"top-k 里有多少是垃圾".

    分母取 min(k, len(top)): 检索库可能返回不足 k 条, 直接除以 k 会白扣分.
    reference 为空时返回 0.0, 与 recall 的 1.0 恰好相反 —— 无相关项时召回无
    东西可漏, 但检索到的任何东西都不相关. 评估器层面会把这类"答案不在语料内"
    的样本整个排除, 不让这个硬口径拉偏平均.
    """
    refs = [r for r in (reference_context or []) if r and r.strip()]
    if k <= 0 or not refs:
        return 0.0
    top = list(retrieved_texts)[:k]
    if not top:
        return 0.0
    hit = sum(1 for t in top if any(context_coverage(r, t) >= threshold for r in refs))
    return hit / min(k, len(top))


def reciprocal_rank_by_context(
    reference_context: Sequence[str],
    retrieved_texts: Sequence[str],
    k: int,
    threshold: float = 0.7,
) -> float:
    """RR@K (文本口径) = 首个命中 chunk 的倒数排名, top-k 内未命中则 0.

    MRR@K 即它在数据集上的均值. 相比 Recall@K 多携带"排多前"的信息:
    同样的 Recall@10, 命中在第 1 位和第 8 位对下游生成完全不是一回事.
    """
    refs = [r for r in (reference_context or []) if r and r.strip()]
    if not refs or k <= 0:
        return 0.0
    for rank, t in enumerate(list(retrieved_texts)[:k], 1):
        if any(context_coverage(r, t) >= threshold for r in refs):
            return 1.0 / rank
    return 0.0


class ContextRecallEvaluator:
    """在 (question, reference_context) 数据集上评估 Recall@K / Precision@K / MRR@K.

    与 RecallEvaluator 的区别: 后者按 chunk_id 对齐 (要求 id 稳定), 本类按原文片段
    文本对齐, 不依赖 doc_id / uuid, 因此**重新入库后评估集依然可用**.
    (类名沿用历史叫法, 报告内容已覆盖三族指标.)
    """

    DEFAULT_KS = (1, 3, 5, 10)

    def __init__(self, ks: Sequence[int] = DEFAULT_KS, threshold: float = 0.7):
        self.ks = tuple(ks)
        self.threshold = threshold

    def evaluate(
        self,
        dataset: List[dict],
        retrieve_fn: Callable[[str], Sequence[dict]],
        text_key: str = "document",
    ) -> Dict[str, object]:
        """retrieve_fn(question) -> 候选列表 (每项含 text_key 指定的文本字段).

        返回 {recall/precision/mrr: {"@k": ...}, n: 参与统计的样本数,
              n_no_answer: 无答案样本数, per_question: [...]}.
        per_question 行里的 "mrr" 是该题的 RR (按各 k 截止), 数据集均值即 MRR@k.

        覆盖判定与 *_by_context 纯函数同口径, 但把 (reference x chunk) 命中矩阵
        只算一次, 三族指标全从矩阵推导 —— context_coverage 里的 SequenceMatcher
        是最贵的一步, 逐 K 重复比对等于白跑 4 倍.
        """
        per_question: List[dict] = []
        sum_recall = {k: 0.0 for k in self.ks}
        sum_precision = {k: 0.0 for k in self.ks}
        sum_rr = {k: 0.0 for k in self.ks}
        n_scored = 0
        n_no_answer = 0

        for item in dataset:
            q = item.get("question", "")
            refs = item.get("reference_context") or item.get("relevant_context") or []
            refs = [r for r in refs if r and r.strip()]
            if not refs:
                # 答案不在语料内的问题不参与召回统计, 否则会把指标刷虚高;
                # 但仍记录一行, 便于人工核对检索是否也"什么都没召回到".
                n_no_answer += 1
                per_question.append(
                    {
                        "question": q,
                        "no_answer": True,
                        "recall": {},
                        "precision": {},
                        "mrr": {},
                        "retrieved": len(retrieve_fn(q)),
                    }
                )
                continue

            texts = [r.get(text_key, "") for r in retrieve_fn(q)]
            hits = [
                [context_coverage(ref, t) >= self.threshold for t in texts]
                for ref in refs
            ]
            chunk_hit = [any(hr[j] for hr in hits) for j in range(len(texts))]
            first_hit_rank = next((j + 1 for j, h in enumerate(chunk_hit) if h), None)

            row: dict = {"question": q, "recall": {}, "precision": {}, "mrr": {}}
            for k in self.ks:
                top = min(k, len(texts))
                rec = sum(any(h[:top]) for h in hits) / len(refs)
                prec = sum(chunk_hit[:top]) / top if top else 0.0
                rr = 1.0 / first_hit_rank if first_hit_rank and first_hit_rank <= k else 0.0
                sum_recall[k] += rec
                sum_precision[k] += prec
                sum_rr[k] += rr
                row["recall"][f"@{k}"] = round(rec, 4)
                row["precision"][f"@{k}"] = round(prec, 4)
                row["mrr"][f"@{k}"] = round(rr, 4)
            per_question.append(row)
            n_scored += 1

        n = n_scored or 1
        return {
            "n": n_scored,
            "n_no_answer": n_no_answer,
            "recall": {f"@{k}": round(sum_recall[k] / n, 4) for k in self.ks},
            "precision": {f"@{k}": round(sum_precision[k] / n, 4) for k in self.ks},
            "mrr": {f"@{k}": round(sum_rr[k] / n, 4) for k in self.ks},
            "per_question": per_question,
        }

    @staticmethod
    def load_dataset(path: str) -> List[dict]:
        """读取 JSONL 或 JSON 数组格式的评估集 (与 RecallEvaluator 同格式)."""
        return RecallEvaluator.load_dataset(path)


# ---------------------------------------------------------------------------
# 配对 bootstrap 显著性
#
# --compare 的各配置跑在同一批问题上, 裸分差没有"多少算显著"的信息:
# 几十条样本上 Recall@5 差 0.05 很可能只是噪音. 逐题差分先消掉题目难度,
# 再对差分均值 bootstrap, CI 跨不跨 0 就是判断标准.
# ---------------------------------------------------------------------------

def paired_bootstrap_ci(
    a: Sequence[float],
    b: Sequence[float],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    """同一批问题在两个配置下的逐题指标, 配对 bootstrap 出均值差的置信区间.

    返回 (a-b 平均差, CI 下界, CI 上界), 置信水平 1-alpha;
    CI 完全落在 0 的同一侧即认为差异显著. seed 固定 —— CI 可复现,
    才能写进对比结论, 否则两次跑出两个说法.
    """
    if len(a) != len(b) or not a:
        raise ValueError("a/b 必须是同一批问题的等长非空逐题指标数组")
    diffs = [x - y for x, y in zip(a, b)]
    n = len(diffs)
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(alpha / 2 * n_boot)]
    hi = means[min(n_boot - 1, int((1 - alpha / 2) * n_boot))]
    return sum(diffs) / n, lo, hi


if __name__ == "__main__":
    import sys

    ds_path = sys.argv[1] if len(sys.argv) > 1 else "eval/recall_dataset.jsonl"
    data = RecallEvaluator.load_dataset(ds_path)
    print(f"loaded {len(data)} qa pairs from {ds_path}")
    print(json.dumps({"ks": RecallEvaluator.DEFAULT_KS}, ensure_ascii=False, indent=2))
