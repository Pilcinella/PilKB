"""In-memory BM25 store (rank_bm25). 适合小到中等规模。生产可换 Elasticsearch / OpenSearch。"""
from __future__ import annotations

import pickle
import re
import threading
from pathlib import Path
from typing import List

from loguru import logger


try:  # jieba 是可选的: 装了走词级切分, 没装自动回退到 2-gram
    import jieba as _jieba
except ImportError:  # pragma: no cover
    _jieba = None

_SEG_RE = re.compile(r"[一-龥]+|[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    """中英混合分词: 中文走 jieba 词级切分, 英文/数字按整词.

    演进过程 (每一步都有实测数据支撑, 别退回旧实现):
      1) 最初是 `re.findall(r"[一-龥]+|[a-z0-9]+")`. `[一-龥]+` 匹配的是**连续汉字串**,
         而中文句子内部没有分隔符, 于是"专业版的保修期是多久"整句变成 1 个 token,
         与任何 chunk 都对不上 —— BM25 打分为 0 后被 `if sc[i] > 0` 过滤掉, 这一路
         召回等于不存在. 实测 62 条查询里 47 条纯中文问题与语料零交集(命中率 0/47),
         能命中的 15 条全靠数字/英文/空格侥幸切开(如 "买 11 个..." 里的 "11").
      2) 改成中文 2-gram: 交集率 24% -> 98%, BM25 单路 Recall@5 从 0.228 升到 0.983,
         混合检索这才真正优于任一单路.
      3) 现在换成 jieba 词级切分: 避免 2-gram 产生 "的保" 这类无意义组合, 且 token
         数减半(索引更小、打分更快), 匹配也更精确.

    英文/数字段不交给 jieba 而由正则整词取出: jieba 会把 "bge-m3"、"p0" 这类切成
    碎片, 而它们恰恰是精确检索最需要的 token.

    jieba 缺失时回退到 2-gram: 召回略降但链路不会挂 —— 回退分支主要给精简环境兜底.
    """
    out: List[str] = []
    for seg in _SEG_RE.findall((text or "").lower()):
        if seg[0].isascii():
            out.append(seg)                                        # 英文/数字: 整词
        elif _jieba is not None:
            out.extend(t for t in _jieba.lcut(seg) if t.strip())   # 中文: 词级切分
        elif len(seg) == 1:
            out.append(seg)
        else:
            out.extend(seg[i : i + 2] for i in range(len(seg) - 1))  # 回退: 2-gram
    return out


class BM25Store:
    def __init__(self, persist_path: str | None = None):
        self.persist_path = persist_path
        self._bm25 = None
        self.docs: List[dict] = []  # 每条: {id, text, metadata}
        # 惰性重建: 写入/删除只动 docs + 置 dirty=True, 不立即 rebuild.
        # 首次 query() 时若 dirty 才重建. 大幅降低 N 次上传的 CPU 开销.
        self._dirty: bool = False
        # 重建成本较高, 用锁串行化 (只允许一个线程在 rebuild).
        self._rebuild_lock = threading.Lock()
        # 重建节流: 阈值以下不重建; 让 query 直接返回空 (避免 1 条文档都触发重建).
        # 设为 1 (默认有内容才建).
        self._min_docs_to_build: int = 1

    def add(self, ids: List[str], documents: List[str], metadatas: List[dict]):
        """追加 doc + 标 dirty. 不立即重建, query 时按需重建."""
        for i, d in enumerate(documents):
            self.docs.append({"id": ids[i], "text": d, "metadata": metadatas[i]})
        self._dirty = True
        self._maybe_persist()

    def delete(self, ids: List[str]) -> int:
        """按 id 删除. 返回实际删除的条数. 不存在的不报错."""
        if not ids:
            return 0
        ids_set = set(ids)
        kept: List[dict] = []
        removed = 0
        for d in self.docs:
            if d["id"] in ids_set:
                removed += 1
                continue
            kept.append(d)
        if removed:
            self.docs = kept
            self._dirty = True
            self._maybe_persist()
        return removed

    def query(self, text: str, top_k: int = 20) -> List[dict]:
        # 用局部快照而不是反复读 self.docs: delete() 是 self.docs = kept (重新绑定),
        # 快照仍指向索引建立时那个 list, 长度与 _bm25 的语料一致.
        # 旧实现直接用 self.docs[idx], 与 delete 并发时 docs 已被截短, 而 scores
        # 还是旧长度 -> IndexError (上传/删除与查询并发时的偶发 500).
        docs = self.docs
        if not docs:
            return []
        self._ensure_index()
        if not self._bm25:
            return []
        tokens = _tokenize(text)
        scores = self._bm25.get_scores(tokens)
        # add() 是原地 append, 会让快照变长; 这里按两者较小值收敛, 保证 idx 合法.
        n = min(len(scores), len(docs))
        ranked = sorted(enumerate(scores[:n]), key=lambda x: x[1], reverse=True)[:top_k]
        out = []
        for idx, score in ranked:
            d = docs[idx]
            out.append({
                "id": d["id"],
                "document": d["text"],
                "metadata": d["metadata"],
                "score": float(score),
            })
        return out

    def count(self) -> int:
        return len(self.docs)

    def rebuild(self, force: bool = False) -> None:
        """显式重建 (例如大批量入库后想马上能 query). 强制覆盖 dirty."""
        with self._rebuild_lock:
            self._do_build()
            self._dirty = False

    def _ensure_index(self) -> None:
        """仅在 dirty 时重建, 锁内串行."""
        if not self._dirty:
            return
        with self._rebuild_lock:
            # 双重检查: 上一个等待者可能已经重建完.
            if not self._dirty:
                return
            if len(self.docs) < self._min_docs_to_build:
                self._bm25 = None
                self._dirty = False
                return
            self._do_build()
            self._dirty = False

    def _do_build(self) -> None:
        from rank_bm25 import BM25Okapi
        tokenized_corpus = [_tokenize(d["text"]) for d in self.docs]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(f"BM25 index built: {len(self.docs)} docs")

    def _maybe_persist(self):
        if not self.persist_path:
            return
        try:
            Path(self.persist_path).parent.mkdir(parents=True, exist_ok=True)
            # 只持久化 docs + dirty 标记; _bm25 反序列化时按需重建.
            with open(self.persist_path, "wb") as f:
                pickle.dump({"docs": self.docs}, f)
        except Exception as e:
            logger.warning(f"BM25 persist failed: {e}")

    def load(self):
        if not self.persist_path or not Path(self.persist_path).exists():
            return
        try:
            with open(self.persist_path, "rb") as f:
                data = pickle.load(f)
            self.docs = data["docs"]
        except Exception as e:  # noqa: BLE001
            # 旧实现没有 try, 异常一路抛到 HybridRetriever.__init__ 里的
            # `except Exception: pass` 被静默吞掉: 索引文件明明存在却查不到东西,
            # 而且下次 _maybe_persist() 会用空 docs 覆盖它 -> 数据永久丢失.
            # 这里改成明确报错 + 隔离损坏文件, 保留事后恢复的可能.
            logger.error(f"BM25 load failed ({self.persist_path}), starting empty: {e}")
            self.docs = []
            self._quarantine_corrupt_file()
        self._dirty = True  # 启动后第一次 query 触发重建

    def _quarantine_corrupt_file(self) -> None:
        """把无法解析的持久化文件改名旁路, 避免被空数据覆盖."""
        if not self.persist_path:
            return
        try:
            src = Path(self.persist_path)
            backup = src.with_name(src.name + ".corrupt")
            src.replace(backup)
            logger.error(f"BM25 已隔离损坏文件 -> {backup}")
        except Exception as e:  # noqa: BLE001
            logger.warning(f"BM25 隔离损坏文件失败 (原文件仍保留): {e}")