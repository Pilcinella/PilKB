"""混合检索：BM25 + 向量 + RRF 融合。"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Dict, List

from app.core.config import settings
from app.rag.bm25_store import BM25Store
from app.rag.embedding import get_embedding_service
from app.rag.vector_store import ChromaStore


# 进程内 HybridRetriever 实例缓存 (按 kb_id).
# 同一 kb 多次请求复用同一 ChromaStore/BM25Store, 避免每次新建 collection/重建 BM25.
_INSTANCE_CACHE: Dict[str, "HybridRetriever"] = {}
_INSTANCE_LOCK = threading.Lock()


def get_retriever(kb_id: str = "default") -> "HybridRetriever":
    """获取/创建 HybridRetriever 实例. 同 kb_id 永远返回同一实例.

    线程安全: 用 _INSTANCE_LOCK 串行化创建, 避免 double-init race.
    测试时可调 reset_retriever_cache() 清空.
    """
    with _INSTANCE_LOCK:
        r = _INSTANCE_CACHE.get(kb_id)
        if r is not None:
            return r
        r = HybridRetriever(kb_id=kb_id)
        _INSTANCE_CACHE[kb_id] = r
        return r


def reset_retriever_cache() -> None:
    """仅供测试: 清空实例缓存."""
    with _INSTANCE_LOCK:
        _INSTANCE_CACHE.clear()


def drop_retriever(kb_id: str) -> None:
    """逐出某个 kb 的缓存实例 (删除知识库时调用).

    必须在删盘**之前**逐出: 实例里的 BM25Store 持有整表内存态, 先删 pkl 再
    逐出的话, 残留实例的下一次 _maybe_persist() 会把刚删的文件原样写回来。
    """
    with _INSTANCE_LOCK:
        _INSTANCE_CACHE.pop(kb_id, None)


def delete_kb_storage(kb_id: str) -> None:
    """删除一个 kb 的全部持久化检索数据 (向量库 + BM25), 并逐出缓存实例.

    DELETE /api/v1/kbs 的执行体 (注册表条目由 kb_registry.remove_kb 处理).
    上传的源文件不动: 它们按 doc_id 命名、不按 kb 归档, 与单文档删除接口
    逐 doc 清理的口径互补 —— 删库后残留的孤儿源文件不影响任何检索行为。
    """
    drop_retriever(kb_id)
    backend = getattr(settings, "vector_store_backend", "chroma")
    if backend == "numpy":
        from app.rag.numpy_vector_store import NumpyVectorStore

        NumpyVectorStore(collection_name=f"kb_{kb_id}").reset()
    else:
        ChromaStore(collection_name=f"kb_{kb_id}").reset()
    Path(settings.bm25_persist_dir).joinpath(f"bm25_{kb_id}.pkl").unlink(missing_ok=True)


def check_store_consistency(kb_id: str) -> Dict[str, object]:
    """比较同一 kb 下 BM25 与向量库的 chunk 数, 供启动自检使用.

    两者不一致意味着 BM25 的内存态已与向量库脱钩. 此时链路**不会报错**:
    检索照常返回结果 (BM25 路静默为空, 退化为纯向量检索), 接口全是 200, 只是
    `expand_parent_recall` 不再补兄弟 chunk、上下文缺段 —— 属于"答案似是而非、
    排查成本极高"的那类问题. 所以在启动时把它明确暴露出来.

    典型成因 (见 Dockerfile 的 --workers 说明):
      - 多进程各自持有一份内存 docs, 并各自把整表覆盖写同一个 pickle -> 丢更新;
      - pickle 损坏被隔离 (BM25Store.load), 而 Chroma 侧数据仍在.

    只读不修: 不一致时由调用方决定如何处理, 本函数不改任何状态.
    """
    r = get_retriever(kb_id)
    bm25_count = r.bm25_store.count()
    vector_count = r.vector_store.count()
    return {
        "kb_id": kb_id,
        "bm25_count": bm25_count,
        "vector_count": vector_count,
        "consistent": bm25_count == vector_count,
    }


def rrf_fuse(rank_lists: List[List[dict]], k: int = 60) -> List[dict]:
    """Reciprocal Rank Fusion 融合多路召回。
    公式：score(d) = sum( 1 / (k + rank_i(d)) )，k 默认 60（标准做法）。"""
    scores: dict = {}
    meta: dict = {}
    for rl in rank_lists:
        for rank, item in enumerate(rl, start=1):
            d_id = item["id"]
            scores[d_id] = scores.get(d_id, 0.0) + 1.0 / (k + rank)
            if d_id not in meta:
                meta[d_id] = item
            else:
                # 保留文本/metadata，score 取最大
                if item.get("rerank_score", 0) > meta[d_id].get("rerank_score", 0):
                    meta[d_id].update(item)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    out = []
    for d_id, s in ranked:
        item = dict(meta[d_id])
        item["rrf_score"] = s
        out.append(item)
    return out


class HybridRetriever:
    def __init__(self, kb_id: str = "default"):
        # 纵深防御: kb_id 会拼进文件路径与 collection 名, API 层已经用
        # kb_registry.validate_kb_id 拦过一次, 这里再兜一次, 保证 Agent / 评估脚本
        # 等非 HTTP 调用方也不能构造出 "../../x" 这种路径.
        from app.core.kb_registry import is_valid_kb_id

        if not is_valid_kb_id(kb_id):
            raise ValueError(
                f"非法的 kb_id: {kb_id!r} (只允许字母、数字、下划线和连字符, 长度 1-64)"
            )
        kb_id = kb_id.strip()
        self.kb_id = kb_id
        # 向量库后端按 settings.vector_store_backend 切换 (numpy 兜底 chroma 跑不起来的平台).
        backend = getattr(settings, "vector_store_backend", "chroma")
        if backend == "numpy":
            from app.rag.numpy_vector_store import NumpyVectorStore

            self.vector_store = NumpyVectorStore(collection_name=f"kb_{kb_id}")
        else:
            self.vector_store = ChromaStore(collection_name=f"kb_{kb_id}")
        # persist 目录来自 settings (默认锚定到 data/), 不写死相对路径 --
        # 相对路径会随进程 CWD 漂移, 测试隔离也方便直接覆盖该配置.
        bm25_path = Path(settings.bm25_persist_dir) / f"bm25_{kb_id}.pkl"
        self.bm25_store = BM25Store(persist_path=str(bm25_path))
        # 启动时尝试恢复 BM25
        try:
            self.bm25_store.load()
        except Exception:
            pass
        self.embedding = get_embedding_service()
        # 写并发锁: 上传 (add_documents) 与删除 (delete_documents) 都涉及
        # embedding + 向量 upsert + BM25 add/delete, 串行化避免数据竞争.
        # 读 (retrieve) 不需要这把锁 (Chroma query 自身线程安全; BM25 _ensure_index
        # 内部有 _rebuild_lock).
        self._write_lock = threading.Lock()

    def add_documents(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[dict],
        embed_texts: List[str] | None = None,
    ):
        """写入 chunk. `embed_texts` 只影响**向量表示**, `documents` 始终是存储文本.

        分开传是为了承接 HyDE 索引侧增强: 用 "chunk 原文 + 假设问题" 算 embedding
        (向量空间里更靠近问句), 但 Chroma / BM25 里存的仍是干净原文, 召回后喂给
        LLM 的上下文不会被生成内容污染. 传 None 时回退为对 documents 做 embedding.

        CPU 密集: embedding + 向量写入 + BM25 add. 同步接口.
        整个流程加锁, 避免两个上传线程同时改 BM25+Chroma.
        """
        with self._write_lock:
            # 长度错位会让向量与文档错配 (ids[i] 对应的向量其实是别人的), 而下游
            # 不会报任何错, 只表现为"检索结果莫名其妙". 在这里直接拦下.
            if embed_texts is not None and len(embed_texts) != len(documents):
                raise ValueError(
                    f"embed_texts 与 documents 长度不一致: {len(embed_texts)} vs {len(documents)}"
                )
            embeddings = self.embedding.embed(embed_texts if embed_texts is not None else documents)
            self.vector_store.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
            self.bm25_store.add(ids=ids, documents=documents, metadatas=metadatas)

    async def add_documents_async(
        self,
        ids: List[str],
        documents: List[str],
        metadatas: List[dict],
        embed_texts: List[str] | None = None,
    ) -> None:
        """async 包装: 在 async 上下文里调用, 不会阻塞事件循环.

        用 asyncio.to_thread 把 CPU 密集型 embedding/序列化
        卸到默认 threadpool, 让 FastAPI 的事件循环能继续处理其他请求.
        """
        return await asyncio.to_thread(self.add_documents, ids, documents, metadatas, embed_texts)

    def delete_documents(self, ids: List[str]) -> int:
        """按 doc_id 删除 (含其下所有 chunk). 返回实际删除的 chunk 数."""
        with self._write_lock:
            n_vec = self.vector_store.delete(ids)
            n_bm25 = self.bm25_store.delete(ids)
        return max(n_vec, n_bm25)

    def delete_by_doc_id(self, doc_id: str) -> int:
        """按 metadata.doc_id 删除整个文档的所有 chunk.

        先从 BM25 内存 docs 中收集 chunk id (O(N) 但无需重建索引),
        再同步删除 vector store 中同 doc_id 的 chunk.
        删除后 BM25 dirty=True, 下次 query 按需惰性重建.
        """
        with self._write_lock:
            ids_to_delete = [d["id"] for d in self.bm25_store.docs if d.get("metadata", {}).get("doc_id") == doc_id]
            if not ids_to_delete:
                # BM25 可能为空或尚未加载, 回退到 vector store 按 metadata 查询
                ids_to_delete = self.vector_store.get_ids_by_doc_id(doc_id)
            if ids_to_delete:
                self.vector_store.delete(ids_to_delete)
                self.bm25_store.delete(ids_to_delete)
        return len(ids_to_delete)

    def rebuild_bm25_from_vectors(self) -> int:
        """从向量库侧全量回灌 BM25 docs. 返回恢复的 chunk 数.

        场景: bm25 pkl 损坏被隔离清空, 或多 worker 覆盖写丢更新 —— 向量库是
        chunk 文本的唯一完整副本 (DB 没有 chunk 表), 回灌即恢复, 不重算
        embedding, 纯数据搬运. 入口是 documents 的 rebuild-bm25 端点 (手动
        触发; 启动自检只告警不自动修, 避免掩盖多 worker 这类持续根因).

        保真度: 向量侧 metadata 是 sanitize_metadata 后的版本 (list 拼成逗号
        字符串, None 丢弃), 回灌出的不是入库原件; 父召回依赖的 doc_id /
        chunk_index 是 str/int, 无损通过.

        必须持写锁: 回灌是整体替换 self.docs, 与并发 add_documents 竞态时
        会把锁外新写入的 chunk 直接吞掉.
        """
        with self._write_lock:
            ids, docs, metas = self.vector_store.dump_all()
            # 长度错位意味着 id 与文本可能错配, 下游不会报错只表现为"检索结果
            # 莫名其妙" (同 add_documents 拦截 embed_texts 错位的理由), 直接拦下.
            if len(docs) != len(ids) or len(metas) != len(ids):
                raise RuntimeError(
                    f"dump_all 长度错位 (kb={self.kb_id}): ids={len(ids)} "
                    f"docs={len(docs)} metas={len(metas)} —— 拒绝回灌"
                )
            self.bm25_store.docs = [
                {"id": i, "text": d, "metadata": m or {}} for i, d, m in zip(ids, docs, metas)
            ]
            self.bm25_store._dirty = True
            self.bm25_store._maybe_persist()
        return len(ids)

    def _index_by_chunk_index(self, wanted: set) -> Dict[str, Dict[int, dict]]:
        """一次遍历把 BM25 docs 按 doc_id 分组, 组内再按 chunk_index 建索引.

        只保留 wanted 里的文档. 单次 O(N) 扫描, 避免对每个父 chunk 重复全量扫描.
        BM25Store.docs 始终持有 {id, text, metadata}, 真实 / Fake 实现通用.
        """
        groups: Dict[str, Dict[int, dict]] = {doc_id: {} for doc_id in wanted}
        for d in self.bm25_store.docs:
            md = d.get("metadata", {}) or {}
            bucket = groups.get(md.get("doc_id"))
            if bucket is None:
                continue
            chunk_index = md.get("chunk_index")
            # chunk_index 由 TextChunker.split 写入 (见 chunker.py), 正常入库的数据必有.
            # 缺字段时宁可不连坐也不按位置猜: 猜错等于把不相邻的 chunk 当上下文喂给
            # LLM, 比不扩展更糟.
            if not isinstance(chunk_index, int):
                continue
            bucket[chunk_index] = {
                "id": d["id"],
                "document": d.get("text", ""),
                "metadata": md,
                "rrf_score": None,  # 占位, 后续按父 chunk 分数继承
            }
        return groups

    def expand_parent_recall(
        self,
        fused: List[dict],
        max_docs: int = 4,
        max_extra_per_doc: int = 6,
    ) -> List[dict]:
        """父文档召回 (连坐召回).

        当 top chunk 被召回时, 把它在原文中**紧邻的前后各一个 chunk** 一并带上,
        保证喂给 LLM 的上下文是连续的段落, 而不是被切碎的孤立句子.

        实现:
          - 仅对 fused 前 `max_docs` 个 doc_id 做连坐 (控成本).
          - 邻居按 metadata.chunk_index 定位, 取 `chunk_index - 1` 与 `+1`.
            父 chunk 位于文档首/尾时对应一侧不存在, 只补 1 个; 单 chunk 文档补 0 个.
          - `max_extra_per_doc` 用于进一步收紧 (0 = 放弃连坐). 取邻居的循环最多只有
            2 次迭代, 所以实际补入数不会超过 2.
          - 兄弟 chunk 继承父 chunk 的 rrf_score (× 0.99), 并**直接插在父 chunk
            之后**而非追加到末尾. 原因: 多路 query 时这些列表还会再过一次
            rrf_fuse (见 pipeline), 而 rrf_fuse 只按列表位置计分 (1/(k+rank)),
            不认存量 rrf_score —— 兄弟在末尾就会按末尾排名沉底, ×0.99 的继承分
            被位置分覆盖, 连坐形同虚设; 插在父后, 位置与分数才都说"紧随父 chunk
            之后" (reranker 降级按原顺序截断时同样受益).
        """
        if not fused:
            return fused

        # 先收集要连坐的 (doc_id, chunk_index), 再一次扫描建索引 (O(N) 而非 O(max_docs x N)).
        # rrf_fuse 已按 id 去重, (doc_id, chunk_index) 不会撞键, 建成 dict 便于
        # 边扫 fused 边查"自己是不是连坐目标".
        targets: Dict[tuple[str, int], float] = {}
        for parent in fused[:max_docs]:
            md = parent.get("metadata", {}) or {}
            doc_id = md.get("doc_id")
            chunk_index = md.get("chunk_index")
            if not doc_id or not isinstance(chunk_index, int):
                continue
            targets[(doc_id, chunk_index)] = parent.get("rrf_score", 0.0) or 0.0
        if not targets:
            return list(fused)

        index = self._index_by_chunk_index({doc_id for doc_id, _ in targets})
        # seen 预置全部已召回 id. 额度计的是"实际补进来的新 chunk 数": 已召回的
        # 邻居跳过且不占额度 -- 先截断再过滤会让额度被已召回的 chunk 白白吃掉.
        seen: set = {c["id"] for c in fused}
        expanded: List[dict] = []
        for c in fused:
            expanded.append(c)
            md = c.get("metadata", {}) or {}
            doc_id = md.get("doc_id")
            chunk_index = md.get("chunk_index")
            if not doc_id or not isinstance(chunk_index, int):
                continue
            parent_score = targets.get((doc_id, chunk_index))
            if parent_score is None:
                continue
            bucket = index.get(doc_id, {})
            added = 0
            # 只取紧邻的前后各一个, 前 -> 后 (前一个缺失时自然落到后一个).
            for neighbor_index in (chunk_index - 1, chunk_index + 1):
                if added >= max_extra_per_doc:
                    break
                sibling = bucket.get(neighbor_index)
                if sibling is None or sibling["id"] in seen:
                    continue
                sibling["rrf_score"] = parent_score * 0.99  # 紧随父 chunk 之后
                expanded.append(sibling)  # 插在父 chunk 之后, 理由见 docstring
                seen.add(sibling["id"])
                added += 1
        return expanded

    def retrieve(self, query: str, top_k: int | None = None) -> List[dict]:
        # CPU 密集: embedding + 两次召回 + RRF. 同步接口.
        # 读不需要锁; BM25 内部 _ensure_index 有 _rebuild_lock.
        top_k = top_k or settings.top_k_final
        qv = self.embedding.embed_query(query)
        vec_hits = self.vector_store.query(qv, top_k=settings.top_k_vector)
        bm25_hits = self.bm25_store.query(query, top_k=settings.top_k_bm25)
        fused = rrf_fuse([vec_hits, bm25_hits], k=60)
        # 候选池: 先取 top_k 个做 RRF 融合结果.
        fused_top = fused[: top_k * 2]
        # 父文档召回 (连坐): 把同文档兄弟 chunk 补回, 提升上下文完整度.
        if getattr(settings, "parent_recall_enabled", False):
            fused_top = self.expand_parent_recall(
                fused_top, max_docs=4, max_extra_per_doc=settings.parent_recall_extra
            )
        return fused_top

    async def retrieve_async(self, query: str, top_k: int | None = None) -> List[dict]:
        """async 包装: 在 async 上下文里调用, 不会阻塞事件循环."""
        return await asyncio.to_thread(self.retrieve, query, top_k)
