"""Chroma vector store wrapper."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import List, Optional

from loguru import logger

from app.core.config import settings


# Chroma 只接受 str/int/float/bool 作为 metadata value, 传 list/dict/None 会直接抛
# ValueError 并让整批 upsert 失败. 上游 (MetadataEnhancer 的 keywords / HyDE 字段)
# 天然会产出 list, 所以在写入层做一次统一归一化, 避免真实上传时崩在最后一步.
_ALLOWED_META_TYPES = (str, int, float, bool)


# 进程级 Chroma 引导锁. 必须是**模块级**而不是实例级:
# 所有 kb 共用同一个 persist 目录 (settings.chroma_persist_dir), 而 chromadb 的
# 首次引导 (建 sqlite 文件 / 跑 schema 迁移 / 建 default_tenant / 建 collection)
# 在并发下不安全. 实测同一进程内 4 线程同时首次触达:
#   - 同一 store:      60 次调用 26 次抛异常 (43%)
#   - 不同 store/同目录: 48 次调用 12 次抛异常 (25%)
# 抛的是 ValueError("Could not connect to tenant default_tenant") 和
# sqlite3.OperationalError("duplicate column name: dimension") —— 后者说明多个线程
# 在同一个 sqlite 上并行跑迁移. 实例级锁挡不住跨 kb 的竞争 (各自一把锁),
# 所以这里用一把全局锁把引导串行化.
_CHROMA_INIT_LOCK = threading.Lock()


def aggregate_documents(metadatas) -> list[dict]:
    """把 chunk 级 metadata 聚合成文档级列表.

    两个向量库实现 (ChromaStore / NumpyVectorStore) 共用, 避免"哪些字段算一份
    文档、怎么排序"这类规则在两个实现里各写一遍而漂移。

    输出每条: {doc_id, filename, user, chunks}.
    - chunks 是该 doc_id 下的 chunk 数 (删除时前端要据此提示影响范围);
    - filename / user 取首个非空值: 同一文档的 chunk 由同一次上传写入, 这几个
      字段必然一致, 无需逐条比对;
    - 缺 doc_id 的条目跳过 (正常入库必有, 见 chunker; 缺失说明数据异常,
      列出来也无法删除, 不如不显示);
    - 排序 (filename, doc_id) 稳定: 删除后列表不会跳动.
    """
    agg: dict[str, dict] = {}
    for m in metadatas:
        if not m:
            continue
        doc_id = m.get("doc_id")
        if not doc_id:
            continue
        item = agg.get(doc_id)
        if item is None:
            agg[doc_id] = {
                "doc_id": doc_id,
                "filename": m.get("filename") or "",
                "user": m.get("user") or "",
                "chunks": 1,
            }
        else:
            item["chunks"] += 1
    return sorted(agg.values(), key=lambda d: (d["filename"], d["doc_id"]))


def sanitize_metadata(meta: Optional[dict]) -> dict:
    """把任意 metadata 归一化成 Chroma 可接受的标量字典。

    - list/tuple/set -> 逗号拼接的字符串 (保留可读性, 便于 where 做 LIKE 式过滤)
    - dict/其它对象  -> str()
    - None           -> 丢弃 (Chroma 不接受 None)
    - 顺序说明: bool 是 int 的子类, 因此会被前面的 int 分支先匹配; 二者 Chroma
      都支持, 这里的排列顺序其实无关紧要 (按 str/int/float/bool 仅为可读性).
    """
    if not meta:
        return {}
    out: dict = {}
    for k, v in meta.items():
        if v is None:
            continue
        key = str(k)
        if isinstance(v, _ALLOWED_META_TYPES):
            out[key] = v
        elif isinstance(v, (list, tuple, set)):
            out[key] = ", ".join(str(x) for x in v if x is not None)
        else:
            out[key] = str(v)
    return out


class ChromaStore:
    def __init__(self, persist_dir: str | None = None, collection_name: str = "default"):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        self._client = None
        self._col = None
        # 多线程并发请求会复用同一实例, 给写操作加锁避免并发 upsert/delete 冲突.
        # (Chroma 0.4+ 单进程单 collection 写并发不安全, 官方建议锁.)
        self._write_lock = threading.Lock()

    def _ensure(self):
        """返回 collection 句柄, 首次调用时完成 chromadb 引导.

        守卫看的是 `_col` 而不是 `_client`: reset() 只把 `_col` 置 None (client 仍
        可用, 复用它重建 collection 即可), 若按 `_client` 判断就会跳过重建并返回
        None, 让 store 在 reset 之后永久失效. 两个字段的职责因此是:
          `_client` —— 只在从未连过时为 None, 建一次就复用
          `_col`    —— 真正的"是否已就绪"标志, reset() 会让它失效
        """
        # 热路径: 已就绪则无锁返回, 锁只在冷启动/重置后被触碰,
        # 所以不会影响并发读 (retrieve 的多 query 并发依赖这一点).
        if self._col is not None:
            return self._col
        with _CHROMA_INIT_LOCK:
            # 双检: 等锁期间别的线程 (或别的 kb 的 store) 可能已经完成引导.
            if self._col is not None:
                return self._col
            import chromadb
            from chromadb.config import Settings as Cfg
            if self._client is None:
                self._client = chromadb.PersistentClient(
                    path=self.persist_dir, settings=Cfg(anonymized_telemetry=False)
                )
            self._col = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return self._col

    def add(self, ids: List[str], documents: List[str], embeddings: List[List[float]], metadatas: List[dict]):
        col = self._ensure()
        safe_metas = [sanitize_metadata(m) for m in (metadatas or [])]
        with self._write_lock:
            col.upsert(ids=ids, documents=documents, embeddings=embeddings, metadatas=safe_metas)
        logger.info(f"Chroma upsert: {len(ids)} vectors into {self.collection_name}")

    def query(self, query_embedding: List[float], top_k: int = 20, where: Optional[dict] = None):
        col = self._ensure()
        kwargs = {"query_embeddings": [query_embedding], "n_results": top_k}
        if where:
            kwargs["where"] = where
        res = col.query(**kwargs)
        # 返回统一结构
        out = []
        if not res or not res.get("ids"):
            return out
        for i, _id in enumerate(res["ids"][0]):
            out.append({
                "id": _id,
                "document": res["documents"][0][i] if res.get("documents") else "",
                "metadata": res["metadatas"][0][i] if res.get("metadatas") else {},
                "distance": res["distances"][0][i] if res.get("distances") else 0.0,
            })
        return out

    def delete(self, ids: List[str]) -> int:
        """按 id 删除. 返回 0/实际删除数 (chroma 没有返回值, 用 count 估算)."""
        if not ids:
            return 0
        col = self._ensure()
        with self._write_lock:
            before = col.count()
            col.delete(ids=ids)
            after = col.count()
        return max(0, before - after)

    def get_ids_by_doc_id(self, doc_id: str) -> List[str]:
        """按 metadata.doc_id 查询并返回所有 chunk id."""
        col = self._ensure()
        try:
            res = col.get(where={"doc_id": doc_id}, include=[])
        except Exception:
            return []
        return res.get("ids", []) or []

    def dump_all(self) -> tuple[list[str], list[str], list[dict]]:
        """全量导出 (id/document/metadata), 供 BM25 从向量库侧回灌重建.

        翻页拉取: 不带 limit 的 get() 会把整个 collection 一次性物化, 大 kb
        时内存不可控; batch 尺寸取 chroma 默认 max batch 的量级即可.
        """
        col = self._ensure()
        ids: List[str] = []
        docs: List[str] = []
        metas: List[dict] = []
        limit = 5000
        offset = 0
        while True:
            res = col.get(include=["documents", "metadatas"], limit=limit, offset=offset)
            batch = res.get("ids") or []
            if not batch:
                break
            ids.extend(batch)
            docs.extend(res.get("documents") or [])
            metas.extend(res.get("metadatas") or [])
            if len(batch) < limit:
                break
            offset += limit
        return ids, docs, metas

    def list_docs(self) -> list[dict]:
        """列出 kb 内的文档 (按 doc_id 聚合), 供前端展示与删除入口.

        只取 metadatas 不取 documents: 文档列表只需要文件名/归属/条数, 拉全文
        是纯浪费 (大 kb 下文本占的字节远超 metadata). 同样翻页, 理由见 dump_all.
        """
        col = self._ensure()
        metas: List[dict] = []
        limit = 5000
        offset = 0
        while True:
            res = col.get(include=["metadatas"], limit=limit, offset=offset)
            batch = res.get("ids") or []
            if not batch:
                break
            metas.extend(res.get("metadatas") or [])
            if len(batch) < limit:
                break
            offset += limit
        return aggregate_documents(metas)

    def count(self) -> int:
        col = self._ensure()
        return col.count()

    def reset(self):
        """删除整个 collection, 让 store 回到"未就绪"状态.

        只置 `_col = None` (不动 `_client`): 下次 _ensure() 会复用同一个 client
        重建 collection —— 这正是 _ensure() 按 `_col` 判断的原因, 见其 docstring.
        """
        self._ensure()
        with self._write_lock:
            try:
                self._client.delete_collection(self.collection_name)
            except Exception:
                pass
        self._col = None
