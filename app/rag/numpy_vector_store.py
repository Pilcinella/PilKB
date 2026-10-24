"""Pure-Python vector store (numpy + cosine). 用来替代 chromadb.

为什么存在:
- chromadb 0.4~1.5 在 Windows + Anaconda Python 3.12 上 PersistentClient / upsert
  会直接 ACCESS_VIOLATION (-1073741819), 跨多版本无法绕.
- 这个 store 接口与 ChromaStore 完全一致 (add / query / delete / count / reset /
  get_ids_by_doc_id), 只是用 numpy 做余弦相似度, 持久化用 pickle.
- 适合中小规模 (< 几十万条). 生产想换回 chroma 只需改 VECTOR_STORE_BACKEND.

性能提示:
- query 是 O(N*d) 的全量扫, 单 kb 几十万以下足够.
- 不做 ANN 近似检索, 因为 chroma 在 win 跑不起来; 真要做大规模再换 faiss /
  hnswlib (那俩也有 C 扩展但生态更稳).
"""
from __future__ import annotations

import pickle
import threading
from pathlib import Path
from typing import List, Optional

from loguru import logger


def _l2_normalize(x):
    """行向量 L2 归一化 (in-place)."""
    import numpy as np
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


class NumpyVectorStore:
    """和 ChromaStore 同接口的纯 numpy 实现, 默认 cosine 距离."""

    def __init__(self, persist_dir: str | None = None, collection_name: str = "default"):
        from app.core.config import settings

        self.persist_dir = persist_dir or settings.chroma_persist_dir
        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self.collection_name = collection_name
        # collection_name 可能包含非法文件名字符 (如 ':'), 用 collection_name.replace 一下
        self._path = Path(self.persist_dir) / f"numpy_{collection_name.replace('/', '_')}.pkl"
        self._lock = threading.Lock()
        # 数据
        self._ids: List[str] = []
        self._docs: List[str] = []
        self._metas: List[dict] = []
        self._vecs = None  # ndarray, shape (N, D), 全部归一化
        self._load()

    # ---------- persistence ----------

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            with self._path.open("rb") as f:
                blob = pickle.load(f)
            self._ids = blob.get("ids", [])
            self._docs = blob.get("docs", [])
            self._metas = blob.get("metas", [])
            vecs = blob.get("vecs")
            if vecs is not None and len(self._ids) > 0:
                self._vecs = vecs
            logger.info(
                f"NumpyVectorStore[{self.collection_name}] loaded {len(self._ids)} vectors from {self._path}"
            )
        except Exception as e:  # noqa: BLE001
            # 反序列化失败最常见的原因是 numpy / 类定义版本漂移 (典型报错:
            # No module named 'numpy._core.numeric'). 此时若只是置空, 下一次
            # _save() 就会用空数据覆盖原文件 -> 整个 kb 的向量永久丢失.
            # 所以先把损坏文件旁路备份, 再以空库启动, 数据仍可事后人工恢复.
            logger.error(
                f"NumpyVectorStore[{self.collection_name}] load failed, starting empty: {e}"
            )
            self._ids, self._docs, self._metas, self._vecs = [], [], [], None
            self._quarantine_corrupt_file()

    def _quarantine_corrupt_file(self) -> None:
        """把无法解析的持久化文件改名旁路, 避免被空数据覆盖 (数据可事后恢复)."""
        try:
            backup = self._path.with_name(self._path.name + ".corrupt")
            self._path.replace(backup)
            logger.error(
                f"NumpyVectorStore[{self.collection_name}] 已隔离损坏文件 -> {backup}"
            )
        except Exception as e:  # noqa: BLE001
            logger.warning(f"隔离损坏文件失败 (原文件仍保留在 {self._path}): {e}")

    def _save(self) -> None:
        import numpy as np
        blob = {
            "ids": self._ids,
            "docs": self._docs,
            "metas": self._metas,
            "vecs": self._vecs if self._vecs is not None else np.empty((0, 0), dtype=np.float32),
        }
        tmp = self._path.with_suffix(self._path.suffix + ".tmp")
        with tmp.open("wb") as f:
            pickle.dump(blob, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(self._path)

    # ---------- API (mirror ChromaStore) ----------

    def add(self, ids: List[str], documents: List[str], embeddings: List[List[float]],
            metadatas: Optional[List[dict]] = None) -> None:
        from app.rag.vector_store import sanitize_metadata

        import numpy as np
        safe_metas = [sanitize_metadata(m) for m in (metadatas or [])]
        if not ids:
            return
        new_vecs = np.asarray(embeddings, dtype=np.float32)
        new_vecs = _l2_normalize(new_vecs)  # cosine -> 内积

        with self._lock:
            # 先划分"更新"与"新增", 再一次性重建 ndarray.
            #
            # 旧实现在同一个 if 里先把 self._vecs 置 None 再 vstack([self._vecs,
            # new_vecs]), 此时 self._vecs 已经是 None, vstack 分支根本走不到, 最终
            # self._vecs 只剩本次 new_vecs. 结果是 self._ids 累积了多次上传的条目、
            # self._vecs 却只有最后一批 -> id 与向量整体错位, 检索返回错误文档,
            # 且在 delete/query 时越界抛 IndexError.
            #
            # 这里用 index 映射替代 O(N) 的 list.index / in 判断, 批量入库从
            # O(N^2) 降到 O(N).
            id_to_pos = {_id: i for i, _id in enumerate(self._ids)}
            updates: list[tuple[int, int]] = []  # (existing_row, local_i)
            appends: list[int] = []              # local_i

            for i, _id in enumerate(ids):
                pos = id_to_pos.get(_id)
                if pos is None:
                    appends.append(i)
                    id_to_pos[_id] = len(self._ids) + len(appends) - 1
                else:
                    updates.append((pos, i))

            # 1) 追加新条目 (文本/元数据)
            for i in appends:
                self._ids.append(ids[i])
                self._docs.append(documents[i] if i < len(documents) else "")
                self._metas.append(safe_metas[i] if i < len(safe_metas) else {})

            # 2) 重建向量矩阵: 先垂直拼接新增行, 再原地覆盖已存在的行.
            #    始终保持 len(self._vecs) == len(self._ids).
            if self._vecs is None or len(self._vecs) == 0:
                self._vecs = (
                    np.ascontiguousarray(new_vecs[appends])
                    if appends
                    else np.empty((0, new_vecs.shape[1]), dtype=np.float32)
                )
            elif appends:
                self._vecs = np.vstack([self._vecs, new_vecs[appends]])
            for row, i in updates:
                self._vecs[row] = new_vecs[i]

            # 3) 覆盖已存在条目的文本/元数据
            for row, i in updates:
                self._docs[row] = documents[i] if i < len(documents) else ""
                self._metas[row] = safe_metas[i] if i < len(safe_metas) else {}

            # 不变量校验: 一旦对不上说明又引入了错位, 直接暴露而不是静默返回错答案.
            if len(self._vecs) != len(self._ids):
                raise RuntimeError(
                    f"NumpyVectorStore[{self.collection_name}] 向量/ID 数量不一致: "
                    f"{len(self._vecs)} vectors vs {len(self._ids)} ids"
                )

            self._save()
        logger.info(f"NumpyVectorStore[{self.collection_name}] upsert {len(ids)} (total {len(self._ids)})")

    def query(self, query_embedding: List[float], top_k: int = 20,
              where: Optional[dict] = None):
        import numpy as np

        # 锁内只做快照拷贝, 计算全部移到锁外: query 的矩阵乘 + 元数据过滤是
        # O(N*d) 热路径, 全程持锁会把 pipeline 多 query 的并发检索
        # (asyncio.gather) 退化成串行, 还会和 add() 的写盘互相阻塞.
        # 快照一致性: delete / add-追加都用新对象整体替换 _vecs / _ids
        # (vstack / 新列表), 快照引用安全; add 的"更新已存在 id"路径会原地写
        # 旧矩阵的行, 但行号 <-> id 的对应不变, 并发读到该行的新旧向量都是
        # 同一 chunk 的合法版本.
        with self._lock:
            if not self._ids or self._vecs is None or len(self._vecs) == 0:
                return []
            ids = self._ids[:]
            docs = self._docs[:]
            metas = self._metas[:]
            vecs = self._vecs
        q = np.asarray(query_embedding, dtype=np.float32)
        q = q / (np.linalg.norm(q) or 1.0)
        sims = vecs @ q  # 余弦相似度 (都已归一化)
        # where 过滤 (cosine 之外的 metadata 标量等值过滤, 与 chroma 行为近似)
        if where:
            # chromadb 的 where 是 AND, 支持 {"k": "v"}
            keep = np.ones(len(ids), dtype=bool)
            for k, v in where.items():
                keep &= np.array([m.get(k) == v for m in metas], dtype=bool)
            sims = np.where(keep, sims, -np.inf)
        k = min(top_k, len(sims))
        # top-k by similarity desc
        idx = np.argpartition(-sims, k - 1)[:k] if k < len(sims) else np.arange(len(sims))
        idx = idx[np.argsort(-sims[idx])]
        out = []
        for i in idx:
            sim = float(sims[i])
            if sim == -np.inf:
                continue
            out.append({
                "id": ids[i],
                "document": docs[i],
                "metadata": metas[i],
                # chroma 用 distance (越小越近), 我们用相似度 -> 距离 = 1 - sim
                "distance": 1.0 - sim,
            })
        return out

    def delete(self, ids: List[str]) -> int:
        import numpy as np
        with self._lock:
            if not ids:
                return 0
            before = len(self._ids)
            # set 必须在循环外构造: 写在条件里会每轮重建一个 set, 复杂度从
            # O(N + M) 退化成 O(N x M) (10 万条删 1 千条 = 1 亿次哈希).
            ids_set = set(ids)
            keep_idx = [i for i, _id in enumerate(self._ids) if _id not in ids_set]
            removed = before - len(keep_idx)
            if removed == 0:
                return 0
            self._ids = [self._ids[i] for i in keep_idx]
            self._docs = [self._docs[i] for i in keep_idx]
            self._metas = [self._metas[i] for i in keep_idx]
            self._vecs = self._vecs[keep_idx] if self._vecs is not None and len(self._vecs) else None
            self._save()
        return removed

    def get_ids_by_doc_id(self, doc_id: str) -> List[str]:
        with self._lock:
            return [self._ids[i] for i, m in enumerate(self._metas) if m.get("doc_id") == doc_id]

    def dump_all(self) -> tuple[list[str], list[str], list[dict]]:
        """全量快照导出 (ids/docs/metas), 供 BM25 从向量库侧回灌重建.

        锁内只做 list 切片快照 (与 query 同一模式); metadata dict 共享引用 ——
        BM25 侧只读不写, 和 add() 直接持有调用方 dict 的既有约定一致.
        """
        with self._lock:
            return self._ids[:], self._docs[:], self._metas[:]

    def list_docs(self) -> list[dict]:
        """列出 kb 内的文档 (按 doc_id 聚合), 供前端展示与删除入口.

        metadata 全在内存, 锁内拷一份 metas 后到锁外聚合 (与 query 同一模式:
        聚合是 O(N) 纯计算, 不该占着写锁).
        """
        from app.rag.vector_store import aggregate_documents

        with self._lock:
            metas = self._metas[:]
        return aggregate_documents(metas)

    def count(self) -> int:
        with self._lock:
            return len(self._ids)

    def reset(self) -> None:
        with self._lock:
            self._ids, self._docs, self._metas = [], [], []
            self._vecs = None
            try:
                self._path.unlink(missing_ok=True)
            except Exception:
                pass