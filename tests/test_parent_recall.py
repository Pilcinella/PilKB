"""测试: 父文档召回 (连坐召回) ParentDocRecall.

行为约定: 命中某个 chunk 时, 只连带召回它在原文中**紧邻的前后各一个** chunk
(按 metadata.chunk_index 定位), 而不是该文档的前 N 个 chunk.
"""
from __future__ import annotations

from app.rag.retriever import HybridRetriever


def _make_retriever(docs: dict[str, int]) -> HybridRetriever:
    """docs: {doc_id: chunk 数}. 按 chunk_index 升序灌入 bm25_store.docs,
    metadata 字段与 TextChunker.split 的产出一致 (见 chunker.py)."""
    r = HybridRetriever(kb_id="test_parent")
    r.bm25_store.docs = [
        {
            "id": f"{doc_id}_c{i}",
            "text": f"{doc_id} 第 {i} 段",
            "metadata": {"doc_id": doc_id, "chunk_id": f"{doc_id}_c{i}", "chunk_index": i},
        }
        for doc_id, n in docs.items()
        for i in range(n)
    ]
    return r


def _hit(retriever: HybridRetriever, ids: list[str], score: float = 0.9) -> list[dict]:
    """挑出指定 id 当作"已召回结果", 字段与 retrieve() 的输出保持一致."""
    by_id = {d["id"]: d for d in retriever.bm25_store.docs}
    return [
        {
            "id": i,
            "document": by_id[i]["text"],
            "metadata": by_id[i]["metadata"],
            "rrf_score": score,
        }
        for i in ids
    ]


# ----- 长文档: 取邻近, 不取文档开头 -----
def test_parent_recall_long_doc_takes_neighbors_not_opening():
    """10 个 chunk 的文档命中中段 c5: 只补 c4 与 c6.

    改动前的行为是"从 c0 起按入库顺序补满 max_extra_per_doc 个", 于是命中 c5
    时会补回 c0..c4/c6 —— 文档开头, 与父 chunk 所在段落并不相邻.
    """
    r = _make_retriever({"docA": 10})
    expanded = r.expand_parent_recall(_hit(r, ["docA_c5"]), max_docs=4, max_extra_per_doc=6)
    ids = [c["id"] for c in expanded]

    assert ids == ["docA_c5", "docA_c4", "docA_c6"]
    # 改动前的表现: 文档开头被当作"兄弟"补进来
    assert "docA_c0" not in ids
    assert "docA_c1" not in ids


def test_parent_recall_does_not_mix_documents():
    r = _make_retriever({"docA": 10, "docB": 10})
    ids = [c["id"] for c in r.expand_parent_recall(_hit(r, ["docA_c5"]))]
    assert ids == ["docA_c5", "docA_c4", "docA_c6"]
    assert all(i.startswith("docA_") for i in ids)


# ----- 边界: 缺前 / 缺后 / 前后都缺 -----
def test_parent_recall_first_chunk_only_has_next():
    r = _make_retriever({"docA": 10})
    ids = [c["id"] for c in r.expand_parent_recall(_hit(r, ["docA_c0"]))]
    assert ids == ["docA_c0", "docA_c1"]  # c-1 不存在 -> 只补 1 个


def test_parent_recall_last_chunk_only_has_prev():
    r = _make_retriever({"docA": 10})
    ids = [c["id"] for c in r.expand_parent_recall(_hit(r, ["docA_c9"]))]
    assert ids == ["docA_c9", "docA_c8"]  # c10 不存在 -> 只补 1 个


def test_parent_recall_single_chunk_doc_adds_nothing():
    """单 chunk 文档前后都没有邻居 -> 补 0 个."""
    r = _make_retriever({"docB": 1})
    fused = _hit(r, ["docB_c0"])
    expanded = r.expand_parent_recall(fused)
    assert [c["id"] for c in expanded] == ["docB_c0"]
    assert expanded == fused


def test_parent_recall_skips_recalled_neighbor_without_using_budget():
    """相邻的两个 chunk 都在召回结果里时, 应补它们**外侧**的邻居.

    已召回的邻居跳过且不占额度 -- 若先截断再过滤, 额度会被 c5/c6 自己吃掉,
    结果是补不进 c4 与 c7.
    """
    r = _make_retriever({"docA": 10})
    fused = _hit(r, ["docA_c5", "docA_c6"])
    ids = [c["id"] for c in r.expand_parent_recall(fused)]
    assert ids == ["docA_c5", "docA_c6", "docA_c4", "docA_c7"]


# ----- 放弃召回 -----
def test_parent_recall_extra_limit_zero_disables_expansion():
    """parent_recall_extra=0 -> 完全放弃连坐, 结果与 fused 一致."""
    r = _make_retriever({"docA": 10})
    fused = _hit(r, ["docA_c5"])
    assert r.expand_parent_recall(fused, max_extra_per_doc=0) == fused


def test_parent_recall_extra_limit_one_takes_prev_only():
    """额度收紧到 1: 只补前一个; 前一个不存在才轮到后一个."""
    r = _make_retriever({"docA": 10})
    assert [c["id"] for c in r.expand_parent_recall(_hit(r, ["docA_c5"]), max_extra_per_doc=1)] == [
        "docA_c5",
        "docA_c4",
    ]
    assert [c["id"] for c in r.expand_parent_recall(_hit(r, ["docA_c0"]), max_extra_per_doc=1)] == [
        "docA_c0",
        "docA_c1",
    ]


def test_parent_recall_empty_fused():
    r = _make_retriever({"docA": 3})
    assert r.expand_parent_recall([]) == []


# ----- 缺字段 / 分数继承 -----
def test_parent_recall_ignores_chunks_without_chunk_index():
    """metadata 缺 chunk_index 时不做连坐: 宁可少补, 也不猜一个不相邻的邻居."""
    r = _make_retriever({"docA": 3})
    fused = [{"id": "docA_c1", "document": "x", "metadata": {"doc_id": "docA"}, "rrf_score": 0.9}]
    assert r.expand_parent_recall(fused) == fused


def test_parent_recall_inherits_parent_score():
    r = _make_retriever({"docA": 10})
    expanded = r.expand_parent_recall(_hit(r, ["docA_c5"]), max_docs=4, max_extra_per_doc=6)
    sib = next(c for c in expanded if c["id"] == "docA_c6")
    assert abs(sib["rrf_score"] - 0.9 * 0.99) < 1e-9
    assert expanded[0]["id"] == "docA_c5"  # 父 chunk 仍排第一
