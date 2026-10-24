"""P2-13 测试: 文档删除接口.

覆盖:
1. 上传文档后 DELETE /documents/{doc_id} 返回 200 并删除所有 chunk.
2. 删除后 BM25/vector count 均为 0.
3. 删除不存在的 doc_id 返回 404.
4. 其他用户无法删除别人 kb 的文档 (403).
5. BM25 删除后 dirty=True, 下次 query 能惰性重建 (不影响其他文档检索).
"""
from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(monkeypatch, tmp_path):
    """构造一个 patch 了嵌入/检索的 TestClient."""
    from app.agents import llm_router as lr_mod
    from app.core import kb_registry as kb_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    import os
    from tests._fakes import (
        FakeBM25Store,
        FakeEmbeddingService,
        FakeLLM,
        FakeReranker,
        FakeVectorStore,
    )

    monkeypatch.setenv("APP_ENV", "dev")
    monkeypatch.setenv("JWT_SECRET", "a-very-long-secret-for-test-only-32chars")
    from app.core.config import get_settings
    get_settings.cache_clear()

    # Windows 没有 /tmp, 统一用 pytest 的 tmp_path
    monkeypatch.chdir(tmp_path)
    os.environ["PILKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()
    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rk_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rt_mod, "reset_retriever_cache", lambda: None)
    monkeypatch.setattr(lr_mod, "get_llm", lambda: FakeLLM())

    from app.api.auth import reset_users_for_tests
    from app.main import app
    reset_users_for_tests()
    kb_mod.reset_for_tests()
    return TestClient(app)


def _register(c: TestClient, username: str, password: str = "password123") -> str:
    r = c.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _upload(c: TestClient, token: str, kb_id: str, text: str, filename: str = "doc.txt"):
    r = c.post(
        f"/api/v1/documents/upload?kb_id={kb_id}",
        headers={"Authorization": f"Bearer {token}"},
        files={"file": (filename, io.BytesIO(text.encode()), "text/plain")},
    )
    assert r.status_code == 200, r.text
    return r.json()["doc_id"]


def test_delete_document_removes_chunks(client):
    """上传后再删除, chunk 数归零."""
    token = _register(client, "alice")
    doc_id = _upload(client, token, "kb_alice", "这是第一篇文档。包含一些关键词。")

    r = client.delete(
        f"/api/v1/documents/{doc_id}?kb_id=kb_alice",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "deleted"
    assert data["doc_id"] == doc_id
    assert data["chunks_deleted"] > 0

    r = client.get(
        "/api/v1/documents/list?kb_id=kb_alice",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["bm25_count"] == 0
    assert r.json()["vector_count"] == 0


def test_delete_nonexistent_document_returns_404(client):
    """删除不存在的 doc_id 返回 404."""
    token = _register(client, "bob")
    r = client.delete(
        "/api/v1/documents/doc_notexist?kb_id=kb_bob",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 404


def test_delete_document_forbidden_for_other_user(client):
    """其他用户不能删除别人 kb 的文档."""
    alice_token = _register(client, "alice2")
    doc_id = _upload(client, alice_token, "kb_private", "alice 的私有文档内容。")

    bob_token = _register(client, "bob2")
    r = client.delete(
        f"/api/v1/documents/{doc_id}?kb_id=kb_private",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert r.status_code == 403, r.text


def test_list_documents_aggregates_chunks_by_doc(client):
    """list 的 documents 按 doc_id 聚合: 多 chunk 的文档只出现一条, 带 chunk 数."""
    token = _register(client, "alice")
    doc_a = _upload(client, token, "kb_alice_list", "苹果 香蕉 橙子 汽车 火车 飞机 " * 40)
    doc_b = _upload(client, token, "kb_alice_list", "短文档。", filename="b.txt")

    r = client.get(
        "/api/v1/documents/list?kb_id=kb_alice_list",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    docs = {d["doc_id"]: d for d in data["documents"]}
    assert set(docs) == {doc_a, doc_b}
    assert docs[doc_a]["filename"] == "doc.txt"
    assert docs[doc_b]["filename"] == "b.txt"
    # 聚合出的 chunks 之和必须等于向量库条数, 否则列表会漏掉/重复
    assert sum(d["chunks"] for d in data["documents"]) == data["vector_count"]
    assert data["documents"] == sorted(data["documents"], key=lambda d: (d["filename"], d["doc_id"]))
    assert data["can_write"] is True


def test_delete_removes_document_from_list(client):
    """端到端: 上传 -> 列表可见 -> 删除 -> 列表消失."""
    token = _register(client, "alice")
    doc_id = _upload(client, token, "kb_alice_del", "要删掉的文档内容。")

    def _list():
        r = client.get(
            "/api/v1/documents/list?kb_id=kb_alice_del",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert r.status_code == 200, r.text
        return r.json()

    assert [d["doc_id"] for d in _list()["documents"]] == [doc_id]

    r = client.delete(
        f"/api/v1/documents/{doc_id}?kb_id=kb_alice_del",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text

    after = _list()
    assert after["documents"] == []
    assert after["vector_count"] == 0
    assert after["bm25_count"] == 0


def test_list_can_write_false_for_public_kb(client):
    """公共 kb 只读: can_write=false, 前端据此不显示删除入口."""
    token = _register(client, "alice")

    r = client.get(
        "/api/v1/documents/list?kb_id=default",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["can_write"] is False


def test_delete_on_public_kb_is_rejected(client):
    """公共 kb 的删除请求必须被写校验拦下 (403), 不能只靠前端隐藏按钮."""
    token = _register(client, "alice")

    r = client.delete(
        "/api/v1/documents/doc_whatever?kb_id=default",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 403, r.text


def test_delete_document_does_not_affect_other_docs(client):
    """删除文档 A 后文档 B 仍可被检索, 且不会召回已删除的 A."""
    token = _register(client, "carol")
    doc_a = _upload(client, token, "kb_shared", "苹果 香蕉 橙子")
    doc_b = _upload(client, token, "kb_shared", "汽车 火车 飞机")

    # 删除 doc_a
    r = client.delete(
        f"/api/v1/documents/{doc_a}?kb_id=kb_shared",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200

    # 查询 "飞机" 仍应有结果, 且结果里只有 doc_b
    from app.rag.retriever import get_retriever
    retriever = get_retriever("kb_shared")
    hits = retriever.retrieve("飞机")
    assert any(h["metadata"].get("doc_id") == doc_b for h in hits)
    assert not any(h["metadata"].get("doc_id") == doc_a for h in hits)
