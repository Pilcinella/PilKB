"""测试: BM25 从向量库侧回灌重建 (HybridRetriever.rebuild_bm25_from_vectors).

场景: bm25 pkl 损坏被隔离后 BM25 空库启动, 向量库仍持有全量 chunk ——
应当能从向量库把 BM25 完整恢复 (回灌 -> 落盘 -> 重启重载 -> 检索恢复),
而不是只能重新上传文档.

与 test_store_consistency 的分工: 那边用 fake 守自检的判据, 这里用**真实**
BM25Store + NumpyVectorStore (tmp_path 持久化), 只 fake embedding ——
回灌/落盘/重载这条真实 IO 链路正是被测对象. API 层的契约另用 fakes 覆盖.
"""
from __future__ import annotations

import io
import os

import pytest
from fastapi.testclient import TestClient

from tests._fakes import FakeEmbeddingService


@pytest.fixture()
def real_stores_in_tmp(monkeypatch, tmp_path):
    """真实 BM25Store + NumpyVectorStore 落到 tmp_path, 只 fake embedding."""
    from app.core.config import settings
    from app.rag import embedding as emb_mod
    from app.rag import retriever as rt_mod

    monkeypatch.setattr(settings, "bm25_persist_dir", str(tmp_path))
    monkeypatch.setattr(settings, "chroma_persist_dir", str(tmp_path))
    monkeypatch.setattr(settings, "vector_store_backend", "numpy")
    os.environ["PILKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    rt_mod.reset_retriever_cache()
    yield
    rt_mod.reset_retriever_cache()
    emb_mod.get_embedding_service.cache_clear()
    os.environ.pop("PILKB_FAKE_EMBEDDING", None)


def _seed(kb_id: str, n: int = 3):
    """灌 n 个 chunk, 两路都写 (与真实上传链路一致)."""
    from app.rag.retriever import get_retriever

    r = get_retriever(kb_id)
    r.add_documents(
        ids=[f"d{i}_c0" for i in range(n)],
        documents=[f"文档 {i} 的正文内容 alpha{i}" for i in range(n)],
        metadatas=[{"doc_id": f"d{i}", "chunk_index": 0} for i in range(n)],
    )
    return r


def test_rebuild_restores_after_corrupt_quarantine(real_stores_in_tmp, tmp_path):
    """完整恢复链路: 损坏隔离 -> 空库启动 -> 回灌 -> 一致 -> 落盘 -> 重载可检索."""
    from app.rag.retriever import (
        check_store_consistency,
        get_retriever,
        reset_retriever_cache,
    )

    _seed("kb_rebuild", 3)

    # 损坏 + "重启": pkl 写垃圾, 重建 retriever 实例 (load -> 隔离 -> 空库)
    pkl = tmp_path / "bm25_kb_rebuild.pkl"
    pkl.write_bytes(b"\x80\x04corrupted")
    reset_retriever_cache()
    r = get_retriever("kb_rebuild")

    assert r.bm25_store.count() == 0, "损坏 pkl 应被隔离, BM25 空库启动"
    assert r.vector_store.count() == 3, "向量侧不受影响"
    assert (tmp_path / "bm25_kb_rebuild.pkl.corrupt").exists()
    assert check_store_consistency("kb_rebuild")["consistent"] is False

    restored = r.rebuild_bm25_from_vectors()
    assert restored == 3
    assert r.bm25_store.count() == 3
    assert check_store_consistency("kb_rebuild")["consistent"] is True

    # 回灌的是内容而不只是数量: id -> 文本/元数据要对得上
    by_id = {d["id"]: d for d in r.bm25_store.docs}
    assert by_id["d1_c0"]["text"] == "文档 1 的正文内容 alpha1"
    assert by_id["d1_c0"]["metadata"]["doc_id"] == "d1"

    # 落盘生效: 再"重启"一次, 从 pkl 恢复而不是又空库
    reset_retriever_cache()
    r2 = get_retriever("kb_rebuild")
    assert r2.bm25_store.count() == 3

    # BM25 路检索真的通了: 精确 token 只存在于 d1_c0
    hits = r2.bm25_store.query("alpha1", top_k=5)
    assert hits and hits[0]["id"] == "d1_c0"


def test_rebuild_does_not_touch_vector_side(real_stores_in_tmp):
    """回灌只写 BM25: 向量库的条数不得变化 (它是权威副本)."""
    from app.rag.retriever import get_retriever

    _seed("kb_rebuild2", 2)
    r = get_retriever("kb_rebuild2")
    assert r.vector_store.count() == 2

    r.rebuild_bm25_from_vectors()

    assert r.vector_store.count() == 2


# ----- API 层: 契约 + 鉴权/ACL (fakes, 与 test_document_delete 同一模式) -----


@pytest.fixture()
def client(monkeypatch, tmp_path):
    from app.agents import llm_router as lr_mod
    from app.core import kb_registry as kb_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    from tests._fakes import (
        FakeBM25Store,
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


def test_rebuild_endpoint_restores_bm25(client):
    """端到端: 上传 -> BM25 丢失 -> POST rebuild-bm25 -> BM25 恢复."""
    from app.rag.retriever import get_retriever

    token = _register(client, "alice")
    _upload(client, token, "kb_alice", "这是第一篇文档。包含一些关键词。")
    r = get_retriever("kb_alice")
    assert r.bm25_store.count() > 0
    r.bm25_store.docs = []  # 模拟 BM25 全丢 (隔离/丢更新后的形态)

    resp = client.post(
        "/api/v1/documents/rebuild-bm25?kb_id=kb_alice",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["status"] == "rebuilt"
    assert data["chunks_before"] == 0
    assert data["chunks_restored"] > 0
    assert get_retriever("kb_alice").bm25_store.count() > 0


def test_rebuild_endpoint_requires_auth(client):
    resp = client.post("/api/v1/documents/rebuild-bm25")
    assert resp.status_code == 401


def test_rebuild_endpoint_respects_kb_acl(client):
    """别人的私有 kb 不可触发回灌 (与读接口同一 ACL)."""
    token_a = _register(client, "alice")
    _upload(client, token_a, "kb_alice", "内容。")
    token_b = _register(client, "bob")

    resp = client.post(
        "/api/v1/documents/rebuild-bm25?kb_id=kb_alice",
        headers={"Authorization": f"Bearer {token_b}"},
    )
    assert resp.status_code == 403


def test_rebuild_endpoint_public_kb_requires_admin(client):
    """公共 kb 不归任何用户所有, 任意访客不得触发全量重建 (403).

    公共 kb 是全站共享的只读库, 重建是 O(N) 读 + 落盘的运维级操作; 读 ACL
    放行等于把它开放给所有登录用户.
    """
    token = _register(client, "carol")

    resp = client.post(
        "/api/v1/documents/rebuild-bm25?kb_id=default",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 403, resp.text


def test_rebuild_endpoint_public_kb_allowed_for_admin(client, monkeypatch):
    """settings.admin_users 白名单里的用户可以重建公共 kb."""
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "admin_users", "carol")
    token = _register(client, "carol")

    resp = client.post(
        "/api/v1/documents/rebuild-bm25?kb_id=default",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["chunks_restored"] == 0  # 空 kb 幂等, 0 条也 200


def test_admin_flag_does_not_leak_to_other_kbs(client, monkeypatch):
    """管理员只对公共 kb 放行: 别人的私有 kb 仍然 403 (不提权)."""
    from app.core.config import get_settings

    monkeypatch.setattr(get_settings(), "admin_users", "carol")
    alice_token = _register(client, "alice")
    _upload(client, alice_token, "kb_alice_priv", "内容。")
    carol_token = _register(client, "carol")

    resp = client.post(
        "/api/v1/documents/rebuild-bm25?kb_id=kb_alice_priv",
        headers={"Authorization": f"Bearer {carol_token}"},
    )
    assert resp.status_code == 403, resp.text


# ----- /documents/list 的一致性判据与权限标志 (前端据此显示告警与按钮) -----


def test_list_exposes_consistency_true_when_healthy(client):
    token = _register(client, "alice")
    _upload(client, token, "kb_alice_ok", "这是文档内容。")

    r = client.get(
        "/api/v1/documents/list?kb_id=kb_alice_ok",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["consistent"] is True
    assert data["can_rebuild"] is True  # owner 可重建自己的 kb


def test_list_reports_inconsistency_after_bm25_loss(client):
    """BM25 丢数据时 list 必须报不一致, 且 owner 仍可重建."""
    from app.rag.retriever import get_retriever

    token = _register(client, "alice")
    _upload(client, token, "kb_alice_warn", "这是文档内容。包含关键词。")
    r = get_retriever("kb_alice_warn")
    assert r.bm25_store.count() > 0
    r.bm25_store.docs = []  # 模拟 pkl 损坏被隔离后的空 BM25

    resp = client.get(
        "/api/v1/documents/list?kb_id=kb_alice_warn",
        headers={"Authorization": f"Bearer {token}"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["bm25_count"] == 0
    assert data["vector_count"] > 0
    assert data["consistent"] is False
    assert data["can_rebuild"] is True


def test_list_can_rebuild_false_for_non_owner_and_public(client, monkeypatch):
    """非 owner 与公共 kb 的 can_rebuild 均为 false (前端据此隐藏按钮)."""
    from app.core.config import get_settings

    # 显式清空白名单: 本机 .env 可能配了 ADMIN_USERS, 不固定住这个用例会随
    # 环境飘 (断言的是"非管理员"的行为).
    monkeypatch.setattr(get_settings(), "admin_users", "")

    alice_token = _register(client, "alice")
    _upload(client, alice_token, "kb_alice_ro", "内容。")
    bob_token = _register(client, "bob")

    # 私有 kb 非 owner
    r = client.get(
        "/api/v1/documents/list?kb_id=kb_alice_ro",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert r.status_code == 403  # 连读都不行, 更谈不上重建

    # 公共 kb + 非管理员: 可读但不可重建
    r = client.get(
        "/api/v1/documents/list?kb_id=default",
        headers={"Authorization": f"Bearer {bob_token}"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["can_rebuild"] is False
