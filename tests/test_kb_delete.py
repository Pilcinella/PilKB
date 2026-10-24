"""DELETE /api/v1/kbs: 删除整个知识库 (权限 + 注册表 + 存储清理).

覆盖:
1. owner 删除自己的 kb: 200, 列表不再出现; 向量 collection reset、BM25 pkl
   一并清除 (delete_kb_storage 的两个落点)。
2. 其他人删除别人的私有 kb: 403 (越权红线)。
3. 删除不存在的 kb: 404; 删除后再删: 404 (幂等语义来自注册表, 不复活)。
4. 公共 kb (default): 普通用户 403 —— 公共库没有 owner, 只有管理员能删。
5. 管理员 (ADMIN_USERS 白名单) 可删公共 kb。

与 test_kb_acl 同一套 TestClient + fakes 模式, 不依赖真实重型模型。
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tests._fakes import FakeVectorStore


# 记录 reset() 调用过的 collection, 用于断言"向量库真的被清了"
class _RecordingStore(FakeVectorStore):
    resets: list[str] = []

    def reset(self):
        _RecordingStore.resets.append(self.collection_name)
        self.docs = []


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """每个测试用独立的 cwd + KB registry, 不污染真实 data/. (同 test_kb_acl)"""
    monkeypatch.chdir(tmp_path)
    from app.api import auth as auth_mod
    from app.core import kb_registry
    from app.core.config import get_settings

    auth_mod.reset_users_for_tests()
    kb_registry.reset_for_tests()
    get_settings.cache_clear()
    _RecordingStore.resets.clear()
    yield
    auth_mod.reset_users_for_tests()
    kb_registry.reset_for_tests()
    get_settings.cache_clear()


@pytest.fixture
def client(monkeypatch):
    """装好 fakes 的 TestClient; ChromaStore 换成带 reset 记录的版本."""
    import os as _os

    from app.agents import llm_router as lr_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    from tests._fakes import (
        FakeBM25Store,
        FakeEmbeddingService,
        FakeLLM,
        FakeReranker,
    )

    _os.environ["PILKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()

    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rk_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    # delete_kb_storage 引用的是 retriever 模块命名空间里的 ChromaStore
    monkeypatch.setattr(rt_mod, "ChromaStore", _RecordingStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(lr_mod, "get_llm", lambda: FakeLLM())

    from app.main import app

    with TestClient(app) as c:
        yield c

    emb_mod.get_embedding_service.cache_clear()
    _os.environ.pop("PILKB_FAKE_EMBEDDING", None)


def _register(client: TestClient, username: str, password: str = "password123") -> str:
    r = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _make_bm25_pkl(kb_id: str) -> Path:
    """在配置的 BM25 目录里伪造一个索引文件, 断言删除接口会把它清掉."""
    from app.core.config import get_settings

    p = Path(get_settings().bm25_persist_dir) / f"bm25_{kb_id}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"fake-bm25-index")
    return p


# ----- 1. owner 删除: 注册表 + 向量库 + BM25 三个落点 -----
def test_owner_delete_removes_kb_and_storage(client):
    token = _register(client, "alice")
    r = client.post("/api/v1/kbs", json={"kb_id": "alice_kb"}, headers=_auth(token))
    assert r.status_code == 200, r.text

    pkl = _make_bm25_pkl("alice_kb")

    r = client.delete("/api/v1/kbs/alice_kb", headers=_auth(token))
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "deleted", "kb_id": "alice_kb"}

    # 注册表: 列表不再包含
    r = client.get("/api/v1/kbs", headers=_auth(token))
    assert r.status_code == 200
    assert "alice_kb" not in [k["kb_id"] for k in r.json()]
    # 向量库: reset 被调用且 collection 名对得上
    assert "kb_alice_kb" in _RecordingStore.resets
    # BM25: pkl 已清除
    assert not pkl.exists()
    # 再删一次: 404 (不复活, 幂等)
    r = client.delete("/api/v1/kbs/alice_kb", headers=_auth(token))
    assert r.status_code == 404


# ----- 2. 越权删除别人的私有 kb: 403 -----
def test_other_user_cannot_delete_private_kb(client):
    token_a = _register(client, "alice")
    token_b = _register(client, "bob")
    client.post("/api/v1/kbs", json={"kb_id": "alice_kb"}, headers=_auth(token_a))

    r = client.delete("/api/v1/kbs/alice_kb", headers=_auth(token_b))
    assert r.status_code == 403, r.text

    # kb 仍在, 数据未被动过
    r = client.get("/api/v1/kbs", headers=_auth(token_a))
    assert "alice_kb" in [k["kb_id"] for k in r.json()]
    assert _RecordingStore.resets == []


# ----- 3. 删除不存在的 kb: 404 -----
def test_delete_missing_kb_is_404(client):
    token = _register(client, "alice")
    r = client.delete("/api/v1/kbs/ghost_kb", headers=_auth(token))
    assert r.status_code == 404


# ----- 4. 公共 kb: 普通用户 403 -----
def test_public_kb_requires_admin(client):
    token = _register(client, "alice")
    r = client.delete("/api/v1/kbs/default", headers=_auth(token))
    assert r.status_code == 403, r.text
    assert _RecordingStore.resets == []


# ----- 5. 管理员可删公共 kb -----
def test_admin_can_delete_public_kb(client, monkeypatch):
    monkeypatch.setenv("ADMIN_USERS", "root1")
    from app.core.config import get_settings

    get_settings.cache_clear()
    token = _register(client, "root1")

    r = client.delete("/api/v1/kbs/default", headers=_auth(token))
    assert r.status_code == 200, r.text
    assert "kb_default" in _RecordingStore.resets
