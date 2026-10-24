"""P2 记忆在**真实 HTTP 链路**上的回归测试.

为什么单独一个文件: 此前 test_agent_memory.py 全部直接 ``graph.ainvoke(state, cfg)``,
把 API 层 (chat.py 的 session_id 透传) 与 pipeline 的接线层 (``_graph_events`` 计算
thread_id / use_memory) 整个绕过去了 —— 这两层任何一处断掉, 图级测试都照不出来。
2026-10-16 用户在 UI 上实测"同一页面连续追问也失忆", 就是从这个盲区里冒出来的。
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """每个测试用独立的 cwd + KB registry, 不污染真实 data/ (同 SSE 契约文件)."""
    monkeypatch.chdir(tmp_path)
    from app.api import auth as auth_mod
    from app.core import kb_registry
    from app.core.config import get_settings

    auth_mod.reset_users_for_tests()
    kb_registry.reset_for_tests()
    get_settings.cache_clear()
    yield
    auth_mod.reset_users_for_tests()
    kb_registry.reset_for_tests()


class _RecordingReasonLLM:
    """reason 决策 LLM: 记录每次收到的完整消息, 永不调工具 (首轮即收工)."""

    def __init__(self):
        self.seen: list[list[str]] = []

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        self.seen.append([str(getattr(m, "content", "") or "") for m in msgs])
        return AIMessage(content="收工")


class _ComposeLLM:
    def __init__(self):
        self.prompts: list[str] = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content="答案文本")

    async def astream(self, prompt):
        self.prompts.append(prompt)
        yield AIMessage(content="答案文本")


@pytest.fixture
def wired(monkeypatch):
    """装 fakes + 两个 patch 点的 recording LLM, 返回 (client, reason_llm)."""
    import os as _os

    from app.agents import llm_router as lr_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import pipeline as pl_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod
    from tests._fakes import (
        FakeBM25Store,
        FakeEmbeddingService,
        FakeReranker,
        FakeVectorStore,
    )

    _os.environ["PILKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()

    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(pl_mod, "get_reranker", lambda *_a, **_k: FakeReranker())

    reason_llm = _RecordingReasonLLM()
    composer = _ComposeLLM()
    # 两个 patch 点不能混: 图节点走 graph.get_llm (模块级绑定), 生成走 llm_router
    # 的模块属性查找 (pipeline._generate_events).
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: reason_llm)
    monkeypatch.setattr(lr_mod, "get_llm", lambda provider=None, model=None: composer)

    from app.main import app

    with TestClient(app) as c:
        yield c, reason_llm, composer

    emb_mod.get_embedding_service.cache_clear()
    _os.environ.pop("PILKB_FAKE_EMBEDDING", None)


def _register(client: TestClient, username: str) -> str:
    r = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": "password123"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _ask(client: TestClient, token: str, question: str, session_id: str, mode: str = "agent"):
    # kb 用公共的 "default": 本测试盯的是记忆接线, 不需要真实语料; 私有 kb 还要
    # 先注册, 平白多一层 ACL 依赖。
    return client.post(
        "/api/v1/chat/stream",
        json={"question": question, "kb_id": "default", "mode": mode, "top_k": 3, "session_id": session_id},
        headers={"Authorization": f"Bearer {token}"},
    )


def test_same_session_remembers_previous_turn(wired):
    """同一 session_id 连续两问: 决策轮**与生成轮**都必须看得到第一轮."""
    client, reason_llm, composer = wired
    token = _register(client, "memuser")

    r1 = _ask(client, token, "第一轮问题", "sess_mem_1")
    assert r1.status_code == 200, r1.text
    r2 = _ask(client, token, "第二轮追问", "sess_mem_1")
    assert r2.status_code == 200, r2.text

    assert len(reason_llm.seen) >= 2, f"reason 只被调了 {len(reason_llm.seen)} 次"
    turn2 = reason_llm.seen[-1]
    assert any("第一轮问题" in m for m in turn2), (
        f"第二轮看不到第一轮的问题, 记忆没生效。第二轮实际看到: {turn2}"
    )
    assert any("答案文本" in m for m in turn2), "第一轮的答案没被追加进记忆"

    # 生成轮的提示词同样要带历史 (2026-10-16 用户实测的真 bug: 决策轮有记忆、
    # 生成轮没有, 追问被模型原样吐回). 只测 reason 的视角漏掉了它。
    assert len(composer.prompts) == 2
    assert "【对话历史】" not in composer.prompts[0]
    assert "第一轮问题" in composer.prompts[1] and "答案文本" in composer.prompts[1], (
        f"第二轮生成提示词缺历史: {composer.prompts[1][:300]}"
    )


def test_different_sessions_do_not_leak(wired):
    """不同 session_id 之间必须隔离 (跨会话泄漏是 P2 的红线)."""
    client, reason_llm, _composer = wired
    token = _register(client, "memuser2")

    _ask(client, token, "秘密问题A", "sess_a")
    _ask(client, token, "普通问题B", "sess_b")

    assert not any("秘密问题A" in m for m in reason_llm.seen[-1]), "跨会话泄漏"


def test_missing_session_id_degrades_to_single_turn(wired):
    """前端没带 session_id 时必须退回单轮 (不炸, 也不串到别人的 thread)."""
    client, reason_llm, _composer = wired
    token = _register(client, "memuser3")

    r1 = client.post(
        "/api/v1/chat/stream",
        json={"question": "无会话问题", "kb_id": "default", "mode": "agent", "top_k": 3},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r1.status_code == 200, r1.text
    assert any("无会话问题" in m for m in reason_llm.seen[-1])
