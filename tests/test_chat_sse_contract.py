"""SSE 线格式契约测试 (前端消费的那条协议).

`/api/v1/chat/stream` 的事件形状是**跨端约定**, 消费者在仓库外的前端:

  - `frontend/lib/api.ts` 的 streamChat: 按 `\\n\\n` 切帧, 每帧必须以 `data:` 开头,
    其余部分 `JSON.parse` 成 `{type, data}`.
  - `frontend/app/chat/page.tsx`: 只认 `token` / `source` / `error` 三种 type,
    `token` 取 `data.text`, `source` 取 `data` 本身 (SourceItem 的字段),
    `error` 取 `data.msg`.
  - `frontend/app/sources/page.tsx`: 只认 `source`, 且要 `data.chunk_id` / `content`
    / `metadata` / `score` 都在.

在这之前这条协议**零测试覆盖** —— test_kb_acl 只断言 status=200, test_smoke 只断言
路径存在。合并 RAG 编排时线格式被刻意冻结不变, 这个文件把它钉住。
"""
from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(autouse=True)
def _isolate_registry(tmp_path, monkeypatch):
    """每个测试用独立的 cwd + KB registry, 不污染真实 data/."""
    monkeypatch.chdir(tmp_path)
    from app.api import auth as auth_mod
    from app.core import kb_registry

    auth_mod.reset_users_for_tests()
    kb_registry.reset_for_tests()
    from app.core.config import get_settings

    get_settings.cache_clear()
    yield
    auth_mod.reset_users_for_tests()
    kb_registry.reset_for_tests()


class _StreamLLM:
    """带 astream() 的假 LLM.

    真实 provider 走 langchain astream (原生 async, ChatOpenAI.astream 产出
    AIMessageChunk), tests/_fakes.py 的 FakeLLM 只有 invoke —— 流式那条路径此前
    在测试里根本走不到 (缺 astream 会抛 AttributeError, 被 SSE 兜底成 error 帧,
    而断言只看 status=200)。
    """

    class _Msg:
        def __init__(self, text: str):
            self.content = text

    def __init__(self, chunks=None, fail_on_stream: bool = False):
        # 默认把 <think> 标签**故意切开**成跨 chunk, 覆盖状态机过滤器的核心场景.
        self._chunks = (
            ["<thi", "nk>思考中, 不该出现在前端</thi", "nk>", "根据检索结果, 回答如下 [1]"]
            if chunks is None
            else chunks
        )
        self._fail = fail_on_stream

    def invoke(self, prompt):
        return self._Msg("根据检索结果, 回答如下 [1]")

    async def astream(self, prompt):
        if self._fail:
            raise RuntimeError("provider exploded")
        for c in self._chunks:
            yield self._Msg(c)


@pytest.fixture
def client(monkeypatch):
    """装好 fakes 的 TestClient, LLM 用可流式的 _StreamLLM."""
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
    # 替 pipeline 命名空间里的 get_reranker (它在 import 时就被绑定了).
    monkeypatch.setattr(pl_mod, "get_reranker", lambda *_a, **_k: FakeReranker())
    _install_llm(monkeypatch, lr_mod, _StreamLLM())

    from app.main import app

    with TestClient(app) as c:
        yield c

    emb_mod.get_embedding_service.cache_clear()
    _os.environ.pop("PILKB_FAKE_EMBEDDING", None)


def _install_llm(monkeypatch, lr_mod, llm) -> None:
    # 签名带两个可选参数: pipeline 的 get_llm_text_callable() 是按
    # get_llm(provider, model) 调的, 用 lambda: ... 会 TypeError (并被 except 吞掉).
    monkeypatch.setattr(lr_mod, "get_llm", lambda provider=None, model=None: llm)


def _register(client: TestClient, username: str) -> str:
    r = client.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": "password123"},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _seed_kb(client: TestClient, token: str, kb_id: str) -> None:
    r = client.post(
        "/api/v1/documents/upload",
        files={"file": ("note.txt", b"hello world, this is a note about prices\n", "text/plain")},
        data={"kb_id": kb_id},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text


def _parse_sse(text: str) -> list[dict]:
    """按前端的解析口径切开帧, 顺便把线格式断言掉."""
    events: list[dict] = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        assert block.startswith("data:"), f"帧必须以 'data:' 开头, 实际: {block[:60]!r}"
        events.append(json.loads(block[5:].strip()))
    return events


def test_rag_stream_frame_format_and_order(client):
    """RAG 分支: 帧格式合法, 顺序为 source* -> token* -> done."""
    token = _register(client, "alice")
    _seed_kb(client, token, "alice_kb")

    r = client.post(
        "/api/v1/chat/stream",
        json={"question": "定价规则是什么?", "kb_id": "alice_kb", "mode": "rag", "top_k": 3},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    assert events, "SSE 一个事件都没发"

    kinds = [e["type"] for e in events]
    # done 必须是最后一帧, 且不带 data (线格式如此, 前端也靠它判断正常收尾)
    assert kinds[-1] == "done", f"最后一帧应为 done, 实际 {kinds}"
    assert "data" not in events[-1], f"done 帧不应带 data: {events[-1]!r}"
    # rag 模式不产 trace (那是 agent 分支的)
    assert "trace" not in kinds
    # 有 source 才有引用溯源; 顺序上 source 必须全部先于 token
    assert "source" in kinds, f"没有 source 帧: {kinds}"
    assert kinds.index("token") > max(i for i, k in enumerate(kinds) if k == "source")
    assert "error" not in kinds, f"不该有 error 帧: {events}"

    # source 帧的字段是 frontend/app/sources/page.tsx 与 chat/page.tsx 共同依赖的
    src = next(e for e in events if e["type"] == "source")["data"]
    for key in ("doc_id", "chunk_id", "content", "score", "metadata"):
        assert key in src, f"source 帧缺少 {key}: {src!r}"
    assert src["chunk_id"], "chunk_id 不能为空, 前端用它做 key"
    assert isinstance(src["score"], float)


def test_token_stream_filters_think_block(client, monkeypatch):
    """<think> 块即使被切到多个 chunk 之间, 也不能流到前端."""
    from app.agents import llm_router as lr_mod

    token = _register(client, "bob")
    _seed_kb(client, token, "bob_kb")
    # 换一个"标签跨 chunk"的假 LLM (fixture 里那个已经是, 这里显式再钉一次)
    _install_llm(
        monkeypatch,
        lr_mod,
        _StreamLLM(chunks=["<thi", "nk>不该出现</thi", "nk>", "答案在这里"]),
    )

    r = client.post(
        "/api/v1/chat/stream",
        json={"question": "定价规则?", "kb_id": "bob_kb", "mode": "rag", "top_k": 2},
        headers={"Authorization": f"Bearer {token}"},
    )
    events = _parse_sse(r.text)
    streamed = "".join(e["data"]["text"] for e in events if e["type"] == "token")
    assert "不该出现" not in streamed, f"思考内容漏到前端了: {streamed!r}"
    assert "think" not in streamed.lower()
    assert streamed == "答案在这里", f"正文拼接结果不对: {streamed!r}"


def test_stream_error_emits_error_frame_and_no_done(client, monkeypatch):
    """生成阶段失败: 发 error 帧, 且**不发** done —— 否则前端会把半截答案当完整结果."""
    from app.agents import llm_router as lr_mod

    token = _register(client, "carol")
    _seed_kb(client, token, "carol_kb")
    _install_llm(monkeypatch, lr_mod, _StreamLLM(fail_on_stream=True))

    r = client.post(
        "/api/v1/chat/stream",
        json={"question": "定价规则?", "kb_id": "carol_kb", "mode": "rag", "top_k": 2},
        headers={"Authorization": f"Bearer {token}"},
    )
    # 已经开流, HTTP 状态码固定 200, 失败只能以 error 事件表达
    assert r.status_code == 200
    events = _parse_sse(r.text)
    kinds = [e["type"] for e in events]
    assert "error" in kinds, f"失败必须发 error 帧: {events}"
    assert "done" not in kinds, f"出错后不应再发 done: {kinds}"
    # error 帧是通用文案, 不携带 str(e) 的内部细节 (provider 报文/路径等)
    msg = next(e for e in events if e["type"] == "error")["data"]["msg"]
    assert "provider exploded" not in msg and msg


# ---------------------------------------------------------------------------
# agent 模式
#
# 改造前 agent 侧**零覆盖**: 它的帧序 (trace 先于 source) 与 rag 不同, 却没有任何
# 断言钉住。这里补上, 顺便把"一张图两种 mode"的对外表现固定下来。
# ---------------------------------------------------------------------------


class _RouterLLM:
    """路由节点专用假 LLM: 只在看到路由提示词时返回决策, 其余情况不该被调到."""

    def __init__(self, decision: dict):
        self._decision = json.dumps(decision)

    def invoke(self, msgs):
        head = msgs[0].content if msgs else ""
        if "任务规划 Agent" in head:
            return _StreamLLM._Msg(self._decision)
        return _StreamLLM._Msg("(不该走到这里)")


def _install_router(monkeypatch, **decision) -> None:
    """装路由假 LLM, 并把引擎切到 router.

    注意 patch 点是 ``app.agents.graph.get_llm`` (模块级绑定), 不是 llm_router 里
    那个 —— 图节点走前者, pipeline 的生成走后者, 两者必须分别替换。
    引擎必须显式切: 默认已是 react, 这些契约断言的 trace 角色是 router 引擎的。
    """
    from app.core.config import settings

    monkeypatch.setattr(settings, "agent_engine", "router")
    monkeypatch.setattr(
        "app.agents.graph.get_llm", lambda *a, **k: _RouterLLM(decision)
    )


def test_agent_stream_emits_trace_then_sources_then_tokens(client, monkeypatch):
    """agent 检索分支: trace* -> source* -> token* -> done, 且 done 无 data."""
    _install_router(monkeypatch, needs_retrieval=True, needs_calculator=False, needs_date=False)
    token = _register(client, "dave")
    _seed_kb(client, token, "dave_kb")

    r = client.post(
        "/api/v1/chat/stream",
        json={"question": "定价规则是什么?", "kb_id": "dave_kb", "mode": "agent", "top_k": 3},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    kinds = [e["type"] for e in events]

    assert kinds[-1] == "done" and "data" not in events[-1]
    assert "error" not in kinds, f"不该有 error 帧: {events}"
    assert "source" in kinds, f"agent 模式也要有引用来源: {kinds}"
    # 第一帧必须是路由轨迹: 用户先看到"在做什么", 而不是干等
    assert kinds[0] == "trace", f"第一帧应是路由轨迹, 实际 {kinds}"
    roles = [e["data"].get("role") for e in events if e["type"] == "trace"]
    assert "router" in roles and "retriever" in roles, f"trace 缺少节点轨迹: {roles}"
    # 引用来源先于正文 —— 与 rag 模式保持同一条约定。
    # (注意不能断言"所有 trace 都在 token 之前": 生成结束后还会补一条 answer 轨迹.)
    first_token = kinds.index("token")
    assert max(i for i, k in enumerate(kinds) if k == "source") < first_token


def test_agent_direct_answer_emits_notice_not_source(client, monkeypatch):
    """路由判不需要检索: 发 notice 提示, 不发 source (没查到任何东西)."""
    _install_router(monkeypatch, needs_retrieval=False, needs_calculator=False, needs_date=False)
    token = _register(client, "erin")
    _seed_kb(client, token, "erin_kb")

    r = client.post(
        "/api/v1/chat/stream",
        json={"question": "我们公司报销标准是多少?", "kb_id": "erin_kb", "mode": "agent", "top_k": 3},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200, r.text
    events = _parse_sse(r.text)
    kinds = [e["type"] for e in events]

    assert "source" not in kinds, f"无检索不应有 source 帧: {kinds}"
    notices = [e for e in events if e["type"] == "notice"]
    assert notices, f"直答必须带可见标记: {kinds}"
    assert "未使用知识库" in notices[0]["data"]["text"]
    # 标记必须在正文之前到达, 否则用户先读了答案才被告知"没查库"
    assert kinds.index("notice") < kinds.index("token")
    # 标记绝不能混进正文 (正文会被送去 RAGAS 打分与历史落库)
    streamed = "".join(e["data"]["text"] for e in events if e["type"] == "token")
    assert "未使用知识库" not in streamed


def test_agent_stream_error_after_sources(client, monkeypatch):
    """agent 生成失败: 已发出的 source 帧必须排在 error 之前, 且不发 done."""
    from app.agents import llm_router as lr_mod

    _install_router(monkeypatch, needs_retrieval=True, needs_calculator=False, needs_date=False)
    token = _register(client, "frank")
    _seed_kb(client, token, "frank_kb")
    # 生成阶段炸掉 (路由仍正常); 事件经队列转交, 顺序不能乱.
    _install_llm(monkeypatch, lr_mod, _StreamLLM(fail_on_stream=True))

    r = client.post(
        "/api/v1/chat/stream",
        json={"question": "定价规则?", "kb_id": "frank_kb", "mode": "agent", "top_k": 2},
        headers={"Authorization": f"Bearer {token}"},
    )
    assert r.status_code == 200
    events = _parse_sse(r.text)
    kinds = [e["type"] for e in events]

    assert "error" in kinds, f"失败必须发 error 帧: {events}"
    assert "done" not in kinds, f"出错后不应再发 done: {kinds}"
    # error 帧是通用文案, 不携带 str(e) 的内部细节 (provider 报文/路径等)
    msg = next(e for e in events if e["type"] == "error")["data"]["msg"]
    assert "provider exploded" not in msg and msg
    # 检索是先跑完的, 它的引用来源不能因为后面生成失败就消失
    assert "source" in kinds, f"已发出的 source 帧被 error 挤掉了: {kinds}"
    assert max(i for i, k in enumerate(kinds) if k == "source") < kinds.index("error")
