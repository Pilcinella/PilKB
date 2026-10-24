"""测试 Agent 模式真的会用 state['kb_id'] 而不是默认 "default".

P0-2 bug 的根因是: chat.py 调 `build_agent_graph()` 时, `AgentState` 里没有传
`kb_id`; 检索节点内部调 `build_tools()` (无参数), 默认值是 "default". 结果是无论
用户选哪个私有 kb, agent 模式永远只检索公共库 "default", 架空了 kb_registry 的 ACL.

本测试覆盖:
1. 检索节点把 `state["kb_id"]` 透传给检索入口.
2. 端到端 ainvoke: 检索拿到的 kb_id 必须等于 state["kb_id"], 而不是 "default".
3. 一次编译的图跑多次请求时, kb_id 跟着每次请求的 state 走 (不被缓存).

注: 检索入口换过三次 —— `build_tools` 里的 hybrid_search 工具 → 模块级
`retrieve_hits` → `pipeline.retrieve_and_rerank_sync` → 现在是 **异步的**
`pipeline.retrieve_and_rerank` (图改成 async 后, 同步桥接已删除). 注入点跟着换,
但断言的目标不变: kb_id 必须跟着 state 走, 不能被写死成 "default".

**本文件曾经用直接赋值 (`pl_mod.retrieve_and_rerank_sync = ...`) 且不还原**,
泄漏的桩会让后续测试拿到空检索结果 (全量跑时 test_integration_retrieval 里那条
"桥接没真正跑完" 的断言就是这么挂的). 现在一律用 monkeypatch, 自动还原。
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from app.agents.graph import build_agent_graph, new_agent_state, rag_node
from app.rag import pipeline as pl_mod


class _RouterLLM:
    def invoke(self, _msgs):
        return AIMessage(
            content=json.dumps(
                {
                    "needs_calculator": False,
                    "needs_retrieval": True,
                    "needs_date": False,
                }
            )
        )


class _AnswerLLM:
    def invoke(self, _prompt):
        return AIMessage(content="答案")

    async def astream(self, _prompt):
        yield AIMessage(content="答案")


def _stub_retrieval(monkeypatch, captured: list, hits=None):
    """异步 `retrieve_and_rerank` 桩: 记录每次收到的 kb_id."""

    async def _fake(question, kb_id="default", retriever=None, trace=None, history=None):
        captured.append(kb_id)
        if hits is not None:
            return hits
        return [
            {
                "id": "c0",
                "document": f"ctx from kb={kb_id}",
                "metadata": {"doc_id": "d0"},
                "rerank_score": 0.9,
            }
        ]

    monkeypatch.setattr(pl_mod, "retrieve_and_rerank", _fake)
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: _RouterLLM())
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM())


async def test_kb_id_propagates_via_graph(monkeypatch):
    """端到端: 给了 alice_kb, 检索就必须查 alice_kb 而不是 default."""
    captured: list = []
    _stub_retrieval(monkeypatch, captured)

    out = await build_agent_graph().ainvoke(
        new_agent_state("hello", kb_id="alice_kb", mode="agent", engine="router")
    )

    assert captured and captured[0] == "alice_kb", (
        f"agent 模式没把 state['kb_id'] 透传给检索入口, 实际 {captured!r}"
    )
    retriever_traces = [t for t in out["trace"] if t["role"] == "retriever"]
    assert retriever_traces, "检索节点没写 trace"
    assert retriever_traces[0]["content"]["kb_id"] == "alice_kb"
    # 结构化候选必须进 sources, 否则上层构造不出引用来源.
    assert out["sources"], "sources 为空, Agent 模式的引用来源又丢了"
    assert out["sources"][0]["id"] == "c0"
    assert out["sources"][0]["score"] == 0.9
    assert out["sources"][0]["metadata"] == {"doc_id": "d0"}


async def test_distinct_kb_id_per_request(monkeypatch):
    """同一张编译好的图跑两次、用不同的 kb_id: 检索必须跟每次请求的 state 走,
    而不是缓存了第一次的."""
    captured: list = []
    _stub_retrieval(monkeypatch, captured)
    g = build_agent_graph()

    await g.ainvoke(new_agent_state("q1", kb_id="kb_one", mode="agent", engine="router"))
    await g.ainvoke(new_agent_state("q2", kb_id="kb_two", mode="agent", engine="router"))

    assert "kb_one" in captured
    assert "kb_two" in captured
    assert "default" not in captured, (
        f"agent 模式不应该默认走 default, 实际 captured={captured!r}"
    )


async def test_empty_kb_id_falls_back_to_default(monkeypatch):
    """kb_id 为空时兜底走 'default' 而不是 crash. 向后兼容保证."""
    captured: list = []
    _stub_retrieval(monkeypatch, captured)

    # 工厂本身就要兜底 (调用方拿不到 kb 时会传空串).
    assert new_agent_state("q", kb_id="").get("kb_id") == "default"
    # 绕过工厂、手工构造的 state 也要在节点里兜底.
    state = new_agent_state("q", kb_id="kb1")
    state["kb_id"] = ""
    await rag_node(state, None)

    assert captured == ["default"], f"kb_id 空时应兜底 'default', 实际 {captured!r}"


async def test_no_hits_keeps_sources_empty(monkeypatch):
    """检索不到东西时不炸, 且 sources 保持空 (上层据此返回空引用列表)."""
    captured: list = []
    _stub_retrieval(monkeypatch, captured, hits=[])

    state = new_agent_state("q", kb_id="kb1")
    out = await rag_node(state, None)

    assert out["sources"] == []
    roles = [t["role"] for t in out["trace"]]
    assert "retriever" in roles
    assert any(
        t["role"] == "retriever" and "未检索到" in str(t["content"])
        for t in out["trace"]
    )
