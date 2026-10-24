"""P2 会话记忆 (checkpointer + thread_id) 的契约测试.

覆盖:
1. 同 thread 两轮: 第二轮 reason 看得到第一轮的问题与**最终答案** (终端节点把
   答案作为 AIMessage 追加进 messages 是记忆闭环的落点).
2. 不同 thread 互相隔离 —— 跨用户/跨会话串扰是 P2 的红线 (graph 的 docstring
   里记过这笔账)。
3. rag 模式同样吃记忆: 第二轮检索改写器拿到的 history 含第一轮问答.
4. build_agent_graph 默认不带 checkpointer (直接调图的脚本/测试没有 thread_id,
   带 checkpointer 会被 langgraph 拒收) —— 唯一开启点是 pipeline._graph_events.
5. checkpoint 落 SQLite (agent_checkpoint_db): 图缓存清空重建 (模拟进程重启)
   后同 thread 的记忆仍在 —— 内存版 checkpointer 在这里会失忆.
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agents.graph import (
    aforget_thread,
    build_agent_graph,
    forget_thread,
    new_agent_state,
    reset_agent_graph_cache,
)
from app.rag import pipeline as pl_mod


class _LoopLLM:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.calls = 0

    def bind_tools(self, tools):
        return self

    def invoke(self, msgs):
        self.seen.append(list(msgs))
        self.calls += 1
        return self.script[min(self.calls - 1, len(self.script) - 1)]


class _AnswerLLM:
    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.answer)

    async def astream(self, prompt):
        self.prompts.append(prompt)
        yield AIMessage(content=self.answer)


def _texts(msgs):
    return [str(getattr(m, "content", "") or "") for m in msgs]


async def test_same_thread_sees_previous_turn(monkeypatch):
    loop_llm = _LoopLLM([AIMessage(content="收工")])
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: loop_llm)
    monkeypatch.setattr(
        "app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("第一轮答案")
    )
    g = build_agent_graph(use_memory=True)
    cfg = {"configurable": {"thread_id": "alice:sess1"}}

    # 记忆模式下 history 必须为 None: 历史由 checkpoint 管理 (见 new_agent_state)
    await g.ainvoke(
        new_agent_state("保修期多久?", kb_id="kb1", mode="agent", engine="react", history=None),
        cfg,
    )
    await g.ainvoke(
        new_agent_state("那价格呢?", kb_id="kb1", mode="agent", engine="react", history=None),
        cfg,
    )

    assert loop_llm.calls == 2
    seen2 = _texts(loop_llm.seen[1])
    assert "保修期多久?" in seen2, "第二轮必须看到第一轮的问题"
    assert "第一轮答案" in seen2, "第一轮的答案被 AIMessage 追加进了记忆"
    assert "那价格呢?" in seen2
    # 第一轮只看到本轮问题: 初始 checkpoint 为空
    assert _texts(loop_llm.seen[0])[-1] == "保修期多久?"


async def test_threads_are_isolated(monkeypatch):
    """跨 thread 串扰 = 跨用户泄漏, 这是 thread_id 设计要守住的红线."""
    loop_llm = _LoopLLM([AIMessage(content="收工")])
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: loop_llm)
    monkeypatch.setattr(
        "app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("答")
    )
    g = build_agent_graph(use_memory=True)

    await g.ainvoke(
        new_agent_state("秘密问题A", kb_id="kb1", mode="agent", engine="react", history=None),
        {"configurable": {"thread_id": "alice:s1"}},
    )
    await g.ainvoke(
        new_agent_state("普通问题B", kb_id="kb1", mode="agent", engine="react", history=None),
        {"configurable": {"thread_id": "bob:s2"}},
    )

    seen_bob = _texts(loop_llm.seen[-1])
    assert "秘密问题A" not in seen_bob, "alice 的会话内容泄进了 bob 的上下文"
    assert "普通问题B" in seen_bob


async def test_rag_mode_feeds_history_from_checkpoint(monkeypatch):
    """rag 模式的检索改写器 history 从 checkpoint 的 messages 派生 (含上一轮答案)."""
    captured: dict = {}

    async def _fake(question, kb_id="default", retriever=None, trace=None, history=None):
        captured["history"] = history
        return [
            {"id": "c0", "document": "ctx", "metadata": {"doc_id": "d0"}, "rerank_score": 1.0}
        ]

    monkeypatch.setattr(pl_mod, "retrieve_and_rerank", _fake)
    monkeypatch.setattr(
        "app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("第一轮答案")
    )
    g = build_agent_graph(use_memory=True)
    cfg = {"configurable": {"thread_id": "u:rag1"}}

    await g.ainvoke(new_agent_state("保修期多久?", kb_id="kb1", mode="rag", history=None), cfg)
    await g.ainvoke(new_agent_state("那价格呢?", kb_id="kb1", mode="rag", history=None), cfg)

    hist = captured["history"] or []
    assert [h["role"] for h in hist] == ["user", "assistant"]
    assert hist[0]["content"] == "保修期多久?"
    assert hist[1]["content"] == "第一轮答案"


def test_default_graph_has_no_checkpointer():
    """直接 build_agent_graph() 必须是无记忆版: 单测/脚本不带 thread_id,
    带 checkpointer 的图在 ainvoke 时会被 langgraph 直接拒收."""
    g_plain = build_agent_graph()
    g_mem = build_agent_graph(use_memory=True)
    assert getattr(g_plain, "checkpointer", None) is None
    assert getattr(g_mem, "checkpointer", None) is not None


def test_forget_thread_supported_or_skipped():
    """当前安装的 checkpointer 支持删除就真删; 不支持则显式返回 False, 不抛异常."""
    build_agent_graph(use_memory=True)  # 确保带记忆的图已编译
    ok = forget_thread("ghost-user:ghost-session")
    assert isinstance(ok, bool)


async def test_memory_survives_graph_rebuild(monkeypatch):
    """SQLite checkpoint 落盘后, 图重建 (模拟进程重启) 记忆不丢.

    这是 checkpointer 从 MemorySaver 换成 SqliteSaver 的回归测试: 内存版重建
    即失忆, 本测试会挂; 落盘版第二轮仍能看到第一轮问答. conftest 把
    AGENT_CHECKPOINT_DB 重定向到会话级临时目录, 不会污染真实 data/.
    """
    loop_llm = _LoopLLM([AIMessage(content="收工")])
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: loop_llm)
    monkeypatch.setattr(
        "app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("重启前答案")
    )
    cfg = {"configurable": {"thread_id": "restart:sess1"}}

    g1 = build_agent_graph(use_memory=True)
    await g1.ainvoke(
        new_agent_state("重启前的问题", kb_id="kb1", mode="agent", engine="react", history=None),
        cfg,
    )

    # 模拟进程重启: 清图缓存重建, 新图从同一个 sqlite 文件里恢复 checkpoint.
    reset_agent_graph_cache()
    g2 = build_agent_graph(use_memory=True)
    await g2.ainvoke(
        new_agent_state("重启后追问", kb_id="kb1", mode="agent", engine="react", history=None),
        cfg,
    )

    assert loop_llm.calls == 2
    seen2 = _texts(loop_llm.seen[1])
    assert "重启前的问题" in seen2, "重启后看不到第一轮的问题 (checkpoint 没落盘?)"
    assert "重启前答案" in seen2, "重启后看不到第一轮的答案"
    assert "重启后追问" in seen2


async def test_aforget_thread_deletes_persisted_checkpoint(monkeypatch):
    """删除会话必须把 SQLite 里的 checkpoint 一并删掉.

    checkpoint 落盘后, "删了 history 文件但记忆还在盘上" 从无害 bug (重启即清)
    变成永久残留 —— history.delete_session 走 aforget_thread, 这里对它做端到端
    验证: 写一条 -> 删 -> get_tuple 为 None.
    """
    loop_llm = _LoopLLM([AIMessage(content="收工")])
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: loop_llm)
    monkeypatch.setattr(
        "app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("将被删除的答案")
    )
    g = build_agent_graph(use_memory=True)
    cfg = {"configurable": {"thread_id": "forget:sess1"}}
    await g.ainvoke(
        new_agent_state("将被删除的问题", kb_id="kb1", mode="agent", engine="react", history=None),
        cfg,
    )
    assert await g.checkpointer.aget_tuple(cfg) is not None, "先确认 checkpoint 真的落了盘"

    assert await aforget_thread("forget:sess1") is True
    assert await g.checkpointer.aget_tuple(cfg) is None, "删除后 checkpoint 仍在盘上"


def test_forget_thread_sync_returns_false_for_async_saver():
    """同步 forget_thread 对异步 checkpointer 必须返回 False 而不是死锁/抛错.

    (它继承的同步 delete_thread 在事件循环线程里会自死锁, 见 graph.py.)"""
    build_agent_graph(use_memory=True)  # 确保带记忆的图已编译
    assert forget_thread("ghost-user:ghost-session") is False
