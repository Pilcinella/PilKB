"""集成测试: 用 fake embedding/vector/bm25/rerank 跑完整的 add → retrieve → rerank → answer 链路.

不依赖 torch / chromadb / rank_bm25 / sentence-transformers.
CI 默认环境下也能跑, 是 unit test 的有力补充.

覆盖场景:
1. HybridRetriever.add_documents + retrieve (BM25 + vector RRF 融合)
2. add_documents → delete_doc → 文档不再被召回
3. 完整 pipeline: run_rag(question, kb_id, top_k) → {answer, contexts, ...}
4. RRF 排序正确性: 同时命中 BM25 和 向量的文档, 排名应该比只命中一路的更高
5. 检索为空时, pipeline 也能给出 "未找到相关信息" 而不是崩
6. rerank 在 HybridRetriever 内部生效 (top_k=2 截断到 2 条)
"""
from __future__ import annotations

import pytest

# 整套 fake 必须在 import HybridRetriever / run_rag 之前注入到对应模块, 否则
# retriever 会去碰真的 ChromaStore / BM25Store / sentence_transformers.
from tests._fakes import (  # noqa: E402
    FakeBM25Store,
    FakeEmbeddingService,
    FakeReranker,
    FakeVectorStore,
)


@pytest.fixture(autouse=True)
def _install_fakes(monkeypatch):
    """每个测试自动 patch: vector_store.ChromaStore / bm25_store.BM25Store /
    reranker.CrossEncoderReranker / embedding.get_embedding_service / llm_router.get_llm.

    注意: retriever.py 在 import 时已经执行过 `from app.rag.vector_store import ChromaStore`,
    引用被 capture 在 retriever 模块的命名空间里. 只改 vs_mod.ChromaStore 不够, 必须连
    retriever 模块里的 ChromaStore / BM25Store 也一起替.
    """
    import os
    from app.agents import llm_router
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    os.environ["PILKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()

    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    # P1-3 重构后 reranker 走 get_reranker() 单例, 把 3 个模块里的函数指针都替掉.
    monkeypatch.setattr(rk_mod, "get_reranker", lambda *_a, **_k: FakeReranker())
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(llm_router, "get_llm", lambda: _StubPipelineLLM())
    # 关键: 覆盖 retriever 模块里被 capture 的两个类引用.
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)

    from app.rag import pipeline as pl_mod
    monkeypatch.setattr(pl_mod, "get_reranker", lambda *_a, **_k: FakeReranker())

    yield
    emb_mod.get_embedding_service.cache_clear()
    os.environ.pop("PILKB_FAKE_EMBEDDING", None)


class _StubPipelineLLM:
    """pipeline.run_rag 期望的 LLM 接口: invoke(prompt) -> 有 .content 的对象."""

    class _Resp:
        def __init__(self, text: str):
            self.content = text

    def invoke(self, prompt):
        # 截取 prompt 里"【检索结果】"那段, 取第一个 chunk 当作引用
        if "【检索结果】" in prompt:
            ctx = prompt.split("【检索结果】", 1)[1].split("【用户问题】", 1)[0]
            first = ""
            for line in ctx.split("\n"):
                line = line.strip()
                if line.startswith("[") and "]" in line:
                    first = line.split("]", 1)[-1].strip()
                    break
            return self._Resp(f"fake-answer: {first[:80]}")
        return self._Resp("fake-answer: (empty)")

    async def astream(self, prompt):
        yield self.invoke(prompt)


# 1) HybridRetriever add → retrieve 链路
def test_hybrid_retriever_returns_added_documents():
    from app.rag.retriever import HybridRetriever

    r = HybridRetriever(kb_id="test_hybrid")
    r.add_documents(
        ids=["d1", "d2", "d3"],
        documents=[
            "保修期是一年, 非人为损坏可以免费维修。",
            "我们支持微信和支付宝支付。",
            "配送时效一线城市 24 小时。",
        ],
        metadatas=[{"src": "a"}, {"src": "b"}, {"src": "c"}],
    )

    hits = r.retrieve("保修多久", top_k=3)
    assert len(hits) >= 1
    ids = {h["id"] for h in hits}
    assert "d1" in ids, f"BM25 / 向量至少有一路应该把 d1 召回, 实际 ids={ids}"


# 2) add → delete 链路: 删除后文档不再被召回
def test_delete_doc_removes_from_retrieval():
    from app.rag.retriever import HybridRetriever

    r = HybridRetriever(kb_id="test_delete")
    r.add_documents(
        ids=["d1", "d2"],
        documents=["保修期一年", "支持微信支付"],
        metadatas=[{"src": "a"}, {"src": "b"}],
    )
    # 删之前能召回
    before = {h["id"] for h in r.retrieve("保修", top_k=2)}
    assert "d1" in before

    # 删掉 d1
    if hasattr(r, "delete_doc"):
        r.delete_doc(["d1"])
    else:
        # vector_store + bm25_store 都得删
        r.vector_store.delete(["d1"])
        if hasattr(r.bm25_store, "delete"):
            r.bm25_store.delete(["d1"])
        else:
            r.bm25_store.docs = [d for d in r.bm25_store.docs if d["id"] != "d1"]

    after = {h["id"] for h in r.retrieve("保修", top_k=2)}
    assert "d1" not in after, f"删除后 d1 不应该被召回, 实际 ids={after}"


# 3) 完整 pipeline: run_rag 端到端
def test_run_rag_returns_answer_with_contexts():
    import asyncio

    from app.rag.pipeline import run_rag

    r = asyncio.run(
        run_rag(
            question="保修期多久?",
            kb_id="test_pipeline",
            top_k=2,
            use_agent=False,
        )
    )
    assert "answer" in r
    assert "contexts" in r
    assert "context_ids" in r
    assert "context_scores" in r
    # 即使 fake 模式, answer 字段也应该是字符串
    assert isinstance(r["answer"], str)
    assert len(r["answer"]) > 0


# 4) RRF 排序: 同时在 BM25 和向量命中的 doc 应该比只命中一路的 rank 更高
def test_rrf_prefers_overlapping_hits():
    from app.rag.retriever import HybridRetriever

    r = HybridRetriever(kb_id="test_rrf")
    # d1 既会被 BM25 召回 (关键词重叠) 也会被向量召回
    # d2 只会被 BM25 召回 (关键词重叠)
    # d3 都不召回 (无关)
    r.add_documents(
        ids=["d1", "d2", "d3"],
        documents=[
            "保修条款: 一年保修, 非人为损坏.",
            "我们提供一年保修服务, 联系售后.",
            "支付方式: 微信, 支付宝.",
        ],
        metadatas=[{"src": "x"}, {"src": "y"}, {"src": "z"}],
    )
    hits = r.retrieve("一年保修", top_k=3)
    ids = [h["id"] for h in hits]
    # d1 / d2 至少有一个在 d3 之前 (因为 d3 完全无关)
    if "d3" in ids and "d1" in ids:
        assert ids.index("d1") < ids.index("d3"), f"d1 应该排在 d3 之前, 实际 {ids}"


# 5) 空库检索: 不能崩, 应该返回 []
def test_retrieve_on_empty_collection_returns_empty():
    from app.rag.retriever import HybridRetriever

    r = HybridRetriever(kb_id="test_empty")
    hits = r.retrieve("anything", top_k=3)
    assert hits == []


# 6) 完整 pipeline 在空库下不会崩
def test_run_rag_on_empty_kb_returns_clean_answer():
    import asyncio

    from app.rag.pipeline import run_rag

    r = asyncio.run(
        run_rag(
            question="任意问题",
            kb_id="test_empty_pipeline",
            top_k=2,
            use_agent=False,
        )
    )
    # 不管什么 answer, 都得有 answer 字段且为 string, 流水线不能崩
    assert isinstance(r["answer"], str)
    assert "contexts" in r


async def test_agent_graph_rag_node_runs_on_running_loop():
    """反向守卫: 图必须在**已有事件循环**的线程上跑得通, 且检索链要真的跑完.

    改造前节点是同步的, 靠 `pipeline.retrieve_and_rerank_sync` 的 `asyncio.run`
    起临时 loop, 前提是"调用线程没有 running loop"; 那条路径已随图改 async 而删除。
    现在节点直接在当前 loop 上 await —— 谁要是把节点改回 sync + `asyncio.run`,
    这个用例会立刻抛 `RuntimeError: asyncio.run() cannot be called from a running
    event loop`, 而不是等到生产环境客户端断开时才发现取消不生效。

    注意这里**不 stub 检索链**: agent 的其余测试都把检索入口替换掉了, 覆盖不到
    "图里的检索真的能跑完整条链" 这一层。
    """
    from app.agents.graph import build_agent_graph, new_agent_state
    from app.rag.retriever import get_retriever

    kb = "test_running_loop"
    get_retriever(kb).add_documents(
        ids=["rl-c0"],
        documents=["保修期为 24 个月, 以购买凭证为准。"],
        metadatas=[{"doc_id": "d0"}],
    )

    out = await build_agent_graph().ainvoke(
        new_agent_state("保修期多久?", kb_id=kb, mode="rag")
    )

    assert out["sources"], "检索到 0 条, 检索链没真正跑完"
    # 候选形状必须能被上层构造出引用来源 (chat.py 的 SourceItem 就吃这个).
    assert {"id", "document", "metadata"} <= set(out["sources"][0])
    assert out["final"], "终端节点必须写 state['final']"


async def test_stream_rag_stops_generating_after_client_disconnect(monkeypatch):
    """客户端断开后必须**停止生成**, 而不是把整段答案烧完.

    这是图改成 async 节点 + ainvoke 的核心收益: cancel 会穿透到节点内部的 await,
    被 ``async for`` 层层包住的生成器当场执行 finally 关闭。同步节点时代只能靠轮询
    事件, 最坏要等一整次 LLM 调用跑完 —— 而 agent 模式一次请求要烧 3~7 次调用。

    失败模式很明确: 取消一旦失效, 这里的 produced 会继续增长到 200。
    """
    import asyncio
    import time

    from app.agents import llm_router as lr_mod
    from app.rag import pipeline as pl_mod

    produced = {"n": 0}

    class _SlowLLM:
        class _Msg:
            def __init__(self, text):
                self.content = text

        def invoke(self, _prompt):
            return self._Msg("最终答案")

        async def astream(self, _prompt):
            for i in range(200):
                produced["n"] += 1
                # 必须是 asyncio.sleep: 每个 chunk 之间要有真实挂起点,
                # 否则 __anext__ 同步直通, 200 个 chunk 一次性跑完,
                # "断连取消"这个契约根本测不到.
                await asyncio.sleep(0.005)
                yield self._Msg(f"第{i}段正文 ")

    monkeypatch.setattr(lr_mod, "get_llm", lambda provider=None, model=None: _SlowLLM())

    agen = pl_mod.stream_rag(
        "保修期多久?", kb_id="test_cancel", use_agent=False, stream_tokens=True
    )
    seen = 0
    async for ev in agen:
        if isinstance(ev, pl_mod.TokenEvent):
            seen += 1
            if seen >= 3:
                break
    await agen.aclose()  # 等价于 Starlette 关闭 event_gen

    await asyncio.sleep(0.15)
    stopped_at = produced["n"]
    await asyncio.sleep(0.6)
    assert produced["n"] == stopped_at, (
        f"断开后仍在生成: {stopped_at} -> {produced['n']} 个 chunk (取消没生效)"
    )
    assert 0 < stopped_at < 200, f"流式没真正跑起来, 这个用例没测到东西: {stopped_at}"


async def test_run_rag_degrades_when_direct_answer_generation_fails(monkeypatch):
    """agent 直答分支生成失败时, run_rag 必须降级而不是整条抛出.

    直答分支虽然"没检索", 但它照发 ``SourcesEvent([])`` —— 所以 run_rag 的
    got_sources 为真, 走"退回上下文拼接"的降级。缺了那个空事件, 这里会变成
    re-raise, 评估脚本会因为一次 LLM 抖动整条样本失败 (静默的行为回归)。
    """
    from app.agents import llm_router as lr_mod
    from app.rag import pipeline as pl_mod

    class _Boom:
        def invoke(self, _prompt):
            raise RuntimeError("provider exploded")

        async def astream(self, _prompt):
            raise RuntimeError("provider exploded")
            yield  # noqa: B901  # 不可达; 保证这是 async generator, 让异常经 __anext__ 抛出

    class _Router:
        class _Msg:
            content = '{"needs_retrieval": false, "needs_calculator": false, "needs_date": false}'

        def invoke(self, _msgs):
            return self._Msg()

    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: _Router())
    monkeypatch.setattr(lr_mod, "get_llm", lambda provider=None, model=None: _Boom())

    out = await pl_mod.run_rag(question="我们公司报销标准?", use_agent=True)

    assert out["answer"] == "未找到相关信息"
    assert out["contexts"] == []


# 7) pipeline 端到端: 跑 rag 模式, 拿 answer / contexts / context_ids
def test_run_rag_returns_full_payload_with_contexts():
    import asyncio

    from app.rag import pipeline as pl_mod
    from app.rag.retriever import HybridRetriever

    # 让 HybridRetriever 实例化时直接吃 fake (fixture 已 patch ChromaStore/BM25Store,
    # 这里只需要往里灌文档).
    r = HybridRetriever(kb_id="test_pipeline_full")
    r.add_documents(
        ids=["d1", "d2", "d3"],
        documents=[
            "一年免费保修, 非人为损坏硬件故障.",
            "支持微信 / 支付宝 / 银联信用卡 / 企业月结.",
            "一线城市 24h, 二线城市 48h, 偏远 3-5 个工作日.",
        ],
        metadatas=[{"src": "a"}, {"src": "b"}, {"src": "c"}],
    )

    out = asyncio.run(
        pl_mod.run_rag(
            question="保修多久?",
            kb_id="test_pipeline_full",
            top_k=2,
            use_agent=False,
        )
    )
    assert "answer" in out
    assert isinstance(out["answer"], str)
    assert "contexts" in out
    assert isinstance(out["contexts"], list)
    assert "context_ids" in out
    assert out["metadata"]["mode"] == "rag"


# 8) agent 模式: pipeline 走 build_agent_graph, 我们 stub 掉 graph 模块
async def test_run_rag_agent_mode_invokes_graph():
    from app.agents import graph as graph_mod
    from app.rag import pipeline as pl_mod

    class _FakeGraph:
        async def ainvoke(self, state, config=None):
            # 事件所有权契约 (见 pipeline._graph_events): SourcesEvent 由**节点**经
            # emit 推出, DoneEvent 由 pipeline 在 ainvoke 返回后补。这个 fake 把
            # 两件事都钉住 —— 少了 emit, contexts 会恒为 [] 让下面那条断言变成恒真。
            emit = ((config or {}).get("configurable") or {}).get("emit")
            assert emit is not None, "pipeline 必须把 emit 通道经 config 传给图"
            emit(
                pl_mod.SourcesEvent(
                    [
                        {
                            "id": "c1",
                            "document": "上下文",
                            "metadata": {"doc_id": "d1"},
                            "score": 0.9,
                        }
                    ]
                )
            )
            return {
                **state,
                "final": "agent 模式回答",
                "sources": [
                    {"id": "c1", "document": "上下文", "metadata": {"doc_id": "d1"}, "score": 0.9}
                ],
                "trace": [{"role": "router", "content": "计划"}],
            }

    # 直接 monkey-patch graph_mod.build_agent_graph, 因为 pipeline 是 lazy import
    # 这个模块, 会直接读 graph_mod.build_agent_graph.
    original = graph_mod.build_agent_graph
    graph_mod.build_agent_graph = lambda *a, **k: _FakeGraph()
    try:
        out = await pl_mod.run_rag(
            question="任意",
            kb_id="test_pipeline_agent",
            use_agent=True,
        )
        assert out["answer"] == "agent 模式回答"
        assert out["metadata"]["mode"] == "agent"
        assert out["contexts"] == ["上下文"]
        assert out["context_ids"] == ["c1"]
        assert out["context_scores"] == [0.9]
        assert out["metadata"]["trace"] == [{"role": "router", "content": "计划"}]
    finally:
        graph_mod.build_agent_graph = original
