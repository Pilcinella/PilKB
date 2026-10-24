"""react 引擎 (reason ⇄ act 工具循环) 的契约测试.

覆盖:
1. 端到端循环: reason 发起 hybrid_search -> act 执行回填 ToolMessage -> reason
   收工 -> compose 用收集到的证据生成; sources 跨轮按 chunk id 去重、按首次
   出现顺序编号, 提示词 [n] 与 sources 顺序严格对齐 (引用溯源红线).
2. max_steps 预算闸门: 贪婪的 LLM 永远要工具, 轮数被闸在 agent_max_steps,
   预算耗尽后不再发起 reason, 直接 compose.
3. 未知工具 / 工具执行失败: 观测带错误文本进 tool_notes, 循环不终止.
4. 第一轮就收工: 无 sources 无 notes -> compose 走 direct 提示词 + NO_KB_NOTICE.
5. read_parent / read_chunk / list_documents handler 的取数口径.

patch 点约定 (与 test_agent.py 一致, 混了不会报错只会静默走真实 LLM):
  - reason 的决策 LLM: ``app.agents.graph.get_llm`` (模块级绑定);
  - compose 的生成 LLM: ``app.agents.llm_router.get_llm`` (模块属性查找);
  - 工具执行: ``tools.HANDLERS`` 的 dict 本身 (graph 侧 import 的是同一个
    dict 对象, setitem 对两边同时生效).
"""
from __future__ import annotations

from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from app.agents import tools as tools_mod
from app.agents.graph import (
    _llm_window,
    build_agent_graph,
    new_agent_state,
    reason_node,
)
from app.core.config import settings
from app.rag import pipeline as pl_mod


class _LoopLLM:
    """脚本化 react 决策: 按顺序吐出预置 AIMessage, 用尽后重复最后一条.

    同时记录每次收到的消息, 供记忆类断言检查图侧传进来的上下文.
    """

    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.calls = 0
        self.bound_tool_names = None

    def bind_tools(self, tools):
        self.bound_tool_names = [t.name for t in tools]
        return self

    def invoke(self, msgs):
        self.seen.append(list(msgs))
        self.calls += 1
        return self.script[min(self.calls - 1, len(self.script) - 1)]


def _tool_call(name, args, cid):
    return AIMessage(
        content="",
        tool_calls=[{"name": name, "args": args, "id": cid, "type": "tool_call"}],
    )


class _AnswerLLM:
    """compose 的生成 LLM: 记录提示词, 供 [n] 对齐断言."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.answer)

    async def astream(self, prompt):
        self.prompts.append(prompt)
        yield AIMessage(content=self.answer)


class _Collector:
    def __init__(self):
        self.events = []

    def __call__(self, ev):
        self.events.append(ev)

    def kinds(self):
        return [type(e).__name__ for e in self.events]


def _patch_llms(monkeypatch, loop_llm, answer_llm):
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: loop_llm)
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: answer_llm)


def _patch_handler(monkeypatch, name, fn):
    monkeypatch.setitem(tools_mod.HANDLERS, name, fn)


# ---------- 端到端循环 ----------


async def test_react_loop_merges_sources_across_rounds(monkeypatch):
    hits1 = [
        {"id": "c0", "document": "保修 24 个月", "metadata": {"doc_id": "d0"}, "rerank_score": 0.9}
    ]
    hits2 = [
        # 第二轮重复带回 c0 (模型换了词重搜), 另有一条新候选
        {"id": "c0", "document": "保修 24 个月", "metadata": {"doc_id": "d0"}, "rerank_score": 0.9},
        {"id": "c1", "document": "高级版 399 元", "metadata": {"doc_id": "d0"}, "rerank_score": 0.8},
    ]
    seen_queries: list = []

    def fake_search(args, kb_id):
        seen_queries.append((args.get("query"), kb_id))
        return "[1] 文本", (hits1 if len(seen_queries) == 1 else hits2)

    _patch_handler(monkeypatch, "hybrid_search", fake_search)
    loop_llm = _LoopLLM(
        [
            _tool_call("hybrid_search", {"query": "保修 多久"}, "call_1"),
            _tool_call("hybrid_search", {"query": "高级版 价格"}, "call_2"),
            AIMessage(content="证据足够"),
        ]
    )
    answer_llm = _AnswerLLM("保修 24 个月, 高级版 399 元 [1]")
    _patch_llms(monkeypatch, loop_llm, answer_llm)

    out = await build_agent_graph().ainvoke(
        new_agent_state("保修多久, 高级版多少钱?", kb_id="kb1", mode="agent", engine="react")
    )

    assert out["tool_rounds"] == 2
    # 模型自己组织的检索词 (不是用户原话) 传给了工具, kb_id 跟着 state 走
    assert seen_queries == [("保修 多久", "kb1"), ("高级版 价格", "kb1")]
    # 跨轮去重 + 首次出现顺序 —— [n] 编号与 sources 的对齐就押在这上面
    assert [s["id"] for s in out["sources"]] == ["c0", "c1"]
    prompt = answer_llm.prompts[0]
    assert prompt.index("[1] 保修 24 个月") < prompt.index("[2] 高级版 399 元")
    assert out["final"] == "保修 24 个月, 高级版 399 元 [1]"
    # 消息通道: 两次决策 AIMessage + 两次 ToolMessage + 最终答案
    kinds = [type(m).__name__ for m in out["messages"]]
    assert kinds.count("ToolMessage") == 2
    assert isinstance(out["messages"][-1], AIMessage)
    assert out["messages"][-1].content == out["final"]
    roles = [t["role"] for t in out["trace"]]
    assert roles == [
        "reason",
        "tool:hybrid_search",
        "reason",
        "tool:hybrid_search",
        "reason",
        "answer",
    ]


async def test_budget_gate_stops_greedy_loop(monkeypatch):
    """模型永远要工具时, 轮数必须被闸死在 agent_max_steps, 之后不再发起 reason."""
    monkeypatch.setattr(settings, "agent_max_steps", 2)
    calls = {"n": 0}

    def fake_search(args, kb_id):
        i = calls["n"]
        calls["n"] += 1
        hit = {"id": f"c{i}", "document": f"文档{i}", "metadata": {}, "rerank_score": 0.5}
        return "[1] 文本", [hit]

    _patch_handler(monkeypatch, "hybrid_search", fake_search)
    loop_llm = _LoopLLM([_tool_call("hybrid_search", {"query": "q"}, "call_x")])  # 永远重复
    _patch_llms(monkeypatch, loop_llm, _AnswerLLM("答案"))

    out = await build_agent_graph().ainvoke(
        new_agent_state("问", kb_id="kb1", mode="agent", engine="react")
    )

    assert out["tool_rounds"] == 2, "预算 = 2 轮, 第三轮必须被闸掉"
    assert loop_llm.calls == 2, "预算耗尽后不应再发起第三次 reason"
    assert [s["id"] for s in out["sources"]] == ["c0", "c1"]
    assert out["final"] == "答案", "预算耗尽后用已收集的证据兜底作答"


async def test_tool_errors_do_not_kill_loop(monkeypatch):
    """未知工具 / 执行抛异常: 观测带错误文本, 循环继续, 最终仍能作答."""

    def boom(args, kb_id):
        raise RuntimeError("网络抖动")

    _patch_handler(monkeypatch, "hybrid_search", boom)
    loop_llm = _LoopLLM(
        [
            _tool_call("no_such_tool", {}, "call_1"),
            _tool_call("hybrid_search", {"query": "q"}, "call_2"),
            _tool_call("calculator", {"expression": "1+"}, "call_3"),
            AIMessage(content="收工"),
        ]
    )
    _patch_llms(monkeypatch, loop_llm, _AnswerLLM("答"))

    out = await build_agent_graph().ainvoke(
        new_agent_state("q", kb_id="kb1", mode="agent", engine="react")
    )

    assert out["final"] == "答"
    assert out["sources"] == []
    joined = "\n".join(out["tool_notes"])
    assert "未知工具: no_such_tool" in joined
    assert "工具执行失败: 网络抖动" in joined
    assert "计算失败" in joined  # 真实 calculator handler: safe_eval 报错转文本
    # 失败观测也进了消息通道, 模型下一轮看得到错误才能调整
    assert any(
        isinstance(m, ToolMessage) and "网络抖动" in m.content for m in out["messages"]
    )


async def test_first_turn_finish_goes_direct(monkeypatch):
    """第一轮就宣布证据足够 (无工具调用) -> compose 走 direct 提示词."""
    loop_llm = _LoopLLM([AIMessage(content="这个问题不用查")])
    answer_llm = _AnswerLLM("闲聊回复")
    col = _Collector()
    _patch_llms(monkeypatch, loop_llm, answer_llm)

    out = await build_agent_graph().ainvoke(
        new_agent_state("你好", kb_id="kb1", mode="agent", engine="react"),
        {"configurable": {"emit": col}},
    )

    assert loop_llm.calls == 1
    assert "【检索结果】" not in answer_llm.prompts[0], "无 sources/notes 应走 direct 提示词"
    kinds = col.kinds()
    # reason 的 TraceEvent 会排在前面, 这里只约束 compose 的两个事件顺序:
    # SourcesEvent 必须发 (即使空列表, run_rag 靠它区分检索/生成阶段), 且先于 Notice.
    assert "SourcesEvent" in kinds and "NoticeEvent" in kinds
    assert kinds.index("SourcesEvent") < kinds.index("NoticeEvent")
    assert out["sources"] == []
    assert out["final"] == "闲聊回复"


# ---------- 工具层取数口径 ----------


def _fake_retriever(monkeypatch, docs):
    r = SimpleNamespace(bm25_store=SimpleNamespace(docs=docs))
    monkeypatch.setattr("app.agents.tools.get_retriever", lambda kb="default": r)
    return r


def test_read_parent_picks_neighbors_only(monkeypatch):
    docs = [
        {"id": "d1_c0", "text": "第一段", "metadata": {"doc_id": "d1", "chunk_index": 0}},
        {"id": "d1_c1", "text": "第二段", "metadata": {"doc_id": "d1", "chunk_index": 1}},
        {"id": "d1_c2", "text": "第三段", "metadata": {"doc_id": "d1", "chunk_index": 2}},
        {"id": "d2_c0", "text": "别的文档", "metadata": {"doc_id": "d2", "chunk_index": 0}},
    ]
    _fake_retriever(monkeypatch, docs)

    obs, hits = tools_mod.HANDLERS["read_parent"]({"chunk_id": "d1_c1"}, "kb1")

    # 只连坐同文档紧邻的前后各一个, 别的文档绝不带进来
    assert [h["id"] for h in hits] == ["d1_c0", "d1_c1", "d1_c2"]
    assert "第一段" in obs and "第三段" in obs
    assert "别的文档" not in obs


def test_read_chunk_and_missing(monkeypatch):
    docs = [{"id": "d1_c0", "text": "唯一 chunk", "metadata": {"doc_id": "d1"}}]
    _fake_retriever(monkeypatch, docs)

    obs, hits = tools_mod.HANDLERS["read_chunk"]({"chunk_id": "d1_c0"}, "kb1")
    assert obs == "唯一 chunk"
    assert hits[0]["id"] == "d1_c0"

    obs, hits = tools_mod.HANDLERS["read_chunk"]({"chunk_id": "nope"}, "kb1")
    assert "chunk 不存在" in obs
    assert hits == []


def test_list_documents_aggregates(monkeypatch):
    docs = [
        {"id": "d1_c0", "text": "a", "metadata": {"doc_id": "d1", "filename": "faq.md", "chunk_index": 0}},
        {"id": "d1_c1", "text": "b", "metadata": {"doc_id": "d1", "filename": "faq.md", "chunk_index": 1}},
        {"id": "d2_c0", "text": "c", "metadata": {"doc_id": "d2", "filename": "price.md", "chunk_index": 0}},
    ]
    _fake_retriever(monkeypatch, docs)

    obs, hits = tools_mod.HANDLERS["list_documents"]({}, "kb1")
    assert "- d1: faq.md (2 chunks)" in obs
    assert "- d2: price.md (1 chunks)" in obs
    assert hits == []


# ---------- LLM 消息窗口的合法性 (记忆模式下的 400 回归) ----------


def _tc(cid, name="hybrid_search"):
    return AIMessage(
        content="", tool_calls=[{"name": name, "args": {}, "id": cid, "type": "tool_call"}]
    )


def _tr(cid, text="观测"):
    return ToolMessage(content=text, tool_call_id=cid)


def _assert_no_orphans(msgs):
    """provider 的两条硬约束: tool 响应必须有前置 tool_calls; tool_calls 必须有齐响应."""
    pending: list = []
    for m in msgs:
        calls = getattr(m, "tool_calls", None) if isinstance(m, AIMessage) else None
        if calls:
            assert not pending, "上一个 tool_calls 还没回应就来了新的"
            pending = [str(c.get("id")) for c in calls]
        elif isinstance(m, ToolMessage):
            assert pending and str(m.tool_call_id) in pending, (
                f"孤立 ToolMessage (会触发 provider 400): {m.tool_call_id}"
            )
            pending.remove(str(m.tool_call_id))
        else:
            assert not pending, f"tool_calls 未回应就遇到 {type(m).__name__}"
    assert not pending, "tool_calls 没有对应响应就结束了"


def test_llm_window_drops_leading_orphan_tool():
    """窗口切在工具交换中间 (起点是 ToolMessage) 时必须裁掉, 而不是原样发出."""
    msgs = [
        _tc("c0"),
        _tr("c0"),
        _tc("c1"),  # <- keep=3 会从这条 ToolMessage 之后切, 制造孤立响应
        _tr("c1"),
    ]
    out = _llm_window(msgs, keep=3)
    _assert_no_orphans(out)


def test_llm_window_drops_incomplete_exchange():
    """只有部分响应的工具交换 (回合被取消留下的半截) 整个丢弃."""
    partial = AIMessage(
        content="",
        tool_calls=[
            {"name": "hybrid_search", "args": {}, "id": "c0", "type": "tool_call"},
            {"name": "calculator", "args": {}, "id": "c1", "type": "tool_call"},
        ],
    )
    msgs = [
        HumanMessage(content="问题"),
        partial,  # 声明了两个 id
        _tr("c0"),  # 只回了一个 -> 整段非法
        AIMessage(content="下一轮回答"),
    ]
    out = _llm_window(msgs, keep=10)
    _assert_no_orphans(out)
    assert not any(isinstance(m, ToolMessage) for m in out), "半截交换的响应必须一起丢"
    assert not any(getattr(m, "tool_calls", None) for m in out), "半截交换的请求也必须丢"


def test_llm_window_keeps_valid_pairs_intact():
    msgs = [
        HumanMessage(content="q1"),
        _tc("c0"),
        _tr("c0"),
        AIMessage(content="a1"),
        HumanMessage(content="q2"),
    ]
    out = _llm_window(msgs, keep=10)
    assert out == msgs, "合法的窗口不应被改动"
    _assert_no_orphans(out)


async def test_reason_node_sends_legal_window_under_memory(monkeypatch):
    """长会话 (超过 keep) 下, reason 发出去的消息必须仍然合法.

    线上 2026-10-16 的 400: 连问多轮后消息超过 20 条, 裸切片把窗口开在 ToolMessage
    上 -> "Messages with role 'tool' must be a response to a preceding message with
    'tool_calls'"。
    """
    monkeypatch.setattr(settings, "agent_memory_max_messages", 4)
    llm = _LoopLLM([AIMessage(content="收工")])
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: llm)

    state = new_agent_state("本轮问题", kb_id="kb1", mode="agent", engine="react")
    # 历史 (会被切): 一个完整交换 + 一个残缺交换 + 本轮问题
    state["messages"] = [
        HumanMessage(content="旧问题"),
        _tc("c0"),
        _tr("c0"),
        AIMessage(content="旧回答"),
        HumanMessage(content="有问题的一轮"),
        _tc("c2"),
        _tr("c2"),
        AIMessage(content="半截回答"),
        HumanMessage(content="本轮问题"),
    ]
    await reason_node(state, None)

    sent = llm.seen[0]
    assert sent[0].content.startswith("你是企业知识库问答 Agent"), "系统提示必须仍在最前"
    _assert_no_orphans(sent[1:])
    assert any("本轮问题" in str(getattr(m, "content", "")) for m in sent), "本轮问题不能丢"


def test_merge_sources_dedup_and_order():
    a = [{"id": "c0", "document": "x", "metadata": {}, "score": 0.9}]
    b = [
        {"id": "c1", "document": "y", "metadata": {}, "rerank_score": 0.8},
        {"id": "c0", "document": "x", "metadata": {}, "rerank_score": 0.99},
    ]
    out = tools_mod.merge_sources(a, b)
    # 去重保序; rerank_score 归一到 score 字段
    assert [c["id"] for c in out] == ["c0", "c1"]
    assert out[0]["score"] == 0.9, "已存在的候选分数不被后续覆盖"
    assert out[1]["score"] == 0.8
