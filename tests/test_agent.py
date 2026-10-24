"""测试 6: LangGraph 节点契约 (用 monkeypatch 把 LLM 换成 mock).

改造后图里的节点: route / reason / act / compose / rag / tools / direct。本文件测
**节点自身**的契约 (state 写入、事件发射顺序、history 转换); 图结构与路由分派由
test_agent_routing.py 覆盖, react 循环由 test_agent_loop.py 覆盖, 端到端帧序由
test_chat_sse_contract.py 覆盖。

注意两套 patch 点不能混:
  - 图里的节点用 `app.agents.graph.get_llm` (模块级绑定);
  - pipeline 的生成用 `app.agents.llm_router.get_llm` (模块属性查找)。
patch 错了不会报错, 只会静默走到真实 LLM —— 所以下面每个用例都显式指定目标。
"""
from __future__ import annotations

from langchain_core.messages import AIMessage

from app.agents.graph import direct_node, new_agent_state, rag_node, route_node, tools_node
from app.rag import pipeline as pl_mod


class FakeLLM:
    """只有 invoke: 对应 stream_tokens=False 的路径."""

    def __init__(self, content):
        self._content = content

    def invoke(self, _msgs):
        return AIMessage(content=self._content)


class StreamLLM:
    """invoke + astream 都有, 且按固定文本产出一个 chunk."""

    def __init__(self, content):
        self._content = content

    def invoke(self, _msgs):
        return AIMessage(content=self._content)

    async def astream(self, _msgs):
        yield AIMessage(content=self._content)


class _Collector:
    """替代 _graph_events 的队列: 按发射顺序收事件."""

    def __init__(self):
        self.events = []

    def __call__(self, ev):
        self.events.append(ev)

    def kinds(self):
        return [type(e).__name__ for e in self.events]

    def index(self, kind):
        for i, e in enumerate(self.events):
            if type(e).__name__ == kind:
                return i
        return -1


class _FakeTool:
    def __init__(self, name, ret):
        self.name = name
        self._ret = ret

    def invoke(self, _arg):
        return self._ret


def _cfg(**kw):
    """构造 RunnableConfig 的 configurable 段 (节点只读这里)."""
    return {"configurable": kw}


def _patch_retrieval(monkeypatch, hits=None):
    """把检索链 stub 掉, 只留一个候选."""
    hits = hits if hits is not None else [
        {"id": "c0", "document": "保修期为 24 个月。", "metadata": {"doc_id": "d0"}, "rerank_score": 0.9}
    ]

    async def _stub(question, kb_id="default", retriever=None, trace=None, history=None):
        return hits

    monkeypatch.setattr(pl_mod, "retrieve_and_rerank", _stub)


def _patch_tools(monkeypatch):
    monkeypatch.setattr(
        "app.agents.graph.build_tools",
        lambda kb_id="default": [
            _FakeTool("calculator", "3"),
            _FakeTool("current_date", "2026-09-19 10:00:00"),
        ],
    )


# ---------------------------------------------------------------------------
# route_node
# ---------------------------------------------------------------------------


async def test_route_node_writes_decision_to_state(monkeypatch):
    monkeypatch.setattr(
        "app.agents.graph.get_llm",
        lambda *a, **k: FakeLLM(
            '{"needs_calculator": false, "needs_retrieval": true, "needs_date": false}'
        ),
    )
    state = new_agent_state("PilKB 怎么部署?", kb_id="default", mode="agent")
    out = await route_node(state, None)

    assert out["plan_decision"] == {
        "needs_calculator": False,
        "needs_retrieval": True,
        "needs_date": False,
    }
    assert any(t["role"] == "router" for t in out["trace"])
    # question 不被改写 —— 改写是未来接记忆时才有的事
    assert out["question"] == "PilKB 怎么部署?"


async def test_route_node_falls_back_to_retrieval_on_garbage(monkeypatch):
    """解析不出来时必须 fail-safe 走检索, 而不是凭模型知识直答."""
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: FakeLLM("我不知道该输出什么"))
    out = await route_node(new_agent_state("q", mode="agent"), None)
    assert out["plan_decision"]["needs_retrieval"] is True


# ---------------------------------------------------------------------------
# 终端节点: 事件顺序与 final
# ---------------------------------------------------------------------------


async def test_rag_node_emits_sources_before_tokens(monkeypatch):
    _patch_retrieval(monkeypatch)
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: StreamLLM("保修期 24 个月。[1]"))
    col = _Collector()
    state = new_agent_state("保修期多久?", kb_id="kb1")

    out = await rag_node(state, _cfg(emit=col, stream_tokens=True, top_k=3))

    assert col.kinds()[0] == "SourcesEvent", "SourcesEvent 必须是第一个事件"
    assert col.index("SourcesEvent") < col.index("TokenEvent")
    assert out["final"] == "保修期 24 个月。[1]"
    assert out["sources"][0]["id"] == "c0"


async def test_rag_node_emits_empty_sources_when_nothing_retrieved(monkeypatch):
    """检索为空也要发 SourcesEvent([]): run_rag 靠"有没有收到过它"判断检索阶段
    是否走完, 缺了它, 生成失败会被误判成检索失败而整条抛出."""
    _patch_retrieval(monkeypatch, hits=[])
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: FakeLLM("未找到相关信息"))
    col = _Collector()

    out = await rag_node(new_agent_state("q", kb_id="kb1"), _cfg(emit=col, top_k=3))

    assert isinstance(col.events[0], pl_mod.SourcesEvent)
    assert col.events[0].sources == []
    assert out["final"] == "未找到相关信息"


async def test_tools_node_emits_notice_and_runs_calculator(monkeypatch):
    _patch_tools(monkeypatch)
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: FakeLLM("1+2=3"))
    col = _Collector()
    state = new_agent_state("1+2 等于几?", kb_id="kb1", mode="agent")
    state["plan_decision"] = {"needs_retrieval": False, "needs_calculator": True, "needs_date": False}

    out = await tools_node(state, _cfg(emit=col, top_k=3))

    assert col.kinds()[0] == "SourcesEvent"
    assert col.events[0].sources == []
    # 没查知识库 -> 必须给用户一个可见标记
    assert any(isinstance(e, pl_mod.NoticeEvent) for e in col.events)
    assert any(t["role"] == "tool:calculator" for t in out["trace"])
    assert out["final"] == "1+2=3"


async def test_direct_node_marks_answer_as_unretrieved(monkeypatch):
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: FakeLLM("我不确定"))
    col = _Collector()
    state = new_agent_state("我们公司报销标准是多少?", kb_id="kb1", mode="agent")
    state["plan_decision"] = {"needs_retrieval": False, "needs_calculator": False, "needs_date": False}

    out = await direct_node(state, _cfg(emit=col, top_k=3))

    assert col.kinds()[0] == "SourcesEvent"
    notices = [e for e in col.events if isinstance(e, pl_mod.NoticeEvent)]
    assert notices and notices[0].text == pl_mod.NO_KB_NOTICE
    # 标记走独立事件, 绝不能混进答案正文 (否则会进 RAGAS 打分与历史落库)
    assert pl_mod.NO_KB_NOTICE not in out["final"]


async def test_nodes_work_without_config(monkeypatch):
    """单测/脚本直接调节点时不带 config —— 每个节点都必须有 noop 兜底."""
    _patch_retrieval(monkeypatch)
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: FakeLLM("答案"))

    out = await rag_node(new_agent_state("q", kb_id="kb1"))
    assert out["final"] == "答案"

    out = await direct_node(new_agent_state("q", mode="agent"))
    assert out["final"] == "答案"


# ---------------------------------------------------------------------------
# history 预留接口
# ---------------------------------------------------------------------------


def test_new_agent_state_converts_history_to_messages():
    history = [
        {"role": "user", "content": "保修期多久?"},
        {"role": "assistant", "content": "24 个月"},
    ]
    state = new_agent_state("那它从什么时候算起?", kb_id="kb1", mode="agent", history=history)

    assert [type(m).__name__ for m in state["messages"]] == [
        "HumanMessage",
        "AIMessage",
        "HumanMessage",
    ]
    assert state["messages"][-1].content == "那它从什么时候算起?"
    # 原文另存一份给未来的改写器 (它要的是 dict, 不是 BaseMessage)
    assert state["history"] == history


def test_new_agent_state_history_none_equals_empty():
    """预留接口的硬约束: None 与 [] 必须完全等价, 否则"零行为变化"就是假的."""
    a = new_agent_state("q", kb_id="kb1", history=None)
    b = new_agent_state("q", kb_id="kb1", history=[])
    assert a["messages"] == b["messages"]
    assert a["history"] == b["history"] == []


def test_new_agent_state_skips_malformed_history():
    state = new_agent_state(
        "q",
        history=[
            {"role": "user", "content": ""},          # 空内容丢掉
            "not-a-dict",                              # 类型不对丢掉
            {"role": "user"},                          # 缺 content 丢掉
            {"role": "assistant", "content": "有效"},
        ],
    )
    assert len(state["messages"]) == 2  # 有效历史 + 本轮问题
    assert state["messages"][0].content == "有效"
