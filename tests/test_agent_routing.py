"""LangGraph 路由与节点契约.

覆盖:
1. ``_parse_planner_decision`` 容错 (纯 JSON / 代码块 / 字符串布尔 / fallback).
   fallback 方向是 **fail-safe 走检索** —— 它现在是一道安全兜底的守卫, 不只是解析测试.
2. ``entry_router``: rag 模式跳过路由节点 (零 LLM 调用), agent 模式才进路由.
3. ``route_after_route``: 三条分支 + 缺决策时 fail-safe + 纯函数 (不改 state).
4. ``rag_node`` 把 state["kb_id"] 同时传给 build_tools 与检索链.
5. ``rag_node`` 用**用户原问题**检索, 且工具结果追加在检索上下文之后.
6. 端到端 ``ainvoke``: 两种 mode 各跑一次, 断言走的是正确的入口.

patch 点有两处**不能混** (混错了不报错, 只会静默走到真实 LLM):
  - 图节点用 ``app.agents.graph.get_llm`` (模块级绑定);
  - pipeline 的生成用 ``app.agents.llm_router.get_llm`` (模块属性查找)。
"""
from __future__ import annotations

import json

from langchain_core.messages import AIMessage

from app.agents.graph import (
    _parse_planner_decision,
    build_agent_graph,
    entry_router,
    new_agent_state,
    rag_node,
    route_after_route,
    run_tools,
)
from app.rag import pipeline as pl_mod

# ---------- _parse_planner_decision 解析 ----------


class TestParsePlannerDecision:
    def test_pure_json(self):
        raw = '{"needs_calculator": true, "needs_retrieval": false, "needs_date": false}'
        d = _parse_planner_decision(raw)
        assert d["needs_calculator"] is True
        assert d["needs_retrieval"] is False

    def test_code_block(self):
        raw = '```json\n{"needs_calculator": true, "needs_retrieval": true, "needs_date": false}\n```'
        d = _parse_planner_decision(raw)
        assert d["needs_calculator"] is True
        assert d["needs_retrieval"] is True

    def test_string_bool(self):
        d = _parse_planner_decision('{"needs_calculator": "yes", "needs_retrieval": "0"}')
        assert d["needs_calculator"] is True
        assert d["needs_retrieval"] is False

    def test_unparseable_fallback(self):
        d = _parse_planner_decision("LLM 输出乱码")
        # fallback 必须 fail-safe 走检索 (慢但安全), 而不是凭模型知识直答 (快但可能编造)。
        assert d["needs_retrieval"] is True
        assert d["needs_calculator"] is False


# ---------- 入口与路由条件边 ----------


def test_entry_router_splits_by_mode():
    # agent 模式默认 react 引擎; router 引擎显式声明才走旧的一次性路由.
    assert entry_router({"mode": "agent"}) == "reason"
    assert entry_router({"mode": "agent", "engine": "react"}) == "reason"
    assert entry_router({"mode": "agent", "engine": "router"}) == "route"
    assert entry_router({"mode": "rag"}) == "rag"
    # 缺 mode (手工构造 state / 老调用点) 时按 rag 处理: 不做多余的 LLM 调用.
    assert entry_router({}) == "rag"


def test_route_after_route_prefers_retrieval():
    assert route_after_route({"plan_decision": {"needs_retrieval": True}}) == "rag"
    # 同时要检索和算数 -> 仍然走检索节点 (工具结果会并进它的提示词).
    assert (
        route_after_route(
            {"plan_decision": {"needs_retrieval": True, "needs_calculator": True}}
        )
        == "rag"
    )


def test_route_after_route_falls_to_tools_then_direct():
    assert (
        route_after_route({"plan_decision": {"needs_retrieval": False, "needs_calculator": True}})
        == "tools"
    )
    assert (
        route_after_route({"plan_decision": {"needs_retrieval": False, "needs_date": True}})
        == "tools"
    )
    assert route_after_route({"plan_decision": {"needs_retrieval": False}}) == "direct"


def test_route_after_route_is_pure_and_fails_safe():
    """决策缺失时必须走检索, 且条件边**不得**修改 state.

    旧 route_review 的教训: branch 函数的副作用不会写回 checkpoint, 在 edge 里改
    state 是无效的 (当年 retry_count 的递增因此被静默丢弃)。
    """
    state = {"plan_decision": {}}
    snapshot = json.dumps(state, sort_keys=True)
    assert route_after_route(state) == "rag"
    assert json.dumps(state, sort_keys=True) == snapshot, "条件边不得修改 state"

    assert route_after_route({}) == "rag", "没有决策时不能凭模型知识直答"


# ---------- rag_node: kb_id / 提问口径 / 上下文顺序 ----------


class _AnswerLLM:
    """pipeline 侧生成用的假 LLM (带 astream, 供流式分支用)."""

    def __init__(self, answer):
        self.answer = answer
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return AIMessage(content=self.answer)

    async def astream(self, prompt):
        self.prompts.append(prompt)
        yield AIMessage(content=self.answer)


def _stub_retrieval(monkeypatch, hits, captured=None):
    async def _stub(question, kb_id="default", retriever=None, trace=None, history=None):
        if captured is not None:
            captured["question"] = question
            captured["kb_id"] = kb_id
            captured["history"] = history
        return hits

    monkeypatch.setattr(pl_mod, "retrieve_and_rerank", _stub)


def _cfg(**kw):
    return {"configurable": kw}


async def test_rag_node_propagates_kb_id(monkeypatch):
    """state['kb_id'] 必须同时进 build_tools 与检索链.

    曾经有个 bug: 检索节点默认走 "default", 架空了私有 kb 的 ACL。
    """
    captured = {"tools_kb": None}

    def _fake_build_tools(kb_id="default"):
        captured["tools_kb"] = kb_id
        return []

    monkeypatch.setattr("app.agents.graph.build_tools", _fake_build_tools)
    _stub_retrieval(monkeypatch, [], captured)
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("ok"))

    state = new_agent_state("q", kb_id="alice_private_kb", mode="agent")
    state["plan_decision"] = {"needs_retrieval": True, "needs_calculator": True}
    await rag_node(state, _cfg(top_k=3))

    assert captured["tools_kb"] == "alice_private_kb"
    assert captured["kb_id"] == "alice_private_kb"


async def test_rag_node_retrieves_with_original_question(monkeypatch):
    """检索 query 用**用户原问题**, 不用路由节点产出的关键词 —— 扩展交给改写器."""
    captured = {}
    _stub_retrieval(
        monkeypatch,
        [{"id": "c0", "document": "ctx", "metadata": {}, "rerank_score": 1.0}],
        captured,
    )
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("ok"))

    state = new_agent_state("123*456 是多少?", kb_id="kb1")
    state["plan_decision"] = {"needs_retrieval": True, "needs_calculator": True}
    out = await rag_node(state, _cfg(top_k=3))

    assert captured["question"] == "123*456 是多少?"
    assert out["sources"][0]["id"] == "c0"
    assert out["sources"][0]["score"] == 1.0


async def test_rag_node_appends_tool_context_after_retrieved(monkeypatch):
    """工具结果必须追加在**检索上下文之后**.

    提示词里的 [1..N] 要与 state["sources"] 一一对应; 工具文本插到前面会让
    前端显示的引用与正文引用的编号整体错位。
    """
    llm = _AnswerLLM("ok")
    _stub_retrieval(
        monkeypatch,
        [{"id": "c0", "document": "保修期 24 个月", "metadata": {}, "rerank_score": 1.0}],
    )
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: llm)
    monkeypatch.setattr(
        "app.agents.graph.build_tools",
        lambda kb_id="default": [_FakeTool("calculator", "56088")],
    )

    state = new_agent_state("123*456 是多少?", kb_id="kb1")
    state["plan_decision"] = {"needs_retrieval": True, "needs_calculator": True}
    await rag_node(state, _cfg(top_k=3))

    prompt = llm.prompts[0]
    assert "[1] 保修期 24 个月" in prompt
    assert prompt.index("[1] 保修期 24 个月") < prompt.index("calculator: 56088"), (
        "工具结果必须排在检索上下文之后"
    )


class _FakeTool:
    def __init__(self, name, ret):
        self.name = name
        self._ret = ret

    def invoke(self, _arg):
        return self._ret


# ---------- run_tools 的确定性执行 ----------


def test_run_tools_only_calls_what_was_decided(monkeypatch):
    calls = {"calc": 0, "date": 0}

    class _T:
        def __init__(self, name, marker):
            self.name = name
            self._marker = marker

        def invoke(self, _arg):
            calls[self._marker] += 1
            return "3"

    monkeypatch.setattr(
        "app.agents.graph.build_tools",
        lambda kb_id="default": [_T("calculator", "calc"), _T("current_date", "date")],
    )
    state = new_agent_state("1+2", kb_id="kb1")

    parts = run_tools({"needs_calculator": True}, "1+2", state, lambda _e: None)
    assert calls == {"calc": 1, "date": 0}
    assert parts and parts[0].startswith("calculator: ")
    # needs_retrieval=False 且无工具需求 -> 一个工具都不调, 也不写 trace.
    state2 = new_agent_state("你好", kb_id="kb1")
    assert run_tools({"needs_retrieval": False}, "你好", state2, lambda _e: None) == []
    assert calls == {"calc": 1, "date": 0}
    assert state2["trace"] == []


# ---------- 端到端: 两种 mode 走对入口 ----------


class _RouterLLM:
    """按 system prompt 分派: 只有 route_node 会看到 "任务规划 Agent"."""

    def __init__(self, decision_json):
        self._decision = decision_json
        self.route_calls = 0

    def invoke(self, msgs):
        head = msgs[0].content if msgs else ""
        if "任务规划 Agent" in head:
            self.route_calls += 1
            return AIMessage(content=self._decision)
        return AIMessage(content="(不该走到这里)")


def _decision(needs_retrieval=True, needs_calculator=False, needs_date=False):
    return json.dumps(
        {
            "needs_calculator": needs_calculator,
            "needs_retrieval": needs_retrieval,
            "needs_date": needs_date,
        }
    )


async def test_graph_agent_mode_routes_then_retrieves(monkeypatch):
    router = _RouterLLM(_decision(needs_retrieval=True))
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: router)
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("答案 [1]"))
    _stub_retrieval(
        monkeypatch,
        [{"id": "c0", "document": "ctx", "metadata": {}, "rerank_score": 0.5}],
    )

    out = await build_agent_graph().ainvoke(
        new_agent_state("q", kb_id="my_kb", mode="agent", engine="router")
    )

    assert router.route_calls == 1
    roles = [t["role"] for t in out["trace"]]
    assert "router" in roles and "retriever" in roles and "answer" in roles
    assert out["final"] == "答案 [1]"
    assert out["sources"][0]["id"] == "c0"


async def test_graph_rag_mode_skips_router(monkeypatch):
    """rag 模式必须**零**路由 LLM 调用 —— 用户已经选了"常规检索问答", 再花一次
    往返判断"要不要检索"是纯粹的延迟浪费."""
    router = _RouterLLM(_decision(needs_retrieval=True))
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: router)
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("答案"))
    _stub_retrieval(monkeypatch, [])

    out = await build_agent_graph().ainvoke(new_agent_state("q", kb_id="kb1", mode="rag"))

    assert router.route_calls == 0, "rag 模式跳过了路由节点, 不该有路由 LLM 调用"
    assert "router" not in [t["role"] for t in out["trace"]]
    assert out["final"] == "答案"


async def test_graph_agent_mode_direct_when_no_retrieval(monkeypatch):
    """路由判不需要检索 -> 走直答, 不发检索请求."""
    router = _RouterLLM(_decision(needs_retrieval=False))
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: router)
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: _AnswerLLM("我不确定"))
    captured = {}
    _stub_retrieval(monkeypatch, [], captured)

    out = await build_agent_graph().ainvoke(
        new_agent_state("我们公司报销标准?", kb_id="kb1", mode="agent", engine="router")
    )

    assert "question" not in captured, "直答分支不该发起检索"
    assert out["final"] == "我不确定"
    assert out["sources"] == []
    assert any(t["role"] == "direct" for t in out["trace"])


async def test_graph_router_engine_does_not_double_messages(monkeypatch):
    """router 引擎 + 多轮历史: route 节点的返回值不得把 messages 带回通道.

    messages 是 add reducer, route 曾是唯一 ``return state`` 的节点 —— 收到的
    整段 messages 被原样追加一遍, 出图历史翻倍, 且按 _recent_history 的口径
    (最后一条 HumanMessage 之前都算历史) 当前问题也混进了【对话历史】。
    """
    router = _RouterLLM(_decision(needs_retrieval=False))
    monkeypatch.setattr("app.agents.graph.get_llm", lambda *a, **k: router)
    gen = _AnswerLLM("好的")
    monkeypatch.setattr("app.agents.llm_router.get_llm", lambda *a, **k: gen)

    history = [
        {"role": "user", "content": "H1"},
        {"role": "assistant", "content": "A1"},
    ]
    out = await build_agent_graph().ainvoke(
        new_agent_state("CURR", kb_id="kb1", mode="agent", engine="router", history=history)
    )

    # 入图 3 条 (H1/A1/CURR) + 终端追加的答案 = 恰好 4 条, 一条不多。
    assert [m.content for m in out["messages"]] == ["H1", "A1", "CURR", "好的"]
    # 生成提示词里历史只出现一次, 当前问题只在【用户问题】出现一次。
    prompt = gen.prompts[-1]
    assert prompt.count("H1") == 1
    assert prompt.count("CURR") == 1
