# -*- coding: utf-8 -*-
"""LangGraph 编排: 双 agent 引擎 + 共享终端, rag 模式直连检索.

一张图, 三个入口 (mode + ``settings.agent_engine`` 决定)::

    mode="rag"     entry ─────────────────────────────→ rag      → END
    engine=router  entry → route ─(条件边)──→ rag | tools | direct → END
    engine=react   entry → reason ⇄ act (工具循环) ──→ compose   → END

**react 引擎 (默认, P0)**: reason 节点 ``bind_tools`` 让模型自主决定调哪些工具、
传什么参数; act 节点执行并把 ToolMessage 回填, 循环直到模型不再要工具或
``agent_max_steps`` 预算耗尽; compose 用收集到的证据生成最终答案。这是 agentic
RAG 的本体: 检索 query 由模型现场组织 (结果不理想时换词重试 = 检索自纠, P1),
read_parent 补全被切断的上下文 —— 而不是 router 引擎那样一次性判定三个布尔。

**router 引擎 (保留)**: 一次 LLM 输出 {needs_retrieval, needs_calculator,
needs_date} 再确定性执行, 快路径 + 消融对照 (tests/run_agent_ablation.py)。

**为什么所有终端都回到 pipeline**: 提示词模板、逐字流式、ThinkBlockFilter、
引用 [n] 与 sources 的对齐, rag / tools / direct / compose 四条链路共用一份
实现 —— 同一件事写两份在这个项目里出过两次事故 (提示词措辞分叉、上下文截断
300 vs 600)。react 循环因此只负责**收集证据**, 最后一步统一走
``build_rag_prompt`` + ``_generate_events``。

**sources 合并口径 (引用溯源红线)**: 多轮检索的候选按 chunk id 去重、按**首次
出现顺序**编号 (``tools.merge_sources``), compose 提示词里的 [n] 与
``state["sources"]`` 严格一一对应; calculator / current_date 这类无候选的工具
结果进 ``tool_notes``, 拼在 sources 之后 (沿用 rag_node 追加 tool_ctx 的旧约定)。

**多轮记忆 (P2)**: ``agent_memory_enabled`` 且请求带 session_id 时图带 SQLite
checkpointer 编译 (checkpoint 落盘到 ``agent_checkpoint_db``, 重启不丢),
thread_id = "{user}:{session}" 隔离跨用户/跨会话状态。``messages`` 通道是
add reducer, 节点只返回**新增**消息 (见 ``_upd``); 终端节点会把最终答案作为
AIMessage 追加, 下一轮对话因此带上下文。无记忆 (关闭配置或无 session) 时与
单轮行为逐字节一致。

**节点是 async 的**, 图用 ``ainvoke`` 驱动。理由是**可取消性**: 客户端断开时
CancelledError 会穿透到节点内部的 await, 被 ``async for`` 逐层包住的生成器当场执行
finally 关闭; 同步节点只能靠轮询事件, 最坏要等一整次 LLM 调用跑完。

``kb_id`` 通过 ``AgentState`` 显式传入, 检索节点用 ``state["kb_id"]`` 调检索链,
配合 kb_registry 的 ACL —— 曾经有个 bug 是它默认走 "default", 架空了私有 kb 的权限。

注: 仍未接 ``interrupt_before`` 的 human-in-the-loop。
"""
from __future__ import annotations

import asyncio
import json
import operator
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Callable, Dict, List, Optional, TypedDict

import aiosqlite

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, StateGraph

from app.agents.llm_router import get_llm
from app.agents.tools import HANDLERS, build_tools, merge_sources
from app.core.config import settings
from app.core.text_utils import strip_think


class AgentState(TypedDict):
    """一次图运行的全部状态.

    ``messages`` 是 **add reducer** 通道: 节点返回值里的 messages 列表会被
    **追加**到通道而不是整体覆盖 —— 这是多轮记忆 (checkpointer) 的基础, 但也
    意味着节点绝不能把收到的 messages 原样带回 (否则历史每过一跳翻一倍),
    统一用 ``_upd`` 构造返回值, 需要追加消息时显式给 ``messages=[新消息]``。
    """

    messages: Annotated[List[BaseMessage], operator.add]
    kb_id: Annotated[str, "目标知识库 ID; 配合 kb_registry 的 ACL"]
    mode: Annotated[str, '来源模式: "rag" | "agent"; 入口条件边读它']
    engine: Annotated[str, 'agent 引擎: "react" | "router"; 仅 agent 模式生效']
    question: Annotated[str, "本次检索/生成用的问句 (路由未来可改写它)"]
    history: Annotated[List[dict], "多轮历史原文; 记忆开启时为空 (历史在 checkpoint)"]
    plan: Annotated[str, "路由节点的原始输出, 留作审计"]
    plan_decision: Annotated[dict, "解析后的结构化路由决策; router 引擎的条件边读它"]
    tool_rounds: Annotated[int, "react 引擎已消耗的工具轮数; 预算闸门 (after_act) 读它"]
    tool_notes: Annotated[List[str], "无候选工具 (calculator/current_date 等) 的观测文本"]
    final: Annotated[str, "最终答案; 由四个终端节点各自写入"]
    # 结构化候选 (id / document / metadata / score), 供上层构造引用来源.
    # 只靠 trace 里的文本快照不行: 那里是截断后的展示串, 缺 chunk_id 与 metadata,
    # 前端拿不到就无法做引用溯源.
    sources: Annotated[List[dict], "检索到的候选; react 引擎跨轮合并, 见 tools.merge_sources"]
    trace: Annotated[List[dict], "推理轨迹; agent 模式下逐条推给客户端"]


ROUTER_SYS = (
    "你是一个任务规划 Agent。给定用户问题，必须严格按以下 JSON 格式输出, 不要输出任何额外文字或 markdown 代码块:\n"
    '{"needs_calculator": true|false, "needs_retrieval": true|false, "needs_date": true|false}\n'
    "判定规则:\n"
    "- 涉及算术/公式/数学函数 -> needs_calculator=true.\n"
    "- 涉及当前日期/时间 -> needs_date=true.\n"
    "- 需要查文档才能回答 -> needs_retrieval=true.\n"
    "- 不要输出检索关键词: 检索 query 由下游的改写器 (QueryRewriter) 用用户原问题扩展,\n"
    "  这里只判断「要不要检索」, 不决定「怎么检索」.\n"
)


# react 引擎的决策系统提示词. 檢索自纠 (P1) 就写在这里: 第 2 条强制模型在结果
# 不相关时换词重试, 而不是基于不相关结果硬答 —— 这是与静态 pipeline 的分界线。
AGENT_SYS = (
    "你是企业知识库问答 Agent, 可以多次调用工具收集证据后再作答。规则:\n"
    "1. 涉及知识库内容的问题, 必须先用 hybrid_search 检索; 检索词自己组织, "
    "不要照抄用户原话 (口语、指代、缩写要展开成文档里可能出现的说法)。\n"
    "2. 检索结果与问题不相关或没覆盖到时, **换关键词/换表述重试**, 而不是基于不相关"
    "的结果硬答; 仍无结果可先用 list_documents 了解库里有什么, 或明确说明未找到。\n"
    "3. 需要某条结果的完整上下文 (正文被切断、要看前后文) 时用 read_parent。\n"
    "4. 算术计算用 calculator, 当前日期时间用 current_date。\n"
    "5. 证据足够后**不再调用工具**, 直接给出简短的最终结论。\n"
    "6. 最终答案中的事实只能来自工具结果, 并按结果中的 [n] 编号标注引用。"
)


# ---------------------------------------------------------------------------
# 节点执行参数 (来自 RunnableConfig, 不是图状态)
# ---------------------------------------------------------------------------


def _noop_emit(_ev: Any) -> None:
    """没有 emit 通道时的占位.

    必须有: 大量单测直接调用节点函数、不带 config, 而且 langgraph 之外的调用方
    (脚本 / 评估) 也不一定注入通道。漏了这个兜底, 那些调用会直接 AttributeError。
    """


@dataclass
class _NodeCfg:
    """一次调用的执行参数.

    走 ``config["configurable"]`` 而不是塞进 ``AgentState``: 这些值的生命周期止于
    本次调用 (谁在听事件、要不要逐 token、请求要几条引用), 混进 state 会让"图状态"
    和"调用参数"搅在一起。另外 config 传递**不会**影响图的结构, ``build_agent_graph``
    的编译缓存因此继续成立。
    """

    emit: Callable[[Any], None] = _noop_emit
    stream_tokens: bool = False
    trace: Any = None
    top_k: int = 0
    history: Optional[List[dict]] = None


def _cfg(config: Any) -> _NodeCfg:
    c = (config or {}).get("configurable") or {}
    return _NodeCfg(
        emit=c.get("emit") or _noop_emit,
        stream_tokens=bool(c.get("stream_tokens")),
        trace=c.get("trace"),
        top_k=int(c.get("top_k") or settings.top_k_final),
        history=c.get("history"),
    )


def _pipeline():
    """就地取 pipeline 模块 (两种取法都不行, 只能函数内 import):

    (a) 顶层 ``from app.rag.pipeline import ...`` 会成环 —— ``app/agents/__init__.py``
        import 本模块, 而 pipeline.py 顶部又 ``from app.agents import llm_router``;
    (b) 顶层 import 把对象绑死在本模块命名空间里, 测试的
        ``monkeypatch.setattr("app.rag.pipeline.retrieve_and_rerank", ...)`` 打不进去。
    """
    from app.rag import pipeline as pl_mod

    return pl_mod


def _append_trace(state, role, content):
    trace = state.get("trace", [])
    trace.append({"role": role, "content": content})
    return trace


def _emit_trace(state, emit, role, content) -> None:
    """追加一条 trace, 并同时推给客户端.

    agent 模式下发射层才真正发出去; rag 模式同样写进 state (排查时有用), 只是
    ``_make_emit`` 会丢掉 —— 节点里因此没有一处 ``if mode == ...``。
    """
    state["trace"] = _append_trace(state, role, content)
    emit(_pipeline().TraceEvent({"role": role, "content": content}))


def _upd(state: dict, messages: Optional[List[BaseMessage]] = None, **fields) -> dict:
    """构造节点返回值: state 的浅拷贝, 但**剥掉 messages 再显式给新值**.

    messages 是 add reducer —— 把收到的列表原样带回会把历史每过一跳翻一倍。
    需要追加消息的节点 (reason / act / 各终端) 用 ``messages=[新消息]`` 只交
    新增部分; 不动 messages 的节点剥掉即可 (通道保持原值)。
    """
    out = {k: v for k, v in state.items() if k != "messages"}
    if messages is not None:
        out["messages"] = messages
    out.update(fields)
    return out


def _recent_history(messages: Optional[List[BaseMessage]]) -> List[dict]:
    """messages -> QueryRewriter / 生成提示词用的 ``{"role","content"}`` 历史 (去掉本轮).

    "本轮"的边界 = **最后一条 HumanMessage**: 它之前的一切才是历史。不能简单
    地"去掉最后一条" —— react 循环里最后一条是决策轮的 AIMessage 或 ToolMessage,
    按位置切会把**当前问题**也当成历史, 提示词里于是出现两遍 (且白付 token)。
    ToolMessage 与决策轮的空 AIMessage 不进历史 (对改写器与生成器都是噪声)。
    """
    msgs = list(messages or [])
    cut = len(msgs)
    for i in range(len(msgs) - 1, -1, -1):
        if isinstance(msgs[i], HumanMessage):
            cut = i
            break
    out: List[dict] = []
    for m in msgs[:cut]:
        if isinstance(m, (SystemMessage, ToolMessage)):
            continue
        content = str(getattr(m, "content", "") or "")
        if not content:
            continue
        role = "assistant" if isinstance(m, AIMessage) else "user"
        out.append({"role": role, "content": content})
    return out


def _llm_window(messages: List[BaseMessage], keep: int) -> List[BaseMessage]:
    """给 LLM 的消息窗口: 最近 keep 条, 且**裁到合法边界**.

    provider (DeepSeek / OpenAI 兼容) 强制两条约束, 违反即 400:
      1. role='tool' 的消息必须紧跟在一个带 tool_calls 的助手消息之后;
      2. 带 tool_calls 的助手消息必须带齐每个 id 的 tool 响应。
    记忆模式下 messages 会超过 keep, 直接切 ``[-keep:]`` 会把窗口开在工具交换的
    中间 —— 2026-10-16 线上实测 400: "Messages with role 'tool' must be a response
    to a preceding message with 'tool_calls'" (连问多轮、第 20 条边界落在 ToolMessage
    上时触发)。回合被取消 (客户端断开) 会让 checkpoint 留下没有响应的 tool_calls,
    同一条约束的反方向, 这里一并兜掉。

    **只裁剪发出去的消息, 不动 state** —— checkpoint 保持原样, 下一轮窗口位置相同,
    结果稳定可复现 (若在这里写回 state, 每轮的裁剪会互相叠加, 行为取决于历史路径)。
    """
    msgs = list(messages)[-max(1, keep):]
    out: List[BaseMessage] = []
    i = 0
    n = len(msgs)
    while i < n:
        m = msgs[i]
        if isinstance(m, ToolMessage):
            # 孤立的工具响应 (其 tool_calls 被切掉或本就没落盘): 丢弃
            i += 1
            continue
        calls = getattr(m, "tool_calls", None) if isinstance(m, AIMessage) else None
        if calls:
            wanted = [str(c.get("id") or c.get("name")) for c in calls]
            got: Dict[str, BaseMessage] = {}
            j = i + 1
            while j < n and isinstance(msgs[j], ToolMessage):
                got[str(getattr(msgs[j], "tool_call_id", ""))] = msgs[j]
                j += 1
            if all(w in got for w in wanted):
                out.append(m)
                out.extend(got[w] for w in wanted)
            # 响应不全 -> 整个交换丢弃 (半个交换两个方向都是 400)
            i = j
            continue
        out.append(m)
        i += 1
    return out


def _history_for_prompt(state, cfg: _NodeCfg) -> List[dict]:
    """生成提示词要带的对话历史 (记忆模式从 messages 派生, 无记忆回落 cfg.history).

    **生成节点必须吃历史** —— 2026-10-16 实测 bug: 记忆写进了 graph state、也喂给了
    决策轮, 但 compose / rag 的生成提示词只有当前问题, 结果追问"我刚才第一句话
    说的是什么"被模型原样吐回 (那一轮它确实只看到一个孤立的问题)。fake LLM 的
    罐头答案掩盖了它 —— 真实模型不读提示词就没法装作记得。

    历史非空时发一条 ``memory`` trace, 让"本轮带了 N 条历史"在 UI 的 trace 面板里
    可见: 这类"状态存了但没接上"的缺陷, 看不见就测不出来。
    """
    keep = int(getattr(settings, "agent_memory_max_messages", 20))
    derived = _recent_history((state.get("messages") or [])[-keep:])
    history = derived or list(cfg.history or [])
    if history:
        _emit_trace(state, cfg.emit, "memory", {"messages": len(history)})
    return history


# ---------- 路由决策解析 ----------

_PLANNER_BLOCK_RE = re.compile(r"\{[^{}]*\"needs_calculator\"[^{}]*\}", re.DOTALL)


def _parse_planner_decision(raw: str) -> dict:
    """解析路由输出: {"needs_calculator", "needs_retrieval", "needs_date"}.

    容错策略:
    1. 整体 json.loads.
    2. 去掉 ```json ... ``` 代码块再 parse.
    3. 正则抓 {"needs_calculator": ...} 块.
    4. fallback 三个都 False / needs_retrieval=True (**fail-safe 走检索**).

    fallback 方向是刻意的, 且与已删除的 reviewer 解析器相反: 路由解析不出来时宁可
    多检索一次 (慢但安全), 而不是凭模型自身知识直答 (快但可能编造)。这个差异就是
    两个解析器当初没有合并成公共骨架的原因。

    Returns: dict, 键齐全, 都是合法类型.
    """
    s = (raw or "").strip()
    fallback = {
        "needs_calculator": False,
        "needs_retrieval": True,
        "needs_date": False,
    }
    if not s:
        return {**fallback, "_fallback_reason": "empty planner output"}
    # 1) 直接
    try:
        d = json.loads(s)
        if isinstance(d, dict) and ("needs_calculator" in d or "needs_retrieval" in d):
            return _normalize_planner(d)
    except Exception:
        pass
    # 2) 去掉代码块
    if "```" in s:
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", s, re.DOTALL)
        if m:
            try:
                d = json.loads(m.group(1))
                if isinstance(d, dict):
                    return _normalize_planner(d)
            except Exception:
                pass
    # 3) 正则抓首块
    m = _PLANNER_BLOCK_RE.search(s)
    if m:
        try:
            d = json.loads(m.group(0))
            return _normalize_planner(d)
        except Exception:
            pass
    # 4) fallback
    return {**fallback, "_fallback_reason": "unparseable planner output"}


def _normalize_planner(d: dict) -> dict:
    def _b(v):
        if isinstance(v, bool):
            return v
        if isinstance(v, int):
            return bool(v)
        if isinstance(v, str):
            return v.strip().lower() in ("true", "yes", "1", "y", "是")
        return False

    return {
        "needs_calculator": _b(d.get("needs_calculator")),
        "needs_retrieval": _b(d.get("needs_retrieval", True)),
        "needs_date": _b(d.get("needs_date")),
    }


def _extract_calc_expression(text: str) -> str:
    """从用户问句里尽力抠出算术表达式. 抠不到就把整句交给 calculator,
    让 calculator 报错也能在 trace 留下调用记录."""
    m = re.search(r"[\d\.\(\)\+\-\*/\s]+", text or "")
    expr = (m.group(0).strip() if m else (text or "")).strip()
    return expr or "0"


# ---------------------------------------------------------------------------
# 入口条件边
# ---------------------------------------------------------------------------


def entry_router(state) -> str:
    """入口条件边: 纯函数, 只读 ``state["mode"]`` 与引擎, **零 LLM 调用**.

    rag 模式跳过路由 —— 用户显式选了"常规检索问答", 再花一次 LLM 往返去判断
    "要不要检索" 是纯粹的延迟浪费 (真实 provider 1~5s)。

    agent 模式按 ``state["engine"]`` (缺省回落 ``settings.agent_engine``) 分流:
    router 引擎走旧的 route 一次性路由; react 引擎直接进 reason 工具循环。

    条件边里不要改 state: branch 函数的副作用不会落盘 (旧 route_review 踩过这个坑,
    写在 edge 里的 retry_count 递增被丢弃)。
    """
    if (state.get("mode") or "rag") != "agent":
        return "rag"
    engine = state.get("engine") or getattr(settings, "agent_engine", "react") or "react"
    return "route" if engine == "router" else "reason"


# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------


async def route_node(state, config=None):
    """LLM 路由: 决定要不要检索知识库 / 要不要调用工具.

    只判断"要不要检索", 不决定"怎么检索" —— 检索 query 由下游的改写器
    (QueryRewriter) 用用户原问题扩展, 这是刻意的分工 (让模型产出关键词的那版实现
    信息量还不如原问题, 已删除)。
    """
    cfg = _cfg(config)
    llm = get_llm()
    # 带上完整 messages (含 history): 未来"根据上下文补全问题"就落在这里。
    # history 为空时 messages 只有本轮问题, 与改造前逐字节等价。
    msgs = [SystemMessage(content=ROUTER_SYS), *state["messages"]]
    # LLM 是阻塞同步 SDK, async 节点里必须卸载到线程池, 否则卡住事件循环。
    out = await asyncio.to_thread(llm.invoke, msgs)
    # 推理模型可能把 <think> 思考过程混进 content, 会让 JSON 解析直接失败
    # (表现为"永远走不到工具分支"), 所以先剥掉再解析。
    raw = strip_think(out.content if hasattr(out, "content") else str(out))
    decision = _parse_planner_decision(raw)
    state["plan"] = raw
    state["plan_decision"] = decision
    # 预留: 多轮记忆落地时, 在这里消费决策里的改写结果覆盖 state["question"]。
    # 当前 _normalize_planner 只产出三个布尔键, 所以这一行恒等于"不改写"。
    _emit_trace(state, cfg.emit, "router", {"raw": raw, "decision": decision})
    # 不能 return state: messages 是 add reducer, 原样带回等于把历史整段重放一遍,
    # 生成提示词的【对话历史】里会混进重复历史与当前问题 (见 _recent_history 口径)。
    # plan / plan_decision / trace 都是普通通道且已就地改过, _upd 剥掉 messages 即可。
    return _upd(state)


def route_after_route(state) -> str:
    """条件边: 纯函数, 只读 ``state["plan_decision"]``.

    - ``needs_retrieval`` 为真 (或缺决策) -> "rag"
    - 否则需要 calculator / current_date -> "tools"
    - 都不需要 -> "direct"

    ``d.get("needs_retrieval", True)`` 里的 True 是**第二道保险**: 即使决策缺失
    (图被直接 invoke、测试手工构造 state), 也走检索而不是凭模型知识直答。
    """
    d = state.get("plan_decision") or {}
    if d.get("needs_retrieval", True):
        return "rag"
    if d.get("needs_calculator") or d.get("needs_date"):
        return "tools"
    return "direct"


def run_tools(decision: dict, question: str, state, emit) -> List[str]:
    """按路由决策**确定性**调用 calculator / current_date, 返回给提示词用的文本片段.

    刻意不走 LLM tool-calling: 决策已经在 route_node 出过了, 这里只是执行 ——
    让模型再决定一次既多一次往返, 又多个不确定源。

    ``build_tools`` 按模块属性查找 (不是 import 进来的局部名), 测试 monkeypatch
    ``app.agents.graph.build_tools`` 打的就是这里。
    """
    tools = build_tools(kb_id=state.get("kb_id") or "default")
    tool_map = {t.name: t for t in tools}
    parts: List[str] = []
    if decision.get("needs_calculator"):
        calc = tool_map.get("calculator")
        if calc is not None:
            expression = _extract_calc_expression(question)
            calc_text = calc.invoke(expression)
            parts.append(f"calculator: {calc_text}")
            _emit_trace(state, emit, "tool:calculator", calc_text)
    if decision.get("needs_date"):
        date_t = tool_map.get("current_date")
        if date_t is not None:
            date_text = date_t.invoke({})
            parts.append(f"current_date: {date_text}")
            _emit_trace(state, emit, "tool:current_date", date_text)
    return parts


# ---------------------------------------------------------------------------
# react 引擎: reason ⇄ act 工具循环 + compose 终端
# ---------------------------------------------------------------------------


async def reason_node(state, config=None):
    """决策节点: 看消息历史 (含工具观测) 决定下一步调哪些工具, 或宣布证据足够.

    决策轮**不流式**: 是否还有 tool_calls 要等整条响应出来才知道, 乐观转发 token
    的话, 决策轮的思考文本会混进最终答案。最终答案统一由 compose 流式生成,
    提示词 / 流式 / 引用编号因此仍然只有一份实现 (见模块 docstring)。
    """
    cfg = _cfg(config)
    kb_id = state.get("kb_id") or "default"
    tools = build_tools(kb_id)
    llm = get_llm()
    # fake LLM (FakeLLM / LocalDemoLLM) 若未实现 bind_tools 则降级为纯文本对话:
    # 图结构不变, 模型无法发起工具调用, 条件边据此直接进 compose。
    bound = llm.bind_tools(tools) if hasattr(llm, "bind_tools") else llm
    # 上下文预算: 只喂最近 N 条, 更早的历史留在 checkpoint, 不进模型上下文。
    # 必须走 _llm_window 而不是裸切片: 窗口开在工具交换中间会被 provider 400。
    keep = int(getattr(settings, "agent_memory_max_messages", 20))
    msgs = [SystemMessage(content=AGENT_SYS), *_llm_window(state["messages"], keep)]
    ai = await asyncio.to_thread(bound.invoke, msgs)
    calls = list(getattr(ai, "tool_calls", None) or [])
    rounds = int(state.get("tool_rounds") or 0)
    if calls:
        _emit_trace(
            state,
            cfg.emit,
            "reason",
            {
                "rounds": rounds + 1,
                "tool_calls": [{"name": c.get("name"), "args": c.get("args")} for c in calls],
            },
        )
        return _upd(state, messages=[ai], tool_rounds=rounds + 1)
    _emit_trace(state, cfg.emit, "reason", {"rounds": rounds + 1, "decision": "done"})
    return _upd(state)


def route_after_reason(state) -> str:
    """条件边: 末条消息带 tool_calls -> act; 否则 (模型宣布足够) -> compose."""
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    if isinstance(last, AIMessage) and getattr(last, "tool_calls", None):
        return "act"
    return "compose"


async def act_node(state, config=None):
    """执行上一轮 reason 的全部 tool_calls, 观测作为 ToolMessage 回填.

    hybrid_search / read_* 的结构化候选合并进 ``state["sources"]`` (按 chunk id
    去重、首次出现顺序编号); 无候选工具 (calculator 等) 的观测进 ``tool_notes``。
    工具失败**不终止循环**: 观测里带上错误文本, 让模型看到后自行调整重试。
    """
    cfg = _cfg(config)
    kb_id = state.get("kb_id") or "default"
    msgs = state.get("messages") or []
    last = msgs[-1] if msgs else None
    calls = (
        list(getattr(last, "tool_calls", None) or []) if isinstance(last, AIMessage) else []
    )
    if not calls:
        return state
    merged = list(state.get("sources") or [])
    notes = list(state.get("tool_notes") or [])
    tool_msgs: List[BaseMessage] = []
    for c in calls:
        name = str(c.get("name") or "")
        handler = HANDLERS.get(name)
        try:
            if handler is None:
                obs, hits = f"未知工具: {name}", []
            else:
                # 检索/重排是 CPU 密集同步代码, 卸线程池 (与项目其它 LLM/检索
                # 调用点同口径), 不卡事件循环。
                obs, hits = await asyncio.to_thread(handler, dict(c.get("args") or {}), kb_id)
        except Exception as e:  # noqa: BLE001 - 失败转观测文本, 循环继续
            obs, hits = f"工具执行失败: {e}", []
        # 观测长度上限: read_parent 会带回多条长文, 不设上限会随轮数累积击穿上下文。
        cap = int(getattr(settings, "context_snippet_chars", 2000)) * 4
        obs = (obs or "")[:cap]
        tool_msgs.append(ToolMessage(content=obs, tool_call_id=str(c.get("id") or name)))
        if hits:
            merged = merge_sources(merged, hits)
        else:
            notes.append(obs[:500])
        _emit_trace(state, cfg.emit, "tool:" + name, (obs or "")[:200])
    return _upd(state, messages=tool_msgs, sources=merged, tool_notes=notes)


def after_act(state) -> str:
    """预算闸门: 工具轮数用尽后不再给模型下一轮, 直接进 compose.

    闸在 act 之后而不是在 reason 里拦: 拦在 reason 里等于先生成 tool_calls 再
    丢弃, 白付一次 LLM 往返; 闸在边上, 预算耗尽的下一跳根本不进 reason。
    """
    if int(state.get("tool_rounds") or 0) >= int(getattr(settings, "agent_max_steps", 4)):
        return "compose"
    return "reason"


async def compose_node(state, config=None):
    """react 终端: 用循环收集到的证据流式生成最终答案.

    contexts = sources 正文 + tool_notes, 顺序与 ``state["sources"]`` 一致 ——
    build_rag_prompt 的 [n] 编号因此与前端引用溯源严格对齐 (引用红线, 同
    rag_node 的口径)。无 sources 且无 notes 时走 direct 提示词 (纯闲聊/直答)。
    """
    pl = _pipeline()
    cfg = _cfg(config)
    question = state.get("question") or ""
    sources = list(state.get("sources") or [])
    notes = list(state.get("tool_notes") or [])

    # 即使空列表也必须发 SourcesEvent: run_rag 靠"收到过它"区分检索失败与生成
    # 失败, 缺了它生成阶段失败会被整条抛出 (见 tools_node 的同款注释)。
    cfg.emit(pl.SourcesEvent(sources))
    if not sources:
        cfg.emit(pl.NoticeEvent(pl.NO_KB_NOTICE))

    if sources or notes:
        contexts = [c.get("document") or "" for c in sources] + notes
        prompt = pl.build_rag_prompt(question, contexts, history=_history_for_prompt(state, cfg))
    else:
        prompt = pl.build_direct_prompt(question, history=_history_for_prompt(state, cfg))
    answer = await pl._generate_events(
        prompt, stream_tokens=cfg.stream_tokens, trace=cfg.trace, emit=cfg.emit
    )
    # final 必须由终端节点写 (谁要是删了它, 上层会读到空答案且不报任何错)。
    state["final"] = answer
    if answer:
        _emit_trace(state, cfg.emit, "answer", answer[:500])
        # 答案作为 AIMessage 追加: 记忆模式下下一轮对话带上下文; 无记忆时
        # messages 只多一条终态 (没有消费方, 行为等价)。
        return _upd(state, messages=[AIMessage(content=answer)])
    return _upd(state)


async def rag_node(state, config=None):
    """检索 + 生成. rag 模式与 agent 模式共用, 也是"复用 RAG 链路"的落点."""
    pl = _pipeline()
    cfg = _cfg(config)
    question = state.get("question") or ""
    kb_id = state.get("kb_id") or "default"
    decision = state.get("plan_decision") or {}

    # 工具结果**追加在检索上下文之后**: 提示词里的 [1..N] 必须始终对应
    # state["sources"], 否则前端引用与正文对不上 (引用溯源的整条链就断在这)。
    tool_ctx = run_tools(decision, question, state, cfg.emit)

    # 改写器与**生成提示词**吃同一份历史: 记忆模式下从 messages 派生 (含上一轮的
    # 答案), 无记忆时回落 cfg.history (与 messages 的构造来源同源, 结果等价)。
    history = _history_for_prompt(state, cfg)
    sources, contexts = await pl._retrieve_sources(
        question, kb_id, cfg.top_k, cfg.trace, history
    )
    state["sources"] = sources
    cfg.emit(pl.SourcesEvent(sources))
    if sources:
        _emit_trace(
            state,
            cfg.emit,
            "retriever",
            {
                "kb_id": kb_id,
                "count": len(sources),
                "preview": (contexts[0] if contexts else "")[:200],
            },
        )
    else:
        _emit_trace(state, cfg.emit, "retriever", "未检索到相关文档。")

    prompt = pl.build_rag_prompt(question, contexts + tool_ctx, history=history)
    answer = await pl._generate_events(
        prompt, stream_tokens=cfg.stream_tokens, trace=cfg.trace, emit=cfg.emit
    )
    # final 必须由终端节点写: 曾经只有 reviewer 节点写它, 谁要是删了那个节点,
    # 上层会读到空答案而且**不报任何错**。
    state["final"] = answer
    if answer:
        _emit_trace(state, cfg.emit, "answer", answer[:500])
        # 答案作为 AIMessage 追加: 记忆模式下下一轮对话带上下文 (见 compose_node)。
        return _upd(state, messages=[AIMessage(content=answer)])
    return _upd(state)


async def tools_node(state, config=None):
    """只算工具、不检索: 路由判 needs_retrieval=false 且需要 calculator / 当前日期."""
    pl = _pipeline()
    cfg = _cfg(config)
    question = state.get("question") or ""
    decision = state.get("plan_decision") or {}

    # 没有检索也要发一条空的 SourcesEvent: run_rag 用"是否收到过它"判断
    # 检索阶段有没有走完 (got_sources), 缺了它, 生成阶段的失败会被误判成
    # 检索失败而整条抛出去 —— 静默的行为回归。
    cfg.emit(pl.SourcesEvent([]))
    cfg.emit(pl.NoticeEvent(pl.NO_KB_NOTICE))
    tool_ctx = run_tools(decision, question, state, cfg.emit)
    if not tool_ctx:
        _emit_trace(state, cfg.emit, "tools", "(no tool selected)")

    prompt = pl.build_rag_prompt(question, tool_ctx, history=_history_for_prompt(state, cfg))
    answer = await pl._generate_events(
        prompt, stream_tokens=cfg.stream_tokens, trace=cfg.trace, emit=cfg.emit
    )
    state["final"] = answer
    if answer:
        _emit_trace(state, cfg.emit, "answer", answer[:500])
        return _upd(state, messages=[AIMessage(content=answer)])
    return _upd(state)


async def direct_node(state, config=None):
    """不检索也不算工具, 直接凭模型自身知识回答.

    这条路径**必须**带 NO_KB_NOTICE 标记: 企业知识库场景下, 用户问内部信息
    (报销标准、流程编号) 时模型不会说"我不知道", 而是编一个听起来合理的答案,
    而且这次连 sources 都没有, 无法溯源。
    """
    pl = _pipeline()
    cfg = _cfg(config)
    question = state.get("question") or ""

    cfg.emit(pl.SourcesEvent([]))
    cfg.emit(pl.NoticeEvent(pl.NO_KB_NOTICE))
    _emit_trace(state, cfg.emit, "direct", {"question": question})

    prompt = pl.build_direct_prompt(question, history=_history_for_prompt(state, cfg))
    answer = await pl._generate_events(
        prompt, stream_tokens=cfg.stream_tokens, trace=cfg.trace, emit=cfg.emit
    )
    state["final"] = answer
    if answer:
        _emit_trace(state, cfg.emit, "answer", answer[:500])
        return _upd(state, messages=[AIMessage(content=answer)])
    return _upd(state)


# ---------------------------------------------------------------------------
# 初始 state / 建图
# ---------------------------------------------------------------------------


def _to_messages(history: Optional[List[dict]]) -> List[BaseMessage]:
    """把 API 层的 ``{"role", "content"}`` 历史转成 BaseMessage.

    只在这里转换 —— 不让 BaseMessage 泄漏到 API 层 (pipeline 的 ``history`` 形参
    就是纯 dict)。``None`` 与 ``[]`` 都返回空列表, 保证预留接口零行为变化。
    """
    out: List[BaseMessage] = []
    for m in history or []:
        if not isinstance(m, dict):
            continue
        text = str(m.get("content") or "")
        if not text:
            continue
        if m.get("role") == "assistant":
            out.append(AIMessage(content=text))
        else:
            out.append(HumanMessage(content=text))
    return out


def new_agent_state(
    question: str,
    kb_id: str = "default",
    mode: str = "rag",
    history: Optional[List[dict]] = None,
    engine: Optional[str] = None,
) -> AgentState:
    """构造一次图运行的初始 state.

    抽成工厂是为了消除调用点上的复制粘贴 —— TypedDict 不提供运行时默认值, 少一个键
    LangGraph 就会在节点读 ``state["xxx"]`` 时抛 KeyError, 而这类错误只有真正跑到
    那条链路时才会炸。集中在工厂里, 改一次即可。

    Args:
        question: 用户问题, 包成 HumanMessage 追加在 history 之后.
        kb_id: 目标知识库; 必须显式传, 否则检索节点会静默走 "default",
               架空私有 kb 的 ACL.
        mode: "rag" 直接进检索节点; "agent" 按引擎进 route 或 reason.
        history: 无记忆模式的多轮历史, 元素 ``{"role", "content"}``; **记忆模式下
               必须传 None** —— 历史已在 checkpoint 里, 再传会被 add reducer
               整段重放 (等于每轮把全部历史说两遍)。
        engine: agent 引擎覆盖 ("react" | "router"); None 时入口条件边回落
               ``settings.agent_engine``。
    """
    return AgentState(
        messages=[*_to_messages(history), HumanMessage(content=question)],
        kb_id=kb_id or "default",
        mode="agent" if mode == "agent" else "rag",
        engine=engine or "",
        question=question,
        history=list(history or []),
        plan="",
        plan_decision={},
        tool_rounds=0,
        tool_notes=[],
        final="",
        sources=[],
        trace=[],
    )


# 编译产物缓存, 按是否带记忆分两个桶 (见 build_agent_graph 的取舍说明).
_GRAPH_CACHE: Dict[bool, Any] = {}


def build_agent_graph(use_memory: Optional[bool] = None):
    """获取编译好的图 (进程内复用, 不重复编译).

    为什么可以缓存:
        图的结构完全静态 —— 节点函数、边、条件边都是模块级常量, 不依赖任何
        请求参数; 一次运行的全部可变数据都通过 ``ainvoke(state, config)`` 传入,
        连 emit 通道都走 config。编译产物 (Pregel) 本身是不可变的计算图描述,
        并发调用之间不共享可变状态。

    为什么要缓存:
        ``StateGraph.compile()`` 会做 schema 校验、channel 构建、节点/边注册 ——
        跟请求量成正比地重复发生, 且拿不到任何新信息。

    为什么按 ``use_memory`` 分两个桶:
        带 checkpointer 的编译产物持有会话状态, 串扰风险因此只存在于**同一
        thread 内** —— 这正是 LangGraph 官方的 thread_id 隔离模式: checkpointer
        按 thread_id (= "{user}:{session}") 分桶, 同一编译产物被多用户并发复用
        也互不可见。无记忆那份 (False 桶) 不持状态, 供无 session 的请求与
        不想带状态的测试使用。

    保留成函数 (而不是直接导出编译产物) 的理由:
        测试需要整个替换成 fake graph (见 tests/test_integration_retrieval.py)。

    Args:
        use_memory: None -> **不带记忆**。是否带记忆由唯一的调用决策点
            (pipeline._graph_events) 按 ``settings.agent_memory_enabled`` 与
            session_id 算好后显式传入; 这里不读 settings, 免得直接调图的脚本 /
            测试 (不带 thread_id) 意外拿到 checkpointer 然后被 langgraph 拒收。
    """
    if use_memory is None:
        use_memory = False
    g = _GRAPH_CACHE.get(use_memory)
    if g is None:
        g = _compile_agent_graph(use_memory)
        _GRAPH_CACHE[use_memory] = g
    return g


class _LazyAsyncSqliteSaver(AsyncSqliteSaver):
    """连接与建表都推迟到第一次真正读写 checkpoint 时.

    为什么需要懒初始化: ``AsyncSqliteSaver.__init__`` 里有 ``get_running_loop()``
    (锁要绑 loop), aiosqlite 建连也要 ``await`` —— 而 ``build_agent_graph`` 是
    同步函数 (测试 / 脚本直接调, pipeline 也按同步名字 monkeypatch). 好在
    checkpoint 的读写全部发生在 ``ainvoke`` 的事件循环里, 在那里补初始化正好;
    每个公开方法先进 ``_ensure_ready``, 同步构造的图因此永远不会在错误的地方
    碰到未建连的 conn.

    ``_ensure_ready`` 是双检锁: 并发首跳只有一个协程真正建连 (aiosqlite 的
    工作线程只能 start 一次, 第二次 ``await conn`` 会 RuntimeError).

    单 loop 假设: ``AsyncSqliteSaver.lock`` 绑定首次使用的 loop, 跨 loop 复用
    会 RuntimeError. 生产是 uvicorn 单 worker 单 loop (Dockerfile 里有论证),
    不会触发; 测试里由 conftest 每个用例重置图缓存, 各自拿全新的 saver.
    """

    def __init__(self, db_path: Path) -> None:
        # 刻意不调 super().__init__(): 它要求 running loop. self._ready 为 False
        # 期间本实例上除了 _ensure_ready 不该有任何属性访问 (Pregel 只在 ainvoke
        # 里调 checkpointer 的 async 方法, 全部被下面的 override 接住).
        self._db_path = db_path
        self._init_lock: Optional[asyncio.Lock] = None
        self._ready = False

    async def _ensure_ready(self) -> None:
        if self._ready:
            return
        # 锁也延迟创建: asyncio.Lock 同样绑 loop, 不能在同步 __init__ 里建.
        # "判 None 再赋值" 之间没有 await, 协程不会在这里被切走, 无竞态.
        if self._init_lock is None:
            self._init_lock = asyncio.Lock()
        async with self._init_lock:
            if self._ready:
                return
            conn = aiosqlite.connect(str(self._db_path))
            # aiosqlite 的工作线程默认非 daemon: 异常逃逸后残留的连接会卡在
            # 队列 get 上, 整个进程退出时被它挂死 (实测一次 setup 失败就把 pytest
            # 拖到超时). 启动前改成 daemon, 失败路径不再拖住进程.
            conn._thread.daemon = True  # noqa: SLF001
            await conn  # __await__: 起工作线程 + 真正建连
            if not hasattr(conn, "is_alive"):
                # langgraph-checkpoint-sqlite 2.x 的 setup() 会调 conn.is_alive(),
                # 但**已发布**的 aiosqlite (0.21/0.22) 都没有这个方法 —— 上游依赖
                # 了未发布的 aiosqlite master, 直接用必 AttributeError. 补一个恒真
                # shim: 连接与进程同生命周期, 线程真死了的话后续操作自然会报错.
                conn.is_alive = lambda: True  # type: ignore[method-assign]
            AsyncSqliteSaver.__init__(self, conn)  # 此时 running loop 一定存在
            await AsyncSqliteSaver.setup(self)  # 建表 + WAL (is_setup 幂等)
            self._ready = True

    # 每个被 Pregel 调用的 async 方法先补初始化, 再走父类实现.
    # 参数名必须与父类一致 (Pregel 按 keyword 传 filter/before/limit).

    async def aget_tuple(self, config):  # noqa: ANN001, ANN201
        await self._ensure_ready()
        return await super().aget_tuple(config)

    async def alist(self, config, *, filter=None, before=None, limit=None):  # noqa: A002, ANN001
        await self._ensure_ready()
        async for item in super().alist(config, filter=filter, before=before, limit=limit):
            yield item

    async def aput(self, config, checkpoint, metadata, new_versions):  # noqa: ANN001
        await self._ensure_ready()
        return await super().aput(config, checkpoint, metadata, new_versions)

    async def aput_writes(self, config, writes, task_id, task_path=""):  # noqa: ANN001
        await self._ensure_ready()
        return await super().aput_writes(config, writes, task_id, task_path)

    async def adelete_thread(self, thread_id: str) -> None:
        await self._ensure_ready()
        return await super().adelete_thread(thread_id)


def _build_sqlite_checkpointer() -> _LazyAsyncSqliteSaver:
    """构造 SQLite 持久化的 checkpointer (进程内单份, 由 _GRAPH_CACHE 间接持有).

    为什么不再是 MemorySaver: 进程内字典重启即清, "继续历史对话"在后端重启后
    模型完全失忆 —— 且记忆模式下初始 state 的 history 强制传 None (历史"已在
    checkpoint 里"), checkpoint 一丢比无记忆模式还彻底。SQLite 落盘后 checkpoint
    随 ``agent_checkpoint_db`` 持久化, 默认在 data/ 下, Docker 的 ./data volume
    天然覆盖, 不需要额外挂载。连接生命周期 = 图缓存生命周期, 进程退出由 OS
    回收, WAL 模式下不存在"写一半丢事务"。
    """
    db_path = Path(settings.agent_checkpoint_db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return _LazyAsyncSqliteSaver(db_path)


def _compile_agent_graph(use_memory: bool):
    """真正建图 + compile 的地方, 被 ``build_agent_graph`` 的双桶缓存包装."""
    g = StateGraph(AgentState)
    g.add_node("route", route_node)
    g.add_node("reason", reason_node)
    g.add_node("act", act_node)
    g.add_node("compose", compose_node)
    g.add_node("rag", rag_node)
    g.add_node("tools", tools_node)
    g.add_node("direct", direct_node)
    # 三个入口: rag 直连检索; router 引擎走 route; react 引擎进 reason 循环。
    g.set_conditional_entry_point(
        entry_router, {"rag": "rag", "route": "route", "reason": "reason"}
    )
    # router 引擎: route -> rag | tools | direct
    g.add_conditional_edges(
        "route",
        route_after_route,
        {"rag": "rag", "tools": "tools", "direct": "direct"},
    )
    # react 引擎: reason -> act (有 tool_calls) | compose (证据足够);
    # act -> reason (预算内) | compose (预算耗尽, 见 after_act)。
    g.add_conditional_edges(
        "reason", route_after_reason, {"act": "act", "compose": "compose"}
    )
    g.add_conditional_edges("act", after_act, {"reason": "reason", "compose": "compose"})
    g.add_edge("rag", END)
    g.add_edge("tools", END)
    g.add_edge("direct", END)
    g.add_edge("compose", END)
    if use_memory:
        return g.compile(checkpointer=_build_sqlite_checkpointer())
    return g.compile()


def reset_agent_graph_cache() -> None:
    """仅供测试: 清掉两个桶, 让下一次 build_agent_graph() 重新编译.

    用于给节点函数打 monkeypatch 之后强制重建图 (已编译的图持有的是编译那一刻的
    函数对象引用, patch 模块属性不会影响已编译的图 —— 注意 ``build_tools`` 是个例外,
    它在 ``run_tools`` 函数体内按名字查找, patch 后即时生效)。
    """
    _GRAPH_CACHE.clear()


def forget_thread(thread_id: str) -> bool:
    """同步删除一条会话记忆. 返回是否真的删了.

    异步 checkpointer (_LazyAsyncSqliteSaver) 一律返回 False: 它继承的同步
    ``delete_thread`` 走 ``run_coroutine_threadsafe(...).result()``, 在事件循环
    线程里调用会**自死锁** (等一个调度不出去的协程)。生产删除路径 (FastAPI
    端点, 永远在 loop 里) 必须走 ``aforget_thread``; 本函数只对"同步且支持
    删除"的 checkpointer 有意义, 保留作兼容入口。
    """
    checkpointer = _GRAPH_CACHE.get(True) and getattr(_GRAPH_CACHE[True], "checkpointer", None)
    if checkpointer is None or isinstance(checkpointer, _LazyAsyncSqliteSaver):
        return False
    deleter = getattr(checkpointer, "delete_thread", None)
    if not callable(deleter):
        return False
    deleter(thread_id)
    return True


async def aforget_thread(thread_id: str) -> bool:
    """删除一条会话记忆 (history 删除会话时调用, 在事件循环里 await).

    checkpoint 落在 SQLite 之后这行才是**真删**: 残留不再随重启消失。图还没
    编译时返回 False (无记忆可删, 不是错误); 懒初始化的连接会在这一步建好。
    """
    graph = _GRAPH_CACHE.get(True)
    checkpointer = getattr(graph, "checkpointer", None) if graph is not None else None
    if checkpointer is None:
        return False
    await checkpointer.adelete_thread(thread_id)
    return True
