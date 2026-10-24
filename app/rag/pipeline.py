"""RAG pipeline: 检索 + 重排 + 生成,作为统一的对外入口。

被 API 层、Agent 层和评估脚本共同使用。可选接入 LangFuse 做可观测。
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, List, Union

from loguru import logger

# LLM 走模块属性查找 (llm_router.get_llm()) 而不是 `from ... import get_llm`:
# 后者在 import 时就把函数对象绑死在 pipeline 的命名空间里, 之后测试
# monkeypatch.setattr(llm_router, "get_llm", ...) 对这里完全无效
# (tests/test_integration_retrieval.py 的 _install_fakes 就是这么 patch 的).
from app.agents import llm_router
from app.core.config import settings
from app.core.observability import get_tracer
from app.core.text_utils import ThinkBlockFilter, strip_think
from app.rag.metadata import MetadataEnhancer
from app.rag.query_rewrite import QueryRewriter
from app.rag.reranker import RerankOptimizer, get_reranker
from app.rag.retriever import HybridRetriever, get_retriever, rrf_fuse


# 查询改写/HyDE 的 LLM 适配器不可用时只告警一次 (见 retrieve_and_rerank):
# 静默吞掉会让 key 配错的生产环境表现为"接口全 200 但检索质量悄悄降级".
_warned_no_llm_adapter = False

_PROMPT_TEMPLATE = (
    "你是一个严谨的企业知识库助手。请严格基于【检索结果】回答用户问题，"
    "并在引用处使用 [1] [2] 等标注。如果检索结果中没有答案，请直接说「未找到相关信息」。\n\n"
    "{history}【检索结果】\n{contexts}\n\n"
    "【用户问题】{question}\n\n"
    "【回答】"
)

# 对话历史的渲染上限. 保留**最近**的整条消息, 超出预算的更早消息丢掉 ——
# 追问通常只依赖最近一两轮, 而历史无限增长会挤掉检索结果的位置.
_HISTORY_MAX_CHARS = 1500


def render_history(history: list[dict] | None, max_chars: int = _HISTORY_MAX_CHARS) -> str:
    """把 ``[{"role", "content"}]`` 历史渲染成提示词里的一段文本 (无历史返回 "").

    从尾部往前收集**整条**消息, 直到预算用完 —— 不从中间截断, 免得把一条回答
    切成半句留在上下文里.
    """
    lines: list[str] = []
    used = 0
    for m in reversed(list(history or [])):
        if not isinstance(m, dict):
            continue
        text = str(m.get("content") or "").strip()
        if not text:
            continue
        role = "助手" if m.get("role") == "assistant" else "用户"
        line = f"{role}: {text}"
        if lines and used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    lines.reverse()
    return "\n".join(lines)


def _history_block(history: list[dict] | None) -> str:
    """历史分节; 无历史时是空串, 提示词与改造前逐字节一致."""
    body = render_history(history)
    if not body:
        return ""
    return (
        "【对话历史】(用于理解指代与追问; 引用编号一律以本次【检索结果】为准)\n"
        + body
        + "\n\n"
    )


def build_rag_prompt(
    question: str,
    contexts: List[str],
    limit: int | None = None,
    history: list[dict] | None = None,
) -> str:
    """把检索结果与问题拼成提示词 (API 主链路与 run_rag 共用).

    之所以抽出来: 两处原先各写一份, 措辞已经悄悄分叉 —— 一处说"在回答末尾标注
    引用来源", 一处说"在引用处使用 [1][2] 等标注". 改一处另一处不生效, 同一份
    文档在两条链路上看到的提示词不一致. 这类分叉不会报错, 只会让对比失去意义.

    contexts 传纯文本列表 (不是候选 dict); 超过 context_snippet_chars 的单条会被
    截断 —— 该上限与 Agent 工具共用, 避免"同一份文档 RAG 答得出、Agent 答不出".

    ``history`` 是多轮对话 (元素 ``{"role","content"}``). 2026-10-16 修: 此前历史
    只喂给检索改写器, **生成提示词本身不带历史** —— 表现为模型能看到检索结果却
    答不对追问 ("我刚才第一句话说的是什么" 被原样吐回, 因为这一轮它确实只看到
    一个孤立的问题). 无历史时渲染为空串, 提示词与此前逐字节一致.
    """
    if limit is None:
        limit = getattr(settings, "context_snippet_chars", 2000)
    ctx_text = "\n\n".join(f"[{i + 1}] {c[:limit]}" for i, c in enumerate(contexts))
    return _PROMPT_TEMPLATE.format(
        contexts=ctx_text or "(无)", question=question, history=_history_block(history)
    )


# 无检索直答用的模板. 刻意**不**复用 _PROMPT_TEMPLATE: 后者硬编码了【检索结果】
# 分节, 没有检索结果时会留下一个空分节, 反而诱导模型去"引用"不存在的内容.
_DIRECT_PROMPT_TEMPLATE = (
    "你是一个严谨的企业知识库助手。本次**没有**检索知识库, 请凭你已有的知识回答用户问题。"
    "如果问题涉及公司内部信息、专有流程, 或你不确定答案, 请直接说明无法确认, 不要编造。\n\n"
    "{history}【用户问题】{question}\n\n"
    "【回答】"
)


def build_direct_prompt(question: str, history: list[dict] | None = None) -> str:
    """无检索直答的提示词 (agent 模式下路由判定不需要检索时用).

    与 build_rag_prompt 一样收在 pipeline 里, 而不是散到 graph.py —— 提示词分叉
    在这个项目里已经出过两次事故 (见 build_rag_prompt 的 docstring).

    ``history`` 同 build_rag_prompt: 多轮追问必须带历史, 否则模型只看到一个孤立
    的当前问题 (2026-10-16 的真 bug).
    """
    return _DIRECT_PROMPT_TEMPLATE.format(
        question=question, history=_history_block(history)
    )


# ===========================================================================
# 统一编排的事件模型
# ===========================================================================
#
# 原先 API 层 (chat.py 的非流式 / 流式两个端点) 和这里的 run_rag 各写一份
# "检索 -> 拼提示词 -> 调 LLM -> 组装输出", 三份的输出口径还不一致: run_rag 经
# _format_contexts 只带出 content/id/score, **丢掉 metadata 与 doc_id**, 而
# chat.py 的 SourceItem 恰好需要它们 (引用溯源就断在这里).
#
# 现在收敛成**一个** async 生成器 stream_rag(), 三个消费方各取所需:
#   run_rag               -> drain 成 5-key dict (RAGAS / 集成测试)
#   POST /api/v1/chat     -> drain 成 ChatResponse
#   POST /api/v1/chat/stream -> 逐事件转成 SSE 帧 (线格式保持不变)
#
# 用 dataclass 而不是 dict: 消费方靠 isinstance 分派, 字段名写错在构造时就报错.


@dataclass(frozen=True)
class SourcesEvent:
    """检索完成: 本次要引用的候选, 已按 top_k 截断.

    元素是 canonical 形状 ``{id, document, metadata, score}`` —— 两个来源
    (retrieve_and_rerank 候选的 rerank_score/rrf_score、agent graph 的 score)
    在 normalize_source() 里统一, 消费方不必再判后端。
    """

    sources: List[dict]


@dataclass(frozen=True)
class TraceEvent:
    """agent 模式的一条推理轨迹 (仅 stream_tokens=True 时逐条产出)."""

    entry: dict


@dataclass(frozen=True)
class TokenEvent:
    """一段可展示的正文, 已经过滤掉 <think> 块."""

    text: str


@dataclass(frozen=True)
class DoneEvent:
    """生成结束.

    ``answer`` 恒为最终答案; ``stream_tokens=True`` 时它等于所有
    TokenEvent.text 的拼接, 消费方可以只用它、不必自己聚合。
    ``agent_trace`` 仅 agent 模式非空。
    """

    answer: str
    agent_trace: List[dict] = field(default_factory=list)


@dataclass(frozen=True)
class NoticeEvent:
    """一条给用户看的提示, **不进入答案正文**.

    目前只有一个来源: agent 模式路由判定"不需要检索"时的 ``NO_KB_NOTICE``.
    为什么不拼进答案: DoneEvent.answer 会被 run_rag 送去 RAGAS 打分、也会被前端
    落进历史会话, 掺进"[未使用知识库]"这类运维提示会污染这两条下游.
    """

    text: str


# 无检索直答时下发给前端的标记. 走独立事件而不是塞进 answer (见 NoticeEvent).
NO_KB_NOTICE = "本次未使用知识库"


RagEvent = Union[SourcesEvent, TraceEvent, TokenEvent, DoneEvent, NoticeEvent]


def normalize_source(c: dict) -> dict:
    """把两种候选形状统一成 ``{id, document, metadata, score}``.

    - retrieve_and_rerank 的候选带 ``rerank_score`` (回退 ``rrf_score``)
    - agent graph 的 sources 已经在 graph.py 里压平成 ``score``

    分数只参与**同一批内部**的相对排序, 所以两个口径不会改变结果; 但下游
    (SourceItem / 评估脚本) 只认一个字段名, 到这里统一掉, 免得每个消费方
    各写一遍 ``c.get("rerank_score", c.get("rrf_score", c.get("score")))``。
    """
    return {
        "id": c.get("id", "") or "",
        "document": c.get("document") or "",
        "metadata": c.get("metadata") or {},
        "score": float(
            c.get("rerank_score", c.get("rrf_score", c.get("score", 0.0))) or 0.0
        ),
    }


def _format_sources(sources: List[dict]) -> tuple[List[str], List[str], List[float]]:
    """从 canonical sources 产出 contexts / context_ids / context_scores 三个并行列表.

    三者必须由**同一套过滤**产出 (strip 后为空则整条跳过), 否则 RAGAS 拿到的
    contexts 与 context_ids 会对不上。也正因为会跳过空文档, 三个列表的长度可能
    小于 ``len(sources)``。
    """
    contexts: List[str] = []
    ids: List[str] = []
    scores: List[float] = []
    for c in sources:
        content = (c.get("document") or "").strip()
        if not content:
            continue
        contexts.append(content)
        ids.append(c.get("id", "") or "")
        scores.append(float(c.get("score", 0.0) or 0.0))
    return contexts, ids, scores


async def retrieve_and_rerank(
    question: str,
    kb_id: str = "default",
    retriever: HybridRetriever | None = None,
    trace: Any = None,
    history: list[dict] | None = None,
) -> list[dict]:
    """共享的「检索 + 重排」链路, 返回重排后的候选 (含 metadata).

    抽出来是为了让 **API 主聊天链路和 run_rag 走完全同一套逻辑**。
    之前 `/api/v1/chat` 自己直连 retriever+reranker, 导致查询改写、
    RerankOptimizer 的候选池截断、父文档召回这些配置在生产链路上是死代码,
    只有评估脚本才吃得到, 线上线下行为不一致。

    Args:
        trace: 可选的 Langfuse trace; 传 None 时所有 span 自动退化为 no-op。
        history: **预留接口** —— 多轮对话历史, 元素形如
            ``{"role": "user"|"assistant", "content": str}``。本次只透传给
            QueryRewriter, 不参与任何计算; 传 None 与传 ``[]`` 行为完全一致。
    """
    tracer = get_tracer()
    # 走 get_retriever() 的进程内实例缓存: 直接 new HybridRetriever() 会重新打开
    # Chroma collection 并重新反序列化整个 BM25 pickle, 评估脚本 / agent 这类
    # 高频调用路径上这是纯粹的重复开销.
    if retriever is None:
        # 首次触达该 kb 时的构造是重 IO (BM25 pickle 全量读盘 + Chroma 引导),
        # 且期间持全局 _INSTANCE_LOCK —— 在事件循环线程同步调会卡住所有并发
        # 请求, 卸到线程池 (documents.py 的三个调用点同理).
        retriever = await asyncio.to_thread(get_retriever, kb_id)

    with tracer.span(trace, "query.rewrite"):
        # 查询改写: 用 LLM 把口语化/歧义问题扩写成多条检索 query;
        # 无 LLM (缺 key/测试) 时走确定性子句切分回退, 不降级.
        rewriter = QueryRewriter(enabled=settings.query_rewrite_enabled)
        try:
            # 注意: 这里必须传"文本适配器"而不是 BaseChatModel —— rewriter 内部按
            # `llm(prompt).strip()` 调用, 直接传 chat model 会拿到 AIMessage 并抛
            # AttributeError, 被 except 吞掉后永远退化成确定性切分回退.
            llm_for_rewrite = llm_router.get_llm_text_callable()
        except Exception as e:
            # 静默吞掉的话, key 配错时查询改写/HyDE 会永远走确定性回退: 接口全
            # 200、无告警, 属于"静默质量下降"类故障. 只告警一次, 不每请求刷日志.
            global _warned_no_llm_adapter
            if not _warned_no_llm_adapter:
                _warned_no_llm_adapter = True
                logger.warning(
                    f"查询改写/HyDE 的 LLM 适配器不可用, 之后各请求将走确定性回退: {e}"
                )
            llm_for_rewrite = None

        # 改写与 HyDE 都要调 LLM, 且都是**阻塞型同步 SDK** —— 必须卸到 threadpool,
        # 否则会在事件循环线程里干等 (真实 provider 一次 1~5s, 推理模型更久), 期间
        # 所有并发请求一起排队. 项目其他 LLM 调用点 (chat / run_rag / documents) 都
        # 已 offload, 这里曾是唯一的例外.
        # 两者互不依赖, 用 gather 并发发出: 这段延迟从"两次之和"压成"较慢的那次".
        hyp = None
        if settings.hyde_query_enabled and llm_for_rewrite is not None and (question or "").strip():
            queries, hyp = await asyncio.gather(
                asyncio.to_thread(rewriter.rewrite, question, llm=llm_for_rewrite, history=history),
                asyncio.to_thread(MetadataEnhancer.hyde_query, question, llm=llm_for_rewrite),
            )
        else:
            queries = await asyncio.to_thread(
                rewriter.rewrite, question, llm=llm_for_rewrite, history=history
            )


        # 空问题 / 纯标点会让 rewrite 返回 [], 兜底成原问题, 避免下面 per_query[0]
        # 直接 IndexError.
        if not queries:
            queries = [question] if (question or "").strip() else []

        # 查询侧 HyDE: 把问题扩写成一段"假设答案文档", 作为额外一路 query 参与召回.
        # 与 QueryRewriter 互补 —— 改写做的是词面扩展, HyDE 做的是语义空间对齐
        # (用户问句与文档陈述句式差异大时收益最明显). hyde_query 内部已吞异常并回退
        # 原问题, 这里只挡掉"没生成出东西", 避免白白多跑一路重复检索.
        if hyp and hyp.strip() and hyp.strip() != (question or "").strip():
            queries = queries + [hyp.strip()]

    with tracer.span(trace, "retriever.hybrid"):
        # 多 query 检索: 每个改写 query 各自走 BM25+向量+RRF+父文档召回,
        # 再把多路候选 RRF 融合, 扩大召回面 (query rewriting 的收益落点).
        # 把候选池预算按 query 数摊分, 避免多 query 把候选数放大过头.
        per_q_topk = max(settings.top_k_vector, settings.rerank_candidate_pool // max(1, len(queries)))
        # 并发而非串行: 各 query 之间没有任何依赖, 串行会让总延迟 = 各路之和
        # (改写出 3 条 query 就是 3 倍 embedding + 3 倍检索时间)。
        # 读路径可以安全并发 —— `add_documents` 的 `_write_lock` 只锁写,
        # BM25 的 `_ensure_index` 内部自带 `_rebuild_lock`, Chroma query 线程安全;
        # 每路又各自 to_thread 卸到线程池, 不会阻塞事件循环。
        per_query: list[list[dict]] = list(
            await asyncio.gather(
                *(retriever.retrieve_async(q, top_k=per_q_topk) for q in queries)
            )
        )
        if len(per_query) > 1:
            candidates = rrf_fuse(per_query, k=60)
        elif per_query:
            candidates = per_query[0]
        else:
            candidates = []

    with tracer.span(trace, "reranker.cross_encoder"):
        # 重排优化: 先把候选池截断到 Top-60 再让 Cross-Encoder 精排到 Top-25,
        # 压住 Cross-Encoder 算力 (11.4s -> 2.3s) 同时保住 Recall@5 (93% -> 98.7%).
        optimizer = RerankOptimizer()
        reranked = await optimizer.optimize_async(
            question,
            candidates,
            pool=settings.rerank_candidate_pool,
            top_n=settings.rerank_top_n,
            reranker=get_reranker(),
        )
    return reranked


def _record_generation(trace: Any, prompt: str, answer: str) -> None:
    """在 trace 上挂一次 generation (未启用 langfuse 时是 no-op). 失败不影响主流程."""
    try:
        get_tracer().generation(
            trace,
            name="answer.generate",
            model=settings.default_llm_model,
            prompt=prompt,
            completion=answer,
        )
    except Exception:
        pass


async def _generate_events(
    prompt: str,
    *,
    stream_tokens: bool,
    trace: Any,
    emit: Callable[[RagEvent], None],
) -> str:
    """跑一次生成, 可见 token 经 ``emit`` 推出, 返回最终 answer.

    **三条链路唯一的生成入口** (rag 节点 / tools 节点 / direct 节点), 所以
    ThinkBlockFilter、strip_think、卸载到线程池的方式只有一份 —— "同一件事写两份"
    在这个项目里已经出过两次事故 (提示词分叉、上下文截断长度 300 vs 600)。

    刻意是**协程而不是异步生成器**: 它返回 answer, 事件通过 emit 出去。若写成
    async generator, "emit 出去的 token" 与"返回的 answer"之间的先后就要靠消费
    时机来保证, 很容易出现 answer 已经落库、token 还在往外流的错位。

    非流式分支不发任何 TokenEvent —— 是否流式由调用方决定, 与 answer 无关。
    """
    tracer = get_tracer()
    with tracer.span(trace, "llm.generate", model=settings.default_llm_model):
        if stream_tokens:
            # 逐 token 过滤 <think>: 推理模型的思考过程不能流到前端, 而标签可能
            # 跨 chunk 切开, 所以必须用有状态过滤器, 不能退化成 strip_think.
            think = ThinkBlockFilter()
            parts: List[str] = []
            async for token in llm_router.get_llm().astream(prompt):
                raw = getattr(token, "content", token) if token else ""
                if not raw:
                    continue
                text = think.feed(raw if isinstance(raw, str) else str(raw))
                if text:
                    parts.append(text)
                    emit(TokenEvent(text))
            tail = think.flush()
            if tail:
                parts.append(tail)
                emit(TokenEvent(tail))
            answer = "".join(parts)
        else:
            # LLM 是阻塞同步 SDK, 卸到 threadpool.
            resp = await asyncio.to_thread(llm_router.get_llm().invoke, prompt)
            answer = strip_think(getattr(resp, "content", "") or str(resp))

    _record_generation(trace, prompt, answer)
    return answer


async def _retrieve_sources(
    question: str,
    kb_id: str,
    top_k: int,
    trace: Any,
    history: list[dict] | None = None,
) -> tuple[List[dict], List[str]]:
    """检索 + 归一 + 截断, 返回 ``(canonical sources, 纯文本 contexts)``.

    两个返回值必须由**同一次**处理产出: ``sources`` 是下发给前端做引用溯源的列表,
    ``contexts`` 是喂进提示词的正文. 后者会被 _format_sources 跳过空文档, 所以可能
    比 sources 短 —— 两者由同一个函数一起返回, 免得调用方各自过滤一遍造成错位.

    检索链只此一个入口: 图的 rag 节点走这里 (rag 模式与 agent 模式共用同一个节点),
    保证改写 / HyDE / 候选池截断 / 父文档召回这些配置在两条路径上一致.
    """
    # 不显式传 retriever: retrieve_and_rerank 内部会走 get_retriever(kb_id) 的进程内
    # 实例缓存. agent 侧也复用它, 两条路的 retriever 是同一个.
    reranked = await retrieve_and_rerank(
        question, kb_id=kb_id, trace=trace, history=history
    )
    sources = [normalize_source(c) for c in reranked[:top_k]]
    contexts, _, _ = _format_sources(sources)
    return sources, contexts


@dataclass(frozen=True)
class _GraphDone:
    """内部信号: 图跑完了, 带上最终 state (不进入对外的 RagEvent 协议)."""

    state: dict


@dataclass(frozen=True)
class _GraphError:
    """内部信号: 图里抛了异常, 原样转交给消费方."""

    exc: BaseException


def _swallow_task_exc(t: "asyncio.Task") -> None:
    """吃掉 task 上残留的异常, 避免 "Task exception was never retrieved" 噪音.

    图内部的异常已经由 _run_graph 经队列转交给消费方了, 这里只需静默; 取消则
    完全忽略 (t.exception() 在已取消的 task 上会抛 CancelledError).
    """
    if not t.cancelled():
        t.exception()


def _make_emit(q: "asyncio.Queue[Any]", *, emit_trace: bool) -> Callable[[RagEvent], None]:
    """把节点推出的事件转进队列; rag 模式下丢掉 TraceEvent.

    为什么在**发射层**按 mode 过滤, 而不是让节点自己判断: 节点里出现
    ``if mode == "agent"`` 会让同一份代码走两条路, 而 rag 模式的线格式是一份
    跨仓库契约 (前端按帧类型消费, tests/test_chat_sse_contract.py 钉着"rag 模式
    不得出现 trace 帧")。过滤集中在这里, 节点只需无条件 emit —— 而且 trace 照常
    写进 state, 排查 rag 模式时一样拿得到, 只是不下发给客户端。
    """

    def emit(ev: RagEvent) -> None:
        if isinstance(ev, TraceEvent) and not emit_trace:
            return
        q.put_nowait(ev)

    return emit


async def _run_graph(graph, state_in, config: dict, q: "asyncio.Queue[Any]") -> None:
    """跑图, 把结果/异常**经队列**送出去.

    异常走队列而不是留给 task, 是为了保住顺序: 已经 emit 出去的 source / token 必须
    排在 error 之前被消费方看到。若让异常从 task 里冒出来, 消费方可能先拿到异常、
    丢掉队列里还没取出的事件 —— 表现就是"报错了, 但引用来源莫名消失"。
    """
    try:
        result = await graph.ainvoke(state_in, config)
    except asyncio.CancelledError:
        raise
    except BaseException as e:  # noqa: BLE001 - 原样转交, 由消费方决定怎么处置
        q.put_nowait(_GraphError(e))
    else:
        q.put_nowait(_GraphDone(result))


async def _graph_events(
    question: str,
    kb_id: str,
    top_k: int,
    stream_tokens: bool,
    trace: Any,
    use_agent: bool,
    history: list[dict] | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> AsyncIterator[RagEvent]:
    """唯一的编排: 跑图, 把节点推出的事件 + 最终的 DoneEvent 转给消费方.

    **事件所有权** (改这条链路前先读):
      - ``SourcesEvent`` / ``TokenEvent`` / ``TraceEvent`` 由**节点** emit, 这里原样转发;
      - ``DoneEvent`` 只在这里构造一次, 节点里不许造 —— 这样"恰好一次"是结构性保证,
        不是靠纪律维持;
      - 图里的异常经队列转交 (见 _run_graph), 保证已经发出的事件排在 error 之前。

    rag 模式与 agent 模式走的是同一张图, 差别只在入口条件边 (rag 跳过路由节点) 与
    事件过滤 (rag 不发 TraceEvent)。这样"检索 + 提示词 + 生成"只有一份实现。

    **多轮记忆**: 请求带 session_id 且 ``agent_memory_enabled`` 时, 取带
    checkpointer 的那份编译产物, thread_id = "{user}:{session}" —— 跨用户/跨会话
    靠它隔离; 此时初始 state 的 history 必须置 None (历史已在 checkpoint 里,
    再传会被 add reducer 整段重放)。无 session 或关闭记忆时与单轮行为完全一致。
    ``history`` 仅在无记忆模式下生效, 逐层透传给检索链的改写器。
    """
    # 关键: 必须是**函数内** import. 测试用直接赋值 graph_mod.build_agent_graph
    # 的方式替换整张图 (tests/test_integration_retrieval.py 的 agent 用例),
    # 提到模块顶层会让那个替换失效 —— 替换的是 graph 模块的属性, 而顶层 import
    # 早就把原函数绑死了.
    from app.agents.graph import build_agent_graph, new_agent_state

    mode = "agent" if use_agent else "rag"
    thread_id = f"{user_id or 'anon'}:{session_id}" if session_id else None
    use_memory = bool(getattr(settings, "agent_memory_enabled", False)) and thread_id is not None
    graph = build_agent_graph(use_memory=use_memory)
    q: "asyncio.Queue[Any]" = asyncio.Queue()
    state_in = new_agent_state(
        question,
        kb_id,
        mode=mode,
        history=None if use_memory else history,
        engine=getattr(settings, "agent_engine", "react") or "react",
    )
    configurable: dict = {
        "emit": _make_emit(q, emit_trace=use_agent),
        "stream_tokens": stream_tokens,
        "trace": trace,
        "top_k": top_k,
        "history": history,
    }
    if use_memory:
        configurable["thread_id"] = thread_id
    config = {"configurable": configurable}
    # 队列/生产者-消费者结构不能省, 即便非流式也不行: SourcesEvent 必须在生成失败
    # **之前**送达消费方, 否则 run_rag 的 got_sources 降级会从"退回上下文拼接"
    # 变成"直接抛", 而那是没有任何测试覆盖的静默行为回归。
    task = asyncio.create_task(_run_graph(graph, state_in, config, q))
    try:
        while True:
            item = await q.get()
            if isinstance(item, _GraphError):
                logger.exception(f"graph error: {item.exc}")
                raise item.exc
            if isinstance(item, _GraphDone):
                answer = strip_think(item.state.get("final") or "")
                if use_agent and not answer:
                    # 空答案是合法结果 (模型真的返回空), 所以只能 warning 不能 raise;
                    # 但"终端节点忘了写 state['final']"也长这样, 这条日志是唯一信号.
                    logger.warning(
                        "图跑完但 state['final'] 为空 —— 检查终端节点是否写了 final"
                    )
                yield DoneEvent(
                    answer=answer,
                    agent_trace=list(item.state.get("trace") or []) if use_agent else [],
                )
                return
            yield item
    finally:
        # 客户端断开 (GeneratorExit) / 出错 / 正常结束都会走到: 取消图任务, 让
        # CancelledError 穿透到节点内部的 await, 生成器当场关闭 —— 不再为已经没人
        # 要的结果继续烧 LLM。这里**不 await**: aclose 期间 await 可能挂住关闭流程,
        # 用 done callback 静默收尾即可。
        if not task.done():
            task.cancel()
        task.add_done_callback(_swallow_task_exc)


async def stream_rag(
    question: str,
    kb_id: str = "default",
    top_k: int | None = None,
    use_agent: bool = False,
    stream_tokens: bool = False,
    user_id: str | None = None,
    session_id: str | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[RagEvent]:
    """**唯一的** RAG 编排. 两种 mode 走同一张图, 由入口条件边分流.

    事件序列 (两种 mode 相同)::

        SourcesEvent -> (TraceEvent*) -> (TokenEvent*) -> DoneEvent

    - ``use_agent=False``: 跳过路由节点直接检索 (省一次 LLM 往返); 不发 TraceEvent
    - ``use_agent=True``: 先由路由节点判断要不要检索 / 用不用工具, 再分流到
      检索 / 工具 / 直答三个终端节点; trace 逐条下发
    - ``stream_tokens=False``: 不产出 TokenEvent, 消费方直接读 ``DoneEvent.answer``;
      ``True`` 时逐 token 过 ThinkBlockFilter, 三种终端节点行为一致
    - **异常一律向外抛**, 不在这里兜底: run_rag 需要"生成失败就退回上下文拼接"
      的降级, 而 API 层需要把它变成一个明确的 500 / error 帧 —— 怎么处置是
      消费方的决定, 编排层不该替它们选.
    - ``history`` 是多轮历史 (元素 ``{"role", "content"}``), **仅在无记忆模式下**
      生效, 逐层透传到检索链的改写器; 带 session_id 且开启记忆时, 历史由
      checkpointer 按 thread_id 管理, 传了也会被忽略 (见 _graph_events).
    """
    top_k = top_k or settings.top_k_final
    tracer = get_tracer()
    # 注意 `as trace`: tracer.trace() 是 generator-based context manager, 不接住
    # yield 出来的那个对象的话, 后面 tracer.span(trace, ...) 收到的是
    # _GeneratorContextManager (没有 .span) —— 会被 span() 内部的 except 吞掉、
    # 静默退化成 noop span, 也就是 langfuse 开着也什么都看不到.
    with tracer.trace(
        name="rag.run",
        metadata={
            "kb_id": kb_id,
            "mode": "agent" if use_agent else "rag",
            "user_id": user_id or "anon",
        },
        user_id=user_id or "anon",
        session_id=session_id,
    ) as trace:
        events = _graph_events(
            question,
            kb_id,
            top_k,
            stream_tokens,
            trace,
            use_agent,
            history,
            user_id=user_id,
            session_id=session_id,
        )
        async for ev in events:
            yield ev


async def run_rag(
    question: str,
    kb_id: str = "default",
    top_k: int | None = None,
    use_agent: bool = False,
    user_id: str | None = None,
    session_id: str | None = None,
) -> dict:
    """评估 / 脚本入口: 把 stream_rag 的事件聚合成一个 dict.

    Returns:
        dict 形如 {answer, contexts, context_ids, context_scores, metadata}
        与 RAGAS 期待的字段对齐(question/answer/contexts/ground_truth)。

    降级策略: **生成阶段**失败时退回上下文拼接 —— 评估脚本不该因为一次 LLM
    抖动就整轮失败。**检索阶段**失败仍然向上抛, 否则评估会因为"检索坏了"而
    拿到一堆看似正常的垃圾答案, 指标反而更好看。
    """
    top_k = top_k or settings.top_k_final
    sources: List[dict] = []
    answer = ""
    agent_trace: List[dict] = []
    got_sources = False
    try:
        async for ev in stream_rag(
            question,
            kb_id=kb_id,
            top_k=top_k,
            use_agent=use_agent,
            stream_tokens=False,
            user_id=user_id,
            session_id=session_id,
        ):
            if isinstance(ev, SourcesEvent):
                got_sources = True
                sources = ev.sources
            elif isinstance(ev, DoneEvent):
                answer = ev.answer
                agent_trace = ev.agent_trace
    except Exception as e:
        if not got_sources:
            raise
        logger.warning(
            f"LLM invocation failed during RAG: {e}; falling back to concatenated contexts."
        )

    contexts, ids, scores = _format_sources(sources)
    if not answer:
        answer = "\n\n".join(contexts) if contexts else "未找到相关信息"

    metadata = (
        {"mode": "agent", "trace": agent_trace}
        if use_agent
        else {"mode": "rag", "kb_id": kb_id}
    )
    return {
        "answer": answer,
        "contexts": contexts,
        "context_ids": ids,
        "context_scores": scores,
        "metadata": metadata,
    }
