"""Tools available to agents: retrieval + doc reading + calculator + current_date.

两个层次, 刻意分开:
- **schema 层** (``@tool`` + docstring): 只为 ``bind_tools`` 提供模型可见的
  名字 / 参数 / 说明;
- **执行层** (``HANDLERS``): react 引擎的 act 节点按名字调用, 返回
  ``(observation, sources_delta)`` 二元组 —— observation 是给模型看的文本,
  sources_delta 是结构化候选 (id/document/metadata/score), 由 act 节点合并进
  ``state["sources"]`` 做引用溯源。

为什么执行不走 ``tool.invoke``: @tool 的返回值是拼好的文本, chunk_id / metadata
在那一层已经丢了 —— Agent 模式 sources 恒空的那次教训 (见 retrieve_hits)。
两个层次共用同一组底层函数, schema 与 handler 不会各自漂移。
"""
from __future__ import annotations
from typing import Callable, Dict, List, Tuple

from langchain_core.tools import tool

from app.core.config import settings
from app.rag.reranker import get_reranker
from app.rag.retriever import get_retriever
from app.rag.vector_store import aggregate_documents
from app.agents.safe_eval import safe_eval


def retrieve_hits(kb_id: str, query: str, top_k: int = 5) -> List[dict]:
    """检索 + 精排, 返回**结构化**候选.

    抽成独立函数是为了让 retriever_node 同时拿到两样东西: 喂 LLM 的文本, 以及
    构造引用来源所需的结构化字段. 原先只有工具返回的拼好字符串, chunk_id /
    doc_id / metadata 在中途就丢了 —— 这正是 Agent 模式的 sources 一直是空列表
    的原因 (前端因此无法溯源).

    与 API 的 RAG 主链路共用 retriever 单例, 确保 Chroma / NumPy 后端切换后
    Agent 查的仍是同一份知识库; reranker 同样走单例, 避免每个请求重载模型.

    刻意**不走** pipeline.retrieve_and_rerank (改写/HyDE/多 query): react 引擎
    的分工就是让模型自己当改写器 —— 它看到第一轮结果不理想会换词重试, 预处理
    改写在这里是重复付费。
    """
    retriever = get_retriever(kb_id)
    reranker = get_reranker()
    candidates = retriever.retrieve(query, top_k=max(10, top_k * 2))
    return reranker.rerank(query, candidates, top_k=top_k)


def format_hits(hits: List[dict]) -> str:
    """把候选拼成喂给 LLM 的文本 (编号 + 重排分 + 截断正文).

    截断上限走配置而不是写死 300: chunk 后半段 (定价、结论等常在文末) 否则会在
    Agent 链路被静默丢掉. 与 RAG 主链路共用同一个配置, 两条路径看到的上下文
    一样长 —— 否则会出现"同一份文档, RAG 答得出来、Agent 答不出来".

    注意: 这里的 [n] 编号只在**单次调用内**有效; compose 节点会用全局合并后的
    sources 重新编号, 模型侧的编号不会泄漏到最终答案。
    """
    limit = getattr(settings, "context_snippet_chars", 2000)
    lines = []
    for i, r in enumerate(hits, 1):
        doc = r.get("document", "")
        snippet = doc[:limit].replace("\n", " ")
        score = r.get("rerank_score", 0)
        lines.append("[" + str(i) + "] (score=" + format(score, ".3f") + ") " + snippet)
    return "\n".join(lines)


def merge_sources(existing: List[dict], hits: List[dict]) -> List[dict]:
    """跨轮合并候选: 按 chunk id 去重, **首次出现顺序**保留, 分数取 max.

    [n] 编号 = 首次出现顺序, compose 的提示词编号与前端引用溯源都以这个顺序
    为准 —— 多轮检索之间一旦重排, [n] 与 sources 就错位, 引用溯源整条链断掉。
    score 取 max 只影响展示, 不影响顺序。
    """
    out = list(existing)
    seen = {c.get("id") for c in out}
    for h in hits:
        cid = h.get("id")
        if not cid or cid in seen:
            continue
        seen.add(cid)
        # canonical 形状与 pipeline.normalize_source 一致, 但这里不能 import
        # pipeline (会经 app.agents.__init__ 绕回成环), 就地写这三个字段。
        out.append(
            {
                "id": cid,
                "document": h.get("document") or "",
                "metadata": h.get("metadata") or {},
                "score": float(h.get("rerank_score", h.get("score", 0.0)) or 0.0),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 执行层: handler(args: dict, kb_id: str) -> (observation, sources_delta)
# ---------------------------------------------------------------------------


def _kb_chunks(kb_id: str) -> List[dict]:
    """取 kb 全部 chunk 的内存态 (BM25 docs, Fake/真实后端同形状 {id,text,metadata}).

    不走 vector_store: read_chunk / read_parent 要按 id / chunk_index 随机访问,
    BM25 与向量库的一致性另有启动自检盯着 (retriever.check_store_consistency)。
    """
    return get_retriever(kb_id).bm25_store.docs


def _canonical(d: dict) -> dict:
    """BM25 内存 doc -> canonical source 形状 (检索工具没有分数, 记 0.0)."""
    return {
        "id": d.get("id", ""),
        "document": d.get("text", "") or "",
        "metadata": d.get("metadata") or {},
        "score": 0.0,
    }


def _h_hybrid_search(args: dict, kb_id: str) -> Tuple[str, List[dict]]:
    query = str(args.get("query") or "").strip()
    if not query:
        return "query 为空, 请给出检索词。", []
    top_k = args.get("top_k") or settings.top_k_final
    try:
        top_k = max(1, min(int(top_k), 20))
    except (TypeError, ValueError):
        top_k = settings.top_k_final
    hits = retrieve_hits(kb_id, query, top_k=top_k)
    if not hits:
        return "未检索到相关文档。可尝试更换关键词。", []
    return format_hits(hits), hits


def _h_read_chunk(args: dict, kb_id: str) -> Tuple[str, List[dict]]:
    chunk_id = str(args.get("chunk_id") or "").strip()
    for d in _kb_chunks(kb_id):
        if d.get("id") == chunk_id:
            return d.get("text", "") or "", [_canonical(d)]
    return f"chunk 不存在: {chunk_id}", []


def _h_read_parent(args: dict, kb_id: str) -> Tuple[str, List[dict]]:
    """取 chunk 及其在原文档中紧邻的前后各一个 chunk.

    只连坐 ±1 而不是整篇读回: 整篇会击穿上下文预算, 而 ±1 与
    expand_parent_recall 的口径一致, 两套"父召回"行为可对照。
    """
    chunk_id = str(args.get("chunk_id") or "").strip()
    docs = _kb_chunks(kb_id)
    center = next((d for d in docs if d.get("id") == chunk_id), None)
    if center is None:
        return f"chunk 不存在: {chunk_id}", []
    md = center.get("metadata") or {}
    doc_id, idx = md.get("doc_id"), md.get("chunk_index")
    picked = [center]
    if doc_id and isinstance(idx, int):
        by_index = {}
        for d in docs:
            m = d.get("metadata") or {}
            if m.get("doc_id") == doc_id and isinstance(m.get("chunk_index"), int):
                by_index[m["chunk_index"]] = d
        for i in (idx - 1, idx + 1):
            if i in by_index:
                picked.append(by_index[i])
    picked.sort(key=lambda d: (d.get("metadata") or {}).get("chunk_index") or 0)
    limit = getattr(settings, "context_snippet_chars", 2000)
    text = "\n---\n".join((d.get("text", "") or "")[:limit] for d in picked)
    return text, [_canonical(d) for d in picked]


def _h_list_documents(args: dict, kb_id: str) -> Tuple[str, List[dict]]:
    docs = aggregate_documents([d.get("metadata") or {} for d in _kb_chunks(kb_id)])
    if not docs:
        return "知识库为空。", []
    lines = [
        f"- {d['doc_id']}: {d['filename'] or '(未命名)'} ({d['chunks']} chunks)"
        for d in docs
    ]
    return "\n".join(lines), []


def _h_calculator(args: dict, kb_id: str) -> Tuple[str, List[dict]]:
    # SafeEvalError 与其它异常一律转文本: 工具失败不终止循环, 让模型看到错误
    # 自己修正表达式 (与旧 calculator @tool 的行为一致)。
    try:
        return str(safe_eval(str(args.get("expression") or ""))), []
    except Exception as e:  # noqa: BLE001
        return "计算失败: " + str(e), []


def _h_current_date(args: dict, kb_id: str) -> Tuple[str, List[dict]]:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S"), []


HANDLERS: Dict[str, Callable[[dict, str], Tuple[str, List[dict]]]] = {
    "hybrid_search": _h_hybrid_search,
    "read_chunk": _h_read_chunk,
    "read_parent": _h_read_parent,
    "list_documents": _h_list_documents,
    "calculator": _h_calculator,
    "current_date": _h_current_date,
}


def build_tools(kb_id="default"):
    """构造 bind_tools 用的 schema 列表. 实际执行一律走 HANDLERS (见模块 docstring)."""

    @tool
    def hybrid_search(query: str, top_k: int = 5) -> str:
        "从企业知识库混合检索（BM25+向量+RRF 重排）。自行组织检索词, 不要照抄用户原话（口语/指代需展开）；结果不相关时换关键词重试。返回 [n] (score=..) 正文 格式的候选。"
        obs, _ = HANDLERS["hybrid_search"]({"query": query, "top_k": top_k}, kb_id)
        return obs

    @tool
    def read_chunk(chunk_id: str) -> str:
        "按 chunk_id 读取一条检索结果的完整原文。"
        obs, _ = HANDLERS["read_chunk"]({"chunk_id": chunk_id}, kb_id)
        return obs

    @tool
    def read_parent(chunk_id: str) -> str:
        "读取某 chunk 及其在原文档中紧邻的前后各一个 chunk, 补全被切断的上下文。"
        obs, _ = HANDLERS["read_parent"]({"chunk_id": chunk_id}, kb_id)
        return obs

    @tool
    def list_documents() -> str:
        "列出当前知识库的全部文档 (doc_id / 文件名 / chunk 数), 适合先了解库里有什么再决定怎么检索。"
        obs, _ = HANDLERS["list_documents"]({}, kb_id)
        return obs

    @tool
    def calculator(expression: str) -> str:
        "安全计算数学表达式: 支持 + - * / // % **, 一元正负号, math 模块常用函数."
        obs, _ = HANDLERS["calculator"]({"expression": expression}, kb_id)
        return obs

    @tool
    def current_date() -> str:
        "获取当前日期时间。"
        obs, _ = HANDLERS["current_date"]({}, kb_id)
        return obs

    return [hybrid_search, read_chunk, read_parent, list_documents, calculator, current_date]
