"""Cross-Encoder 重排序，抑制幻觉。

两个后端, 由 `settings.rerank_backend` 选择:
  - `local`: sentence-transformers 本地加载 CrossEncoder (默认)
  - `api`:   Cohere 风格的 /rerank HTTP 接口 (如硅基流动的 BAAI/bge-reranker-v2-m3)

两者对外暴露**完全相同**的 `rerank(query, candidates, top_k)` 契约, 所以
RerankOptimizer / pipeline / agents.tools 三条调用路径一行都不用改。
"""
from __future__ import annotations

import asyncio
import os
import threading
import time
from functools import lru_cache
from typing import List, Optional

import httpx
from loguru import logger

from app.core.config import settings


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or settings.rerank_model
        self._model = None
        # 同 embedding: 冷启动并发会重复加载 CrossEncoder. double-checked
        # locking 保证单例只初始化一次.
        self._lock = threading.Lock()

    def _ensure(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import CrossEncoder
                    logger.info(f"Loading rerank model: {self.model_name}")
                    self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """candidates: [{id, document, metadata, score|distance}]"""
        if not candidates:
            return []
        try:
            m = self._ensure()
            pairs = [(query, c.get("document", "")) for c in candidates]
            scores = m.predict(pairs, show_progress_bar=False)
            for c, s in zip(candidates, scores):
                c["rerank_score"] = float(s)
            ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
            return ranked[:top_k]
        except Exception as e:
            logger.warning(f"Rerank failed, fallback to original order: {e}")
            return candidates[:top_k]


class APIReranker:
    """走 HTTP 的 rerank 后端 (默认硅基流动 `BAAI/bge-reranker-v2-m3`).

    与 `CrossEncoderReranker` 的差别: 不加载本地模型 —— 不需要 torch /
    sentence-transformers, 不占磁盘与显存, 冷启动也没有模型加载等待. 代价是
    每次检索多一次网络往返、按量计费, 且强依赖服务可用性.

    接口格式 (Cohere / Jina / 硅基流动共用)::

        POST {base_url}/rerank
        {"model", "query", "documents": [...], "top_n", "return_documents": false}
        -> {"results": [{"index": 0, "relevance_score": 0.98}, ...]}

    换其它 provider 通常只需改 `rerank_api_base`; 若某家字段名不同 (例如返回
    `score` 而非 `relevance_score`), 改 `_parse` 一处即可。

    两处与本地版**不同但需要注意**的地方:
      1) **分数量纲不同**: 本地 CrossEncoder 输出原始 logits (可负、无上界), 服务端
         返回归一化到 0~1 的 relevance_score. 两者都只参与**同一批候选内部**的相对
         排序, 所以不影响检索结果; 但不要跨后端比对分数绝对值, 前端展示的含义也变了.
      2) **失败会降级**: 重试耗尽后返回原始顺序 (与本地版行为一致), 但必须打 warning
         —— 降级一旦静默, 就表现为"检索质量莫名下降", 这类问题在本项目里出现过多次.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        max_chars: int = 2000,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_chars = max_chars
        self.max_retries = max(1, max_retries)
        # httpx.Client 线程安全且自带连接池: 每次检索都会调一次 rerank, 复用连接
        # 可以把 TLS 握手开销摊掉. 实例本身由 get_reranker() 做进程内单例.
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=timeout,
        )

    def rerank(self, query: str, candidates: List[dict], top_k: int = 5) -> List[dict]:
        """契约与 `CrossEncoderReranker.rerank` 完全一致.

        candidates: [{id, document, metadata, score|distance}]
        """
        if not candidates:
            return []
        # 截断与本地版对齐: 本地 CrossEncoder 也会按模型 max_length 截断
        # (bge-reranker-base 是 512 token). 不截断则长 chunk 按 token 全额计费.
        documents = [(c.get("document") or "")[: self.max_chars] for c in candidates]
        try:
            scores = self._score(query, documents)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"API rerank failed, fallback to original order: {e}")
            return candidates[:top_k]
        for c, s in zip(candidates, scores):
            c["rerank_score"] = s
        ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return ranked[:top_k]

    def _score(self, query: str, documents: List[str]) -> List[float]:
        """调用一次 /rerank, 返回与 documents 等长且**按下标对齐**的分数."""
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            # 显式要回全部候选的分数. 不给 top_n 时服务端可能只返回前若干条,
            # 缺分数的候选会退化成 0 分排到末尾 —— 那是静默的排序错误.
            "top_n": len(documents),
            # 只要分数不要原文: 原文我们手上已有, 回传纯属浪费带宽.
            "return_documents": False,
        }
        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries):
            retryable = True
            try:
                resp = self._client.post("/rerank", json=payload)
                if resp.status_code == 429 or resp.status_code >= 500:
                    # 限流 / 服务端故障: 值得退避后重试.
                    raise httpx.HTTPStatusError(
                        f"rerank retryable status {resp.status_code}",
                        request=resp.request,
                        response=resp,
                    )
                try:
                    resp.raise_for_status()
                except httpx.HTTPStatusError:
                    # 其余 4xx 是请求本身的问题 (模型名不存在 / key 失效 / 参数不合法),
                    # 重试只会把同一个错误再犯一遍 —— 立刻放弃重试, 交给上层降级.
                    retryable = False
                    raise
                return self._parse(resp.json(), len(documents))
            except Exception as e:  # noqa: BLE001
                last_err = e
                if not retryable or attempt + 1 >= self.max_retries:
                    break
                # 指数退避 0.5s / 1s / 2s …, 上限 4s.
                time.sleep(min(0.5 * (2 ** attempt), 4.0))
        if last_err is not None:
            raise last_err
        raise RuntimeError("rerank failed without exception")

    @staticmethod
    def _parse(data: dict, n: int) -> List[float]:
        """把响应还原成与输入等长、按下标对齐的分数列表.

        必须按 `index` 回填, **不能按顺序 zip**: 服务端返回的 `results` 是**按分数
        降序**排列的, 顺序早已不等于输入顺序. 顺序 zip 会把高分挂到错误的候选上,
        而且不抛任何异常 —— 表现为"重排后结果莫名其妙", 与 `embed_texts` 错位
        是同一类失败 (见 retriever.add_documents 的长度校验).
        """
        scores = [0.0] * n
        for item in ((data or {}).get("results") or []):
            idx = item.get("index")
            if not isinstance(idx, int) or not (0 <= idx < n):
                continue
            raw = item.get("relevance_score", item.get("score"))
            if raw is None:
                continue
            scores[idx] = float(raw)
        return scores


@lru_cache(maxsize=4)
def get_reranker(model_name: Optional[str] = None) -> "CrossEncoderReranker | APIReranker":
    """获取 rerank 单例, 后端由 `settings.rerank_backend` 决定.

    `model_name` 是缓存 key 的一部分, 换模型才会新建实例; 同一模型下始终
    复用同一对象 —— local 后端避免每次请求都从磁盘/HuggingFace 重新加载模型
    (第一次调用会触发 `CrossEncoderReranker._ensure()`, 之后 hot path 就是
    普通的 `m.predict` forward); api 后端则避免每次请求都重建连接池.

    注意: 后端类型 / base_url / api_key 等**不在**缓存 key 里 (它们在函数体内
    读 settings). 运行时改这些配置需要调 `reset_reranker_cache()` 让下一次调用
    重建 —— 与 `get_embedding_service` 的取舍一致, 测试里已有此用法.

    类似 `embedding.get_embedding_service` 和 `retriever.get_retriever`,
    整个 codebase 都用 module-level 单例.
    """
    backend = getattr(settings, "rerank_backend", "local")
    if backend == "api":
        api_key = settings.rerank_api_key or os.environ.get("RERANK_API_KEY", "")
        if not api_key:
            # 明确抛错而不是悄悄退回本地模型: 两个后端的模型与分数量纲都不同
            # (本地 logits vs 服务端 0~1), 混着用会让排序口径在请求之间漂移 ——
            # 静默降级比直接失败危险得多 (与 embedding 侧同一取舍).
            raise RuntimeError(
                "rerank_backend=api 但没拿到 API key: 请设置环境变量 RERANK_API_KEY"
                "(或在项目根目录 .env 里配置 RERANK_API_KEY)"
            )
        logger.info(
            f"Using APIReranker: model={settings.rerank_api_model} "
            f"base={settings.rerank_api_base}"
        )
        return APIReranker(
            model=model_name or settings.rerank_api_model,
            base_url=settings.rerank_api_base,
            api_key=api_key,
            max_chars=settings.rerank_api_max_chars,
            timeout=settings.rerank_api_timeout,
            max_retries=settings.rerank_api_max_retries,
        )
    return CrossEncoderReranker(model_name)


def reset_reranker_cache() -> None:
    """仅供测试: 清空单例缓存."""
    get_reranker.cache_clear()


class RerankOptimizer:
    """重排候选池优化 (Top-K 候选 -> Top-N 精排).

    Cross-Encoder 是 RAG 链路里最贵的环节 (逐对 (query, doc) 前向).
    若把全部召回 (可能上百) 都丢给重排, 延迟随候选数线性爆炸.
    工程经验: 先用 RRF 分数**截断候选池到 Top-60**, 再让 Cross-Encoder
    精排到 **Top-25**, 既保住召回上限 (Recall@5 93% -> 98.7%), 又把
    Cross-Encoder 算力压住 (实测 11.4s -> 2.3s 量级).

    设计:
      - `optimize(query, candidates, pool, top_n)`:
        1) 按已有分数 (rrf_score 或 rerank_score) 降序取前 `pool` 个;
        2) 交给 CrossEncoderReranker 精排到 `top_n`.
      - `candidates` 为空直接返回空, 不触发模型加载.
    """

    def __init__(self, reranker: Optional[CrossEncoderReranker] = None):
        self._reranker = reranker

    def _score(self, c: dict) -> float:
        return float(c.get("rerank_score", c.get("rrf_score", 0.0)) or 0.0)

    def optimize(
        self,
        query: str,
        candidates: List[dict],
        pool: int = 60,
        top_n: int = 25,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> List[dict]:
        if not candidates:
            return []
        rk = reranker or self._reranker or get_reranker()
        # 1) RRF 截断候选池 (控制 Cross-Encoder 输入规模).
        pooled = sorted(candidates, key=self._score, reverse=True)[:pool]
        # 2) Cross-Encoder 精排到 top_n.
        return rk.rerank(query, pooled, top_k=top_n)

    async def optimize_async(
        self,
        query: str,
        candidates: List[dict],
        pool: int = 60,
        top_n: int = 25,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> List[dict]:
        return await asyncio.to_thread(
            self.optimize, query, candidates, pool, top_n, reranker
        )
