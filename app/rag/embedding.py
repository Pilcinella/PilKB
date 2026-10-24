"""Embedding service: 本地 sentence-transformers (默认) / OpenAI 兼容 API / fake.

后端由 `settings.embedding_backend` 选择, 见 get_embedding_service().
"""
from __future__ import annotations

import hashlib
import os
import threading
from functools import lru_cache
from typing import List

from loguru import logger

from app.core.config import settings


class EmbeddingService:
    """维度：bge-small-zh-v1.5 -> 512, bge-base-zh-v1.5 -> 768, OpenAI text-embedding-3-small -> 1536"""

    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.embedding_model
        self.device = device or settings.embedding_device
        self._model = None
        # 冷启动时并发请求会同时进入 _ensure_model(): 每个线程都各自
        # SentenceTransformer(...) 一遍, 轻则重复占几百 MB 显存/内存, 重则撞上
        # torch/tf 初始化本身不是线程安全的. 用 double-checked locking 保证
        # 模型只加载一次, 加载完成后 hot path 是一次无竞争的属性读.
        self._lock = threading.Lock()
        self.dim = 512  # default for bge-small

    def _ensure_model(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading embedding model: {self.model_name} on {self.device}")
                    self._model = SentenceTransformer(self.model_name, device=self.device)
                    self.dim = self._model.get_sentence_embedding_dimension()
        return self._model

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        m = self._ensure_model()
        vecs = m.encode(texts, normalize_embeddings=True, show_progress_bar=False, convert_to_numpy=True)
        return vecs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed([text])[0]


class FakeEmbeddingService:
    """deterministic embedding: 字符哈希到 64 维, 不依赖 torch / sentence-transformers.

    用在 CI / 沙箱 / 评估脚本验证链路通断. 行为类似 bge-small, 但语义质量远不如真模型.
    """

    dim = 64

    def _vec(self, text: str) -> List[float]:
        v = [0.0] * self.dim
        for ch in text:
            idx = int(hashlib.md5(ch.encode()).hexdigest(), 16) % self.dim
            v[idx] += 1.0
        norm = sum(x * x for x in v) ** 0.5 or 1.0
        return [x / norm for x in v]

    def embed(self, texts: List[str]) -> List[List[float]]:
        return [self._vec(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._vec(text)


class APIEmbeddingService:
    """OpenAI 兼容接口的 embedding 后端, 基于 langchain 的 OpenAIEmbeddings.

    与 EmbeddingService 的差别: 不本地加载模型, 走 HTTP —— 不占磁盘与显存, 代价是
    每次调用有网络延迟、按量计费, 且强依赖服务可用性.

    为什么用 langchain 的类而不是裸 HTTP: 连接池、重试、分批都由 SDK 负责, 且与 LLM
    侧 (llm_router 用 ChatOpenAI) 是同一套风格. 对外仍暴露 embed / embed_query 两个
    方法, 与另外两个后端接口一致, 所以 retriever 侧一行都不用改.

    两个实测注意点 (换用前务必知道):
      1) **维度随模型变**: bge-small-zh 是 512 维, bge-m3 是 1024 维. 换模型等于换
         向量空间, 旧 collection 必须重建, 否则查询会因维度不匹配直接报错.
      2) **跨请求并非逐位确定**: 同一段文本相隔数十秒的两次请求, 向量分量最大差
         约 1e-3 (服务端侧差异, 不是浮点误差). 不影响检索本身, 但意味着召回评估的
         Recall@1 有约 ±0.02 的噪声 —— 比对指标时小于该幅度的差异不可信.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str,
        batch: int = 16,
        max_chars: int = 8000,
        timeout: float = 60.0,
        max_retries: int = 3,
    ):
        # 惰性 import: 与 llm_router 的做法一致, 用 local 后端时不必加载 langchain-openai.
        from langchain_openai import OpenAIEmbeddings

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.batch = batch
        self.max_chars = max_chars
        # 维度在首次成功调用后确定: 不写死在配置里, 避免换模型时忘了同步改维度,
        # 那会导致写入的向量与 collection 期望维度不符, 且报错点在很下游.
        self.dim = 0
        self._emb = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
            base_url=self.base_url,
            # 关掉 SDK 的 token 级切分. 它用的是 tiktoken —— OpenAI 的 tokenizer,
            # 对 bge-m3 这类第三方模型的切分边界是错的; 开着会得到"按错误边界切开、
            # 再按 token 数加权平均"的向量, 比直接发出更难排查. 关掉后超长文本原样
            # 发给服务端, 由服务端按自己的上限处理.
            check_embedding_ctx_length=False,
            chunk_size=batch,          # 每批条数, 与服务端单请求条数限制对齐
            max_retries=max_retries,
            request_timeout=timeout,
        )

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # 先按**字符**截断, 兜底防止单条异常长文本让整批请求被服务端拒绝.
        # 注意这是字符不是 token: 当前 chunk 上限 500 字, 远小于模型窗口, 这个截断
        # 实际不触发; 若将来把 chunk 调大, 需要改成按 token 设限.
        prepared = [t[: self.max_chars] for t in texts]
        vecs = self._emb.embed_documents(prepared)
        if vecs and not self.dim:
            self.dim = len(vecs[0])
        return vecs

    def embed_query(self, text: str) -> List[float]:
        # 刻意不走 OpenAIEmbeddings.embed_query: 部分版本会按模型给查询加指令前缀,
        # 而 bge-m3 不需要前缀; 这里也要保证查询与文档走**完全相同**的编码路径.
        return self.embed([text])[0]


@lru_cache
def get_embedding_service() -> "EmbeddingService | APIEmbeddingService | FakeEmbeddingService":
    """按 settings.embedding_backend 选后端, 进程内单例.

    优先级: fake (测试/CI) > api > local.
    """
    if settings.fake_embedding or os.environ.get("PILKB_FAKE_EMBEDDING", "").lower() in ("1", "true", "yes"):
        logger.info("Using FakeEmbeddingService (PILKB_FAKE_EMBEDDING=1)")
        return FakeEmbeddingService()

    backend = getattr(settings, "embedding_backend", "local")
    if backend == "api":
        api_key = settings.embedding_api_key or os.environ.get("EMBEDDING_API_KEY", "")
        if not api_key:
            # 明确抛错而不是悄悄退回本地模型: 两种后端的语义空间和维度都不同,
            # 混着用会让整个向量库失效 —— 静默降级比直接失败危险得多.
            raise RuntimeError(
                "embedding_backend=api 但没拿到 API key: 请设置环境变量 EMBEDDING_API_KEY"
                "(或在项目根目录 .env 里配置 EMBEDDING_API_KEY)"
            )
        # 用 embedding_api_model 而不是 embedding_model: 两个后端的合理默认值不同
        # (local 用 bge-small-zh 省资源, api 用 bge-m3 拿 1024 维 + 8192 token 上下文).
        # 注意换模型等于换向量空间, 旧 collection 必须重建, 详见 config.py 的注释.
        logger.info(
            f"Using APIEmbeddingService: model={settings.embedding_api_model} "
            f"base={settings.embedding_api_base}"
        )
        return APIEmbeddingService(
            model=settings.embedding_api_model,
            base_url=settings.embedding_api_base,
            api_key=api_key,
            batch=settings.embedding_api_batch,
            max_chars=settings.embedding_max_chars,
        )
    return EmbeddingService()
