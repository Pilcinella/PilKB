"""LangFuse 可观测性集成 (可选模块)。

启用方法: 在 .env 里设 LANGFUSE_ENABLED=true, 并补齐 LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY。
关闭状态下全部 hook 均为 no-op, 主流程不受影响。

设计要点:
1. 对 langfuse 包无强依赖; 未安装时自动退化。
2. 提供 trace / span / generation 三层 wrapper, 对 LangChain / LangGraph 的调用方透明。
3. 链路: HTTP request -> RAG pipeline -> retriever / rerank / LLM, 每一层的耗时与 token 用量都能在 LangFuse 查看。
"""
from __future__ import annotations

import os
import time
from contextlib import contextmanager
from typing import Any, Optional

from loguru import logger

from app.core.config import settings


class _NoopSpan:
    """功能关闭时使用的占位 span, 各方法均返回原对象。"""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, **_):
        return self

    def end(self, **_):
        return self


class _NoopTrace:
    def __init__(self, *_args, **_kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def update(self, **_):
        return self


class LangfuseTracer:
    """单例 tracer, 用于屏蔽安装与配置上的差异。"""

    def __init__(self):
        self._enabled = bool(settings.langfuse_enabled)
        self._client = None
        if self._enabled:
            self._try_init()

    def _try_init(self):
        try:
            from langfuse import Langfuse  # type: ignore

            self._client = Langfuse(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            logger.info(f"Langfuse initialized, host={settings.langfuse_host}")
        except Exception as e:
            logger.warning(f"Langfuse init failed, falling back to noop: {e}")
            self._enabled = False
            self._client = None

    @property
    def enabled(self) -> bool:
        return self._enabled and self._client is not None

    @contextmanager
    def trace(self, name: str, **metadata):
        """顶层 trace: 对应一次请求或一次评估。"""
        if not self.enabled:
            yield _NoopTrace()
            return
        try:
            with self._client.trace(name=name, metadata=metadata) as t:
                yield t
        except Exception as e:
            logger.debug(f"langfuse trace '{name}' error: {e}")
            yield _NoopTrace()

    @contextmanager
    def span(self, trace, name: str, **metadata):
        if not self.enabled or trace is None:
            yield _NoopSpan()
            return
        try:
            with trace.span(name=name, metadata=metadata) as s:
                yield s
        except Exception as e:
            logger.debug(f"langfuse span '{name}' error: {e}")
            yield _NoopSpan()

    def generation(
        self,
        trace,
        name: str,
        model: str,
        prompt: str,
        completion: Optional[str] = None,
        usage: Optional[dict] = None,
        **metadata: Any,
    ):
        """记一笔 LLM 调用。"""
        if not self.enabled or trace is None:
            return None
        try:
            return trace.generation(
                name=name,
                model=model,
                input=prompt,
                output=completion,
                usage_details=usage or {},
                metadata=metadata,
            )
        except Exception as e:
            logger.debug(f"langfuse generation '{name}' error: {e}")
            return None


_tracer: Optional[LangfuseTracer] = None


def get_tracer() -> LangfuseTracer:
    global _tracer
    if _tracer is None:
        _tracer = LangfuseTracer()
    return _tracer


# ---- 装饰器与上下文管理器形式的高级封装 ----


@contextmanager
def traced(name: str, **metadata):
    """基于全局 tracer 的便捷入口。"""
    t = get_tracer()
    with t.trace(name=name, **metadata) as ctx:
        yield ctx


def timed(name: str):
    """轻量计时上下文, 与 Langfuse 无关, 供本地 profiling 用。"""

    @contextmanager
    def _wrap():
        start = time.perf_counter()
        try:
            yield
        finally:
            logger.debug(f"[perf] {name} elapsed={(time.perf_counter() - start) * 1000:.1f}ms")

    return _wrap()
