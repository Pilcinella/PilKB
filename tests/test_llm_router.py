"""测试 get_llm 的按请求切换能力.

要求 (出自 P3-8):
1. `get_llm()` 无参调用时, 返回默认 provider/model 的 client (向后兼容).
2. `get_llm("qwen")` 给出 qwen 的 client, 换 provider 不再需要重启进程.
3. `get_llm("openai", "gpt-4o")` 任意组合均按 (provider, model) 缓存.
4. 组合不同则 client 不同 (per-request 切换名副其实).
"""
from __future__ import annotations

from unittest.mock import patch
import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from app.agents import llm_router


@pytest.fixture(autouse=True)
def _clear_cache(monkeypatch):
    """每个测试前后重置缓存, 防止彼此污染."""
    monkeypatch.setattr(llm_router.settings, "fake_llm", False)
    llm_router.reset_llm_cache()
    yield
    llm_router.reset_llm_cache()


def _fake_router_get(provider=None, model=None):
    """顶替 LLMRouter.get, 以一个简单 dataclass 记录 provider/model."""
    from app.core.config import settings

    provider = (provider or settings.default_llm_provider).lower()
    model = model or settings.get_llm_model(provider)

    class _FakeLLM:
        def __init__(self):
            self.provider = provider
            self.model = model

    return _FakeLLM()


def test_get_llm_default_returns_provider_model():
    """get_llm() 给出 (default_provider, default_model) 组合."""
    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm()
    from app.core.config import settings

    assert a.provider == settings.default_llm_provider.lower()
    assert a.model == settings.get_llm_model(settings.default_llm_provider.lower())


def test_get_llm_per_provider_returns_distinct_clients():
    """get_llm('qwen') 与 get_llm('deepseek') 给出不同的 client."""
    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm("qwen")
        b = llm_router.get_llm("deepseek")
    assert a.provider == "qwen"
    assert b.provider == "deepseek"
    assert a is not b


def test_get_llm_per_model_returns_distinct_clients():
    """get_llm('openai', 'gpt-4o') 与 get_llm('openai', 'gpt-4o-mini') 对应不同的 cache key."""
    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm("openai", "gpt-4o")
        b = llm_router.get_llm("openai", "gpt-4o-mini")
    assert a.model == "gpt-4o"
    assert b.model == "gpt-4o-mini"
    assert a is not b


def test_get_llm_same_args_returns_same_instance():
    """相同 (provider, model) 必须给出同一实例 (复用 client, 免去重复握手)."""
    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm("qwen", "qwen-max")
        b = llm_router.get_llm("qwen", "qwen-max")
    assert a is b


def test_get_llm_resolve_default_uses_provider_default_model():
    """get_llm('qwen') 未指定 model 时, model 取 qwen provider 的默认 model."""
    from app.core.config import settings

    with patch.object(llm_router, "LLMRouter") as MockRouter:
        MockRouter.return_value.get.side_effect = lambda *a, **kw: _fake_router_get(*a, **kw)
        a = llm_router.get_llm("qwen")
    assert a.model == settings.get_llm_model("qwen")


def test_get_llm_uses_local_demo_model_when_explicitly_enabled(monkeypatch):
    """本地演示开关应避免外部模型请求, 同时保留流式接口。"""
    monkeypatch.setenv("PILKB_FAKE_LLM", "1")

    llm = llm_router.get_llm()

    assert llm.invoke("【检索结果】\n[1] 产品支持文档上传和混合检索。\n\n【用户问题】核心功能？").content
    assert list(llm.stream("任意问题"))[0].content


def test_local_demo_model_plans_retrieval_and_calculation(monkeypatch):
    """离线演示模型应产出 Agent 可消费的工具调用规划。"""
    monkeypatch.setenv("PILKB_FAKE_LLM", "1")
    llm = llm_router.get_llm()

    out = llm.invoke([
        SystemMessage(content="你是一个任务规划 Agent。必须输出 JSON。"),
        HumanMessage(content="根据产品手册计算 1280 * 0.85，并说明核心功能。"),
    ])
    plan = json.loads(out.content)

    assert plan["needs_calculator"] is True
    assert plan["needs_retrieval"] is True
