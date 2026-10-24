# -*- coding: utf-8 -*-
"""统一 LLM 路由：DeepSeek / Qwen / OpenAI，自动按 provider 拼装 chat model。

支持按请求切换 provider / model: `get_llm(provider, model)` 会按 (provider, model)
作 key 缓存, 同一组合复用同一 client, 换 provider / model 才新建.

向后兼容: `get_llm()` 不带参数仍返回默认 provider 的默认 model (走 `settings`),
这样老的调用点 `get_llm().invoke(...)` 不需要改动.

测试里如果 patch 了 `_build_llm`, 也可以直接 monkey-patch 该函数实现自定义 fake.
"""
from __future__ import annotations
from functools import lru_cache
import json
import os
import re
from typing import Callable, Literal, Optional, Tuple

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk

from app.core.config import settings


Provider = Literal["deepseek", "qwen", "openai"]


class LocalDemoLLM:
    """不依赖外部服务的本地演示模型，仅在显式环境开关下使用。"""

    @staticmethod
    def _answer(prompt: object) -> str:
        messages = prompt if isinstance(prompt, (list, tuple)) else []
        system_text = "\n".join(
            str(getattr(message, "content", "")) for message in messages[:-1]
        )
        user_text = str(getattr(messages[-1], "content", "")) if messages else str(prompt)
        text = "\n".join(str(getattr(message, "content", message)) for message in messages) or str(prompt)

        if "任务规划 Agent" in system_text:
            has_math = bool(re.search(r"\d\s*[+\-*/%]\s*\d|计算|折扣|总价", user_text))
            has_date = any(word in user_text for word in ("今天", "当前日期", "现在日期", "日期"))
            needs_retrieval = any(
                word in user_text for word in ("文档", "手册", "产品", "知识库", "根据", "说明")
            )
            return json.dumps({
                "needs_calculator": has_math,
                "needs_retrieval": needs_retrieval,
                "needs_date": has_date,
                "keywords": user_text[:200],
            }, ensure_ascii=False)

        # 无检索直答 (pipeline.build_direct_prompt). 必须显式识别: 那条提示词里
        # 没有 [1] 这种编号, 落到最后的兜底文案会得到"请先上传文档再提问" ——
        # 在演示里看着像正常回答, 其实完全答非所问。
        if "没有**检索知识库" in text:
            q = text.split("【用户问题】", 1)[-1].split("【回答】", 1)[0].strip()
            return "（本地演示：本次未检索知识库）关于「" + q + "」，请以真实模型接入后的回答为准。"

        contexts = re.findall(
            r"\[\d+\]\s*(.*?)(?=\n\s*\[\d+\]|\n\n【用户问题】|\Z)",
            text,
            flags=re.DOTALL,
        )
        if contexts:
            return "根据知识库检索结果：" + contexts[0].strip()[:300] + " [1]"
        return "本地演示模式已开启。请先上传文档，再基于检索结果提问。"

    _AGENT_SYS_MARK = "知识库问答 Agent"

    @staticmethod
    def _agent_step(messages) -> AIMessage:
        """react 引擎的脚本化决策, 由消息内容驱动 (无实例状态, 可重放):

        - 已有 ToolMessage (收到过工具观测) -> 宣布证据足够, 循环终止;
        - 问题含数学 -> 发一次 calculator;
        - 问题像知识库提问 -> 发一次 hybrid_search;
        - 都不像 -> 纯文本回应 (不带 tool_calls, 直接进 compose 的 direct 分支)。
        """
        from langchain_core.messages import ToolMessage

        if any(isinstance(m, ToolMessage) for m in messages):
            return AIMessage(content="证据已足够，可以作答。")
        user_text = str(getattr(messages[-1], "content", "")) if messages else ""
        if re.search(r"\d\s*[+\-*/%]\s*\d|计算|折扣|总价", user_text):
            m = re.search(r"[\d\.\(\)\+\-\*/\s]+", user_text)
            expr = (m.group(0).strip() if m else "0") or "0"
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "calculator",
                        "args": {"expression": expr},
                        "id": "call_demo_calc",
                        "type": "tool_call",
                    }
                ],
            )
        if any(
            w in user_text
            for w in ("文档", "手册", "产品", "知识库", "根据", "说明", "报销", "价格", "流程")
        ):
            return AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "hybrid_search",
                        "args": {"query": user_text[:100]},
                        "id": "call_demo_search",
                        "type": "tool_call",
                    }
                ],
            )
        return AIMessage(content="（本地演示）该问题无需检索知识库。")

    def bind_tools(self, tools):
        """react 循环会 bind_tools; 本地演示模型没有真正的 function calling,
        返回自身、按 ``_agent_step`` 的脚本响应, 接口形状对齐即可。"""
        return self

    def invoke(self, prompt: object) -> AIMessage:
        messages = prompt if isinstance(prompt, (list, tuple)) else []
        system_text = "\n".join(str(getattr(m, "content", "")) for m in messages[:-1])
        if self._AGENT_SYS_MARK in system_text:
            return self._agent_step(messages)
        return AIMessage(content=self._answer(prompt))

    def stream(self, prompt: object):
        yield AIMessageChunk(content=self._answer(prompt))

    async def astream(self, prompt: object):
        yield AIMessageChunk(content=self._answer(prompt))


def _parse_extra_body(raw: Optional[str]) -> Optional[dict]:
    """解析 OPENAI_EXTRA_BODY: JSON 字符串 -> dict; 失败/空 -> None."""
    if not raw:
        return None
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return None
    return parsed if isinstance(parsed, dict) else None


class LLMRouter:
    def get(self, provider=None, model=None) -> BaseChatModel:
        provider = (provider or settings.default_llm_provider).lower()
        model = model or settings.default_llm_model
        api_key = settings.get_llm_api_key(provider)
        if not api_key:
            raise ValueError("API key for provider '" + provider + "' not set. Please fill " + provider.upper() + "_API_KEY in .env")
        base_url = settings.get_llm_base_url(provider)
        from langchain_openai import ChatOpenAI
        kwargs = dict(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.2,
            streaming=True,
            # 显式超时与重试: openai SDK 默认最长挂 600s, 而 LLM 调用点都是
            # to_thread 卸载 (线程不可取消), provider 半死不活时线程池会被挂起
            # 调用占满, 直到 SDK 自己超时.
            timeout=settings.llm_request_timeout,
            max_retries=settings.llm_max_retries,
        )
        # 仅 openai 槽位走 extra_body (其它 provider 走自己的封装, 不掺和).
        if provider == "openai":
            extra = _parse_extra_body(settings.openai_extra_body)
            if extra:
                kwargs["extra_body"] = extra
        return ChatOpenAI(**kwargs)


def _resolve_default_pair() -> Tuple[str, str]:
    return (settings.default_llm_provider.lower(), settings.default_llm_model)


@lru_cache(maxsize=16)
def _build_llm(provider: str, model: str) -> BaseChatModel:
    """按 (provider, model) 缓存 client 实例.

    同样组合多次调用复用同一个对象, 既避免重复 handshake, 也方便做 per-request
    切换 (`get_llm("qwen", "qwen-max")` 和 `get_llm("deepseek", "deepseek-chat")`
    会得到两个不同的 client).
    """
    if settings.fake_llm or os.environ.get("PILKB_FAKE_LLM", "").lower() in ("1", "true", "yes"):
        return LocalDemoLLM()  # type: ignore[return-value]
    return LLMRouter().get(provider=provider, model=model)


def get_llm(provider: Optional[str] = None, model: Optional[str] = None) -> BaseChatModel:
    """获取 LLM client.

    - `get_llm()` -> 默认 provider/model (进程启动时由环境变量固定).
    - `get_llm("qwen")` -> 指定 provider, model 用对应 provider 的默认 model.
    - `get_llm("openai", "gpt-4o")` -> 任意组合, 按 (provider, model) 缓存.
    """
    if provider is None and model is None:
        provider, model = _resolve_default_pair()
    else:
        provider = (provider or settings.default_llm_provider).lower()
        if not model:
            model = settings.get_llm_model(provider)
    return _build_llm(provider, model)


def reset_llm_cache() -> None:
    """仅供测试: 清空 (provider, model) -> client 缓存."""
    _build_llm.cache_clear()


def get_llm_text_callable(
    provider: Optional[str] = None, model: Optional[str] = None
) -> Callable[[str], str]:
    """返回一个 (prompt: str) -> str 的纯文本适配器.

    场景: metadata/HyDE/查询改写等只需要模型吐出的"文本", 但 `get_llm()` 返回的是
    `BaseChatModel`, 直接 `llm(prompt).strip()` 拿到的是 `AIMessage`, 会抛
    `AttributeError` 被 except 吞掉, 导致 LLM 分支永远走不到 (这是之前查询改写/HyDE
    不生效的根因). 这里统一取出 `.content`.
    """
    llm = get_llm(provider, model)

    def _call(prompt: str) -> str:
        out = llm.invoke(prompt)
        if isinstance(out, str):
            return out
        return getattr(out, "content", "") or ""

    return _call
