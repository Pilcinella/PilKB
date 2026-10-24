"""CrossEncoderReranker 单例缓存的测试.

要求 (出自 P1-3):
1. `get_reranker()` 连续调用两次返回同一对象 (is 判断).
2. model_name 不同则实例不同.
3. 底层 CrossEncoder 仅实例化一次 (通过 mock `CrossEncoder.__init__` 计数).
4. 业务代码 (`pipeline.py` / `chat.py` / `tools.py`) 均不再直接 `CrossEncoderReranker()`.
"""
from __future__ import annotations

import sys
import types

import pytest


@pytest.fixture
def fake_cross_encoder(monkeypatch):
    """将 sentence_transformers.CrossEncoder 换成可计数的 fake 实现."""
    created = {"count": 0}

    class _FakeCE:
        def __init__(self, model_name):
            created["count"] += 1
            self.model_name = model_name

        def predict(self, pairs, **_kwargs):
            # 默认分数全部相同, 不干扰排序断言.
            return [0.5 for _ in pairs]

    fake_mod = types.ModuleType("sentence_transformers")
    fake_mod.CrossEncoder = _FakeCE  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", fake_mod)
    # 重置单例缓存, 保证测试从干净状态开始.
    from app.rag import reranker as rk_mod

    rk_mod.reset_reranker_cache()
    yield created, _FakeCE
    rk_mod.reset_reranker_cache()


def test_get_reranker_returns_same_instance(fake_cross_encoder):
    """get_reranker() 对同一 model_name 必须返回同一对象."""
    from app.rag.reranker import get_reranker

    a = get_reranker()
    b = get_reranker()
    assert a is b, "两次连续 get_reranker() 必须拿到同一实例"


def test_get_reranker_loads_model_only_once(fake_cross_encoder):
    """即便 rerank 被调用多次, 底层 CrossEncoder 也只构造一次."""
    from app.rag.reranker import get_reranker

    count, _ = fake_cross_encoder
    rk = get_reranker()
    # 多次调用 rerank, 底层 CrossEncoder 仍只构造一次 (懒加载 + 缓存).
    for _ in range(3):
        rk.rerank("query", [{"document": "x"}, {"document": "y"}], top_k=2)
    # 反复 get_reranker() 同样不应触发重新构造.
    for _ in range(3):
        get_reranker().rerank("query", [{"document": "x"}], top_k=1)
    assert count["count"] == 1, f"底层 CrossEncoder 应仅构造 1 次, 实际构造了 {count['count']} 次"


def test_get_reranker_distinct_models(fake_cross_encoder):
    """model_name 不同则落到不同实例 (缓存按 model_name 分桶)."""
    from app.rag.reranker import get_reranker

    a = get_reranker("model-A")
    b = get_reranker("model-A")
    c = get_reranker("model-B")
    assert a is b, "相同 model_name 应给出同一实例"
    assert a is not c, "相异 model_name 应给出不同实例"


def test_business_code_uses_singleton(monkeypatch):
    """pipeline.py / chat.py / tools.py 均使用 get_reranker(), 而非 CrossEncoderReranker().

    对这几个文件做静态扫描, 禁止 `CrossEncoderReranker()` 这类直接实例化写法.
    """
    targets = [
        "app/rag/pipeline.py",
        "app/api/chat.py",
        "app/agents/tools.py",
    ]
    for path in targets:
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        # import 与类型注解里出现 `CrossEncoderReranker` 是允许的, 直接调用不允许.
        # 简化处理: 不去区分 `CrossEncoderReranker` 后接 `(` 是否为方法调用.
        # 此处仅放行 `get_reranker()` 与 (调试用的) `reset_reranker_cache`.
        bad = []
        for line in src.splitlines():
            # 注释行与 import 行直接跳过.
            stripped = line.strip()
            if stripped.startswith(("#", "from ")):
                continue
            # 形如 `CrossEncoderReranker(...)` 即直接实例化.
            if "CrossEncoderReranker(" in line:
                bad.append(line)
        assert not bad, (
            f"{path} 中仍存在直接 CrossEncoderReranker() 实例化, 应改用 get_reranker():\n"
            + "\n".join(bad)
        )


def test_get_reranker_can_be_reset(fake_cross_encoder):
    """reset_reranker_cache 之后, 再 get_reranker 将重建对象."""
    from app.rag.reranker import get_reranker, reset_reranker_cache

    a = get_reranker()
    a.rerank("query", [{"document": "x"}], top_k=1)  # 促使底层模型加载
    count, _ = fake_cross_encoder
    assert count["count"] == 1

    reset_reranker_cache()
    b = get_reranker()
    b.rerank("query", [{"document": "x"}], top_k=1)  # 促使新对象加载底层模型
    assert a is not b
    assert count["count"] == 2, "reset 之后底层 CrossEncoder 被重新构造了一次"