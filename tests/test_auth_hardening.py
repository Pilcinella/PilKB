"""鉴权加固测试 (对应本轮四项改动):

1. token 吊销: 登出 / bump token_version 后旧 token 立即 401; 老 token (无 ver)
   与 dev-token (无 DB 行) 不受影响。
2. XFF 可信边界: 默认不信任 X-Forwarded-For, 伪造 XFF 换 IP 绕不开登录锁定;
   配置了可信代理后按"从右往左跳过可信跳"解析。
3. 登录时序拉平: 用户名不存在时也走一次完整 bcrypt verify。
4. kb_registry 的 is_public 字段真正参与读决策 (公开可读, 但 owner 之外仍不可写)。
"""
from __future__ import annotations

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from app.api import auth as auth_mod
from app.core import net as net_mod
from app.core import rate_limit as rl_mod
from app.core import token_version as tv_mod
from app.core.config import get_settings


@pytest.fixture(autouse=True)
def _fresh_state():
    """每个测试前清用户表 / 限流 / token 缓存, 结束后还原 settings 缓存."""
    auth_mod.reset_users_for_tests()
    rl_mod.reset_for_tests()
    tv_mod.reset_for_tests()
    net_mod.reset_cache_for_tests()
    yield
    auth_mod.reset_users_for_tests()
    rl_mod.reset_for_tests()
    tv_mod.reset_for_tests()
    get_settings.cache_clear()
    net_mod.reset_cache_for_tests()


# ============ 单元: client_ip (XFF 可信边界) ============


def _req(peer: str, xff: str | None = None) -> Request:
    headers = [(b"x-forwarded-for", xff.encode("latin-1"))] if xff is not None else []
    scope = {
        "type": "http",
        "method": "GET",
        "path": "/",
        "headers": headers,
        "query_string": b"",
        "client": (peer, 12345),
    }
    return Request(scope)


def _set_trusted(monkeypatch, value: str | None):
    if value is None:
        monkeypatch.delenv("TRUSTED_PROXIES", raising=False)
    else:
        monkeypatch.setenv("TRUSTED_PROXIES", value)
    get_settings.cache_clear()
    net_mod.reset_cache_for_tests()


def test_client_ip_ignores_xff_without_trusted_proxies(monkeypatch):
    """默认 (未配置可信代理): XFF 是客户端可随意伪造的头, 整个丢弃."""
    _set_trusted(monkeypatch, None)
    assert net_mod.client_ip(_req("203.0.113.7", "6.6.6.6")) == "203.0.113.7"


def test_client_ip_ignores_xff_when_peer_not_in_list(monkeypatch):
    """配置了可信名单但直连方不在其中: 同样不采信 XFF."""
    _set_trusted(monkeypatch, "10.0.0.1")
    assert net_mod.client_ip(_req("203.0.113.7", "6.6.6.6")) == "203.0.113.7"


def test_client_ip_uses_xff_when_peer_trusted(monkeypatch):
    """直连方可信: 采信其追加的真实来源 (右端)."""
    _set_trusted(monkeypatch, "10.0.0.1")
    assert net_mod.client_ip(_req("10.0.0.1", "198.51.100.9")) == "198.51.100.9"


def test_client_ip_walks_right_to_left_past_trusted_hops(monkeypatch):
    """多级代理: 从右往左跳过可信跳, 取第一个非可信地址.

    客户端自带伪造头 "spoofed.example"; 10.0.0.2 (可信) 追加了真实来源
    203.0.113.7; 直连方 10.0.0.1 (可信). 旧实现"取第一个"会拿到 spoofed。
    """
    _set_trusted(monkeypatch, "10.0.0.1,10.0.0.2")
    req = _req("10.0.0.1", "spoofed.example, 203.0.113.7, 10.0.0.2")
    assert net_mod.client_ip(req) == "203.0.113.7"


def test_client_ip_trusted_peer_without_xff_falls_back_to_peer(monkeypatch):
    _set_trusted(monkeypatch, "10.0.0.1")
    assert net_mod.client_ip(_req("10.0.0.1")) == "10.0.0.1"


def test_client_ip_wildcard_trusts_everyone(monkeypatch):
    """"*" = 信任一切直连方 (仅限本机调试): 取最右一跳."""
    _set_trusted(monkeypatch, "*")
    assert net_mod.client_ip(_req("1.2.3.4", "spoofed, 198.51.100.9")) == "198.51.100.9"


# ============ 单元: kb_registry is_public 参与决策 ============


def test_registry_is_public_grants_read_but_not_write():
    from fastapi import HTTPException

    from app.core import kb_registry as kbr

    kbr.reset_for_tests()
    try:
        kbr.ensure_kb_for_write("beta", "alice")
        # 模拟"所有者把 kb 标成公开" (暂无 API, 走内部结构改标记并落盘)
        with kbr._LOCK:
            data = kbr._load_unlocked()
            data["beta"]["is_public"] = True
            kbr._save_unlocked(data)

        # 其他用户可读
        kb = kbr.ensure_kb_for_read("beta", "bob")
        assert kb.is_public is True
        # 但 owner 之外仍不可写
        with pytest.raises(HTTPException) as exc:
            kbr.ensure_kb_for_write("beta", "bob")
        assert exc.value.status_code == 403
        # owner 本人公开后仍可写
        kbr.ensure_kb_for_write("beta", "alice")
    finally:
        kbr.reset_for_tests()


# ============ 集成: HTTP 端点 ============


def _client_with_fakes(monkeypatch):
    """构造一个 patch 了 embedding/vector/bm25/reranker/llm 的 TestClient."""
    from app.agents import llm_router as lr_mod
    from app.rag import bm25_store as bm_mod
    from app.rag import embedding as emb_mod
    from app.rag import reranker as rk_mod
    from app.rag import retriever as rt_mod
    from app.rag import vector_store as vs_mod

    from tests._fakes import (
        FakeBM25Store,
        FakeEmbeddingService,
        FakeLLM,
        FakeReranker,
        FakeVectorStore,
    )

    import os

    os.environ["PILKB_FAKE_EMBEDDING"] = "1"
    emb_mod.get_embedding_service.cache_clear()
    monkeypatch.setattr(vs_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(bm_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(rk_mod, "CrossEncoderReranker", FakeReranker)
    monkeypatch.setattr(emb_mod, "EmbeddingService", FakeEmbeddingService)
    monkeypatch.setattr(rt_mod, "ChromaStore", FakeVectorStore)
    monkeypatch.setattr(rt_mod, "BM25Store", FakeBM25Store)
    monkeypatch.setattr(lr_mod, "get_llm", lambda: FakeLLM())

    from app.main import app

    return TestClient(app)


def _register(c: TestClient, username: str, password: str = "password123") -> str:
    r = c.post(
        "/api/v1/auth/register",
        json={"username": username, "email": f"{username}@example.com", "password": password},
    )
    assert r.status_code == 200, r.text
    return r.json()["access_token"]


def _list_ok(c: TestClient, token: str) -> int:
    return c.get(
        "/api/v1/documents/list",
        params={"kb_id": "default"},
        headers={"Authorization": f"Bearer {token}"},
    ).status_code


def test_logout_revokes_bearer_token(monkeypatch):
    """带 Bearer 登出后, 同一个 token 必须立刻 401 (而不是只删了 cookie)."""
    c = _client_with_fakes(monkeypatch)
    token = _register(c, "alice")
    assert _list_ok(c, token) == 200

    r = c.post("/api/v1/auth/logout", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert _list_ok(c, token) == 401

    # 重新登录拿到新 ver 的 token, 恢复可用
    r = c.post("/api/v1/auth/login", json={"username": "alice", "password": "password123"})
    assert r.status_code == 200, r.text
    assert _list_ok(c, r.json()["access_token"]) == 200


def test_logout_via_cookie_revokes_all_tokens(monkeypatch):
    """cookie 登出: 即便 token 已被拷走 (Bearer), 也一并吊销."""
    c = _client_with_fakes(monkeypatch)
    token = _register(c, "bob")
    assert _list_ok(c, token) == 200

    # 用 cookie 触发登出 (TestClient 会带上注册时存的 cookie)
    r = c.post("/api/v1/auth/logout")
    assert r.status_code == 200
    # 被拷走的 Bearer token 也要失效 —— 这才是 bump token_version 的意义
    assert _list_ok(c, token) == 401


def test_direct_version_bump_invalidates_tokens(monkeypatch):
    """不走 logout, 直接 bump token_version: 机制层校验."""
    c = _client_with_fakes(monkeypatch)
    token = _register(c, "carol")
    assert _list_ok(c, token) == 200
    tv_mod.bump_version("carol")
    assert _list_ok(c, token) == 401


def test_legacy_token_without_ver_still_valid(monkeypatch):
    """不带 ver 的老 token (按 0 处理) 对上初始 token_version=0, 向后兼容."""
    from app.core.security import create_access_token

    c = _client_with_fakes(monkeypatch)
    _register(c, "dave")  # 确保 DB 里有这个用户, version=0
    legacy = create_access_token(subject="dave")
    assert _list_ok(c, legacy) == 200


def test_dev_token_still_works_without_user_row(monkeypatch):
    """dev-token 的 subject (dev-user) 不在 users 表, 不做吊销校验."""
    monkeypatch.setenv("APP_ENV", "dev")
    get_settings.cache_clear()

    c = _client_with_fakes(monkeypatch)
    r = c.post("/api/v1/auth/dev-token")
    assert r.status_code == 200, r.text
    assert _list_ok(c, r.json()["access_token"]) == 200


def test_spoofed_xff_cannot_dodge_login_lock(monkeypatch):
    """回归: 伪造 X-Forwarded-For 换 IP 不再能绕过登录失败锁定."""
    monkeypatch.setenv("RATE_LIMIT_LOGIN_FAIL_MAX", "3")
    monkeypatch.setenv("RATE_LIMIT_LOGIN_WINDOW_SEC", "60")
    monkeypatch.setenv("RATE_LIMIT_LOGIN_LOCK_SEC", "60")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "1")
    monkeypatch.delenv("TRUSTED_PROXIES", raising=False)
    get_settings.cache_clear()
    net_mod.reset_cache_for_tests()

    c = _client_with_fakes(monkeypatch)
    for i in range(3):
        r = c.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "wrong"},
            headers={"X-Forwarded-For": f"1.2.3.{i}"},  # 每次换假 IP
        )
        assert r.status_code == 401
    # 换第 4 个假 IP 也逃不掉: 直连地址没变
    r = c.post(
        "/api/v1/auth/login",
        json={"username": "alice", "password": "wrong"},
        headers={"X-Forwarded-For": "9.9.9.9"},
    )
    assert r.status_code == 429, r.text
    assert r.headers.get("Retry-After") is not None


def test_login_unknown_user_still_runs_bcrypt(monkeypatch):
    """时序拉平: 用户名不存在也必须走一次 verify_password (哑哈希)."""
    c = _client_with_fakes(monkeypatch)
    calls = {"n": 0}
    orig = auth_mod.verify_password

    def counting(plain: str, hashed: str) -> bool:
        calls["n"] += 1
        return orig(plain, hashed)

    monkeypatch.setattr(auth_mod, "verify_password", counting)
    r = c.post("/api/v1/auth/login", json={"username": "ghost-user", "password": "whatever"})
    assert r.status_code == 401
    assert calls["n"] == 1, "用户不存在时也应执行一次 bcrypt verify 拉平时长"
