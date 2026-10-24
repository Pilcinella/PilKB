"""Auth API: 注册 / 登录（演示用，生产请接企业 SSO / OAuth）.

P2-12: 用户表从内存 dict 迁移到 SQLite (SQLAlchemy).
"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, HTTPException, Request, Response, status
from loguru import logger
from sqlalchemy.exc import IntegrityError

from app.core import token_version
from app.core.config import get_settings, settings
from app.core.net import client_ip
from app.core.rate_limit import (
    check_fixed_quota,
    check_login_failure,
    record_login_failure,
    record_login_success,
)
from app.core.security import create_access_token, decode_token, hash_password, verify_password
from app.db import SessionLocal, User, create_tables, engine
from app.schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])

# cookie 名
_AUTH_COOKIE_NAME = "pilkb_token"

# 时序拉平用的哑哈希: 用户名不存在时也跑一次完整 bcrypt verify。否则"用户不存在"
# 的响应明显快于"密码错误" (后者要花 ~100ms 做 bcrypt), 等于把用户名枚举口子留给
# 计时攻击 —— 错误文案统一防枚举的初衷就被绕开了。惰性生成, 不增加 import 耗时。
_DUMMY_HASH: str | None = None


def _verify_dummy_password(password: str) -> None:
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        _DUMMY_HASH = hash_password("pilkb-timing-equalizer")
    verify_password(password, _DUMMY_HASH)


def _extract_raw_token(request: Request) -> str:
    """非抛错版 token 提取 (与 deps.get_current_user 同序): Bearer 优先, 回退 cookie."""
    auth = request.headers.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return request.cookies.get(_AUTH_COOKIE_NAME) or ""


# cookie 在生产 (非 dev) 时, secure=True + samesite=strict.
# dev 时 secure=False (因为 http://localhost:3000), samesite=lax (避免跨站干扰).
def _set_auth_cookie(response: Response, token: str, expire_seconds: int) -> None:
    cfg = get_settings()
    is_prod = cfg.app_env == "prod"
    response.set_cookie(
        key=_AUTH_COOKIE_NAME,
        value=token,
        max_age=expire_seconds,
        httponly=True,           # JS 读不到, 防 XSS 窃 token
        secure=is_prod,          # dev 用 http, 不强制 https
        samesite="strict" if is_prod else "lax",
        path="/",
    )


def _clear_auth_cookie(response: Response) -> None:
    cfg = get_settings()
    is_prod = cfg.app_env == "prod"
    response.delete_cookie(
        key=_AUTH_COOKIE_NAME,
        path="/",
        httponly=True,
        secure=is_prod,
        samesite="strict" if is_prod else "lax",
    )


def _get_user_by_username_sync(username: str) -> User | None:
    db = SessionLocal()
    try:
        return db.query(User).filter(User.username == username).first()
    finally:
        db.close()


def _create_user_sync(username: str, email: str, password: str) -> User:
    db = SessionLocal()
    try:
        existing = db.query(User).filter(User.username == username).first()
        if existing:
            raise ValueError("用户已存在")
        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
        )
        db.add(user)
        try:
            db.commit()
        except IntegrityError:
            # check-then-insert 的窗口内, 并发注册同名用户会撞 username 的 unique
            # 约束. 不接的话第二个请求直接 500.
            db.rollback()
            raise ValueError("用户已存在")
        db.refresh(user)
        return user
    finally:
        db.close()


def reset_users_for_tests() -> None:
    """仅供测试使用: 清空 SQLite users 表, 让每个测试隔离."""
    create_tables()
    db = SessionLocal()
    try:
        db.query(User).delete()
        db.commit()
    finally:
        db.close()


@router.post("/register", response_model=TokenResponse)
async def register(req: RegisterRequest, request: Request, response: Response):
    # register 无认证可依, 只能按 IP 限流: 不设限可以无限注册填爆 users 表.
    ip = client_ip(request)
    decision = check_fixed_quota(
        "register", ip, get_settings().rate_limit_register_per_min, 60.0
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=decision.reason,
            headers={"Retry-After": str(int(decision.retry_after) + 1)},
        )
    try:
        user = await asyncio.to_thread(_create_user_sync, req.username, req.email, req.password)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    token = create_access_token(subject=req.username, token_version=user.token_version)
    _set_auth_cookie(response, token, settings.jwt_expire_minutes * 60)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/login", response_model=TokenResponse)
async def login(req: LoginRequest, request: Request, response: Response):
    ip = client_ip(request)
    decision = check_login_failure(req.username, ip)
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=decision.reason,
            headers={"Retry-After": str(int(decision.retry_after) + 1)},
        )

    user = await asyncio.to_thread(_get_user_by_username_sync, req.username)
    if user is None:
        # 时序拉平: 用户不存在也走一次完整 bcrypt (见 _verify_dummy_password),
        # 不给计时攻击留"用户名是否存在"的信号。同时记得记失败次数。
        await asyncio.to_thread(_verify_dummy_password, req.password)
        record_login_failure(req.username, ip)
        raise HTTPException(status_code=401, detail="用户名或密码错误")
    # bcrypt 是 ~100ms 级的 CPU 密集操作, 卸到线程池避免堵住事件循环。
    if not await asyncio.to_thread(verify_password, req.password, user.password_hash):
        record_login_failure(req.username, ip)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    record_login_success(req.username, ip)
    token = create_access_token(subject=req.username, token_version=user.token_version)
    _set_auth_cookie(response, token, settings.jwt_expire_minutes * 60)
    return TokenResponse(access_token=token, expires_in=settings.jwt_expire_minutes * 60)


@router.post("/logout", summary="清空 auth cookie + 吊销当前用户全部 token")
async def logout(request: Request, response: Response):
    _clear_auth_cookie(response)
    # 尽力吊销: 从 Bearer / cookie 解析出用户名就 bump token_version, 该用户名下
    # 所有旧 token (包括已经拷走的) 一并失效。解析不出有效 token (本来就没登录)
    # 也照常返回 ok, 登出保持幂等 —— 清 cookie 这步永远先做。
    token = _extract_raw_token(request)
    payload = decode_token(token) if token else None
    sub = payload.get("sub") if payload else None
    if sub:
        try:
            await asyncio.to_thread(token_version.bump_version, sub)
        except Exception as e:  # noqa: BLE001
            logger.warning(f"logout: bump token_version failed for {sub}: {e}")
    return {"status": "ok"}


@router.post("/dev-token", response_model=TokenResponse, summary="仅 dev 模式：直接拿一个 token 调试用")
async def dev_token(response: Response):
    """仅在 APP_ENV=dev 时可用，方便本地调试。

    生产环境被 fail-fast 启动校验保护 (见 app.main.lifespan),
    这里再加运行时二次校验 + 警告日志, 便于运维监控误用.
    """
    # 用 get_settings() 实时读取, 避免模块级缓存的 settings 与进程 env 不一致
    # (例如测试时动态改 env 的场景).
    cfg = get_settings()
    if cfg.app_env != "dev":
        logger.warning(
            "/auth/dev-token 被拒绝调用: app_env={!r} (期望 'dev'). "
            "如果这是生产环境被攻击, 请检查 APP_ENV / JWT_SECRET 配置.",
            cfg.app_env,
        )
        raise HTTPException(status_code=403, detail="仅 dev 模式可用")
    logger.warning("/auth/dev-token 被调用, 颁发 dev-user token; 仅 dev 环境应启用此接口.")
    token = create_access_token(subject="dev-user")
    _set_auth_cookie(response, token, cfg.jwt_expire_minutes * 60)
    return TokenResponse(access_token=token, expires_in=cfg.jwt_expire_minutes * 60)
