"""JWT auth + password hashing."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import JWTError, jwt
from loguru import logger
from passlib.context import CryptContext

from app.core.config import settings

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")


# 标记"未改的默认值". 当 prod 环境检测到 jwt_secret 仍是这一类值时, 必须 fail-fast
# 启动, 不允许运行时静默允许 /auth/dev-token 这类口子.
_DEFAULT_JWT_SECRET_PREFIXES = (
    "change-me",
    "changeme",
    "please-change",
    "default-secret",
)


class InsecureJWTConfigError(RuntimeError):
    """生产环境检测到 JWT 配置不安全 (例如 jwt_secret 是默认值)."""


def hash_password(password: str) -> str:
    return _pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


def create_access_token(
    subject: str,
    expires_delta: Optional[timedelta] = None,
    token_version: Optional[int] = None,
) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    payload = {"sub": subject, "exp": expire, "iat": datetime.now(timezone.utc)}
    # ver: 颁发时刻该用户的 token_version, 用于登出/吊销校验 (见 token_version.py).
    # 不传则不带 (老 token / 纯调试 token), 校验侧按 0 处理。
    if token_version is not None:
        payload["ver"] = token_version
    return jwt.encode(payload, settings.jwt_secret, algorithm=settings.jwt_algorithm)


def decode_token(token: str) -> Optional[dict]:
    """Return full payload or None if invalid."""
    try:
        return jwt.decode(token, settings.jwt_secret, algorithms=[settings.jwt_algorithm])
    except JWTError:
        return None


def verify_token(token: str) -> Optional[str]:
    """Return subject (user id) or None if invalid."""
    payload = decode_token(token)
    return payload.get("sub") if payload else None


def is_default_jwt_secret(secret: str) -> bool:
    """判断 jwt_secret 是否还是 config.py 里的默认值/弱口令. 用于 prod fail-fast."""
    if not secret:
        return True
    s = secret.strip().lower()
    if not s:
        return True
    if any(s.startswith(p) for p in _DEFAULT_JWT_SECRET_PREFIXES):
        return True
    # 长度 < 32 也视为弱: HS256 是对称签名, 密钥就是防伪造的全部本钱,
    # 报错文案一直写的是"建议 ≥ 32 字符", 阈值与文案对齐。
    if len(secret) < 32:
        return True
    return False


def validate_production_security(cfg=None) -> None:
    """生产环境安全校验. 不通过抛 InsecureJWTConfigError, 调用方应当 fail-fast.

    检查项:
    - app_env == "prod" 且 jwt_secret 是默认/弱口令 → 拒绝启动
    - app_env == "prod" 且 app_env 真被设置 (避免遗忘)
    """
    if cfg is None:
        from app.core.config import settings as _default_cfg

        cfg = _default_cfg
    if cfg.app_env != "prod":
        return
    if is_default_jwt_secret(cfg.jwt_secret):
        raise InsecureJWTConfigError(
            "生产环境 (APP_ENV=prod) 检测到 jwt_secret 仍是默认值或弱口令, "
            "请通过环境变量 JWT_SECRET 设置一个高熵随机串 (建议 ≥ 32 字符). "
            "这通常意味着 .env 没有被正确加载, 启动必须 fail-fast, 而不是运行时留后门."
        )
