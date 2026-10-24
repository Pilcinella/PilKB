"""FastAPI dependencies: auth, current user."""
from __future__ import annotations

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core import token_version
from app.core.security import decode_token

bearer_scheme = HTTPBearer(auto_error=False)


# P2-11: JWT 改为 HttpOnly cookie, 但继续兼容 Bearer header (API 调用 / 第三方工具).
def get_current_user(
    request: Request,
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> str:
    """优先从 Authorization: Bearer 取 token, 否则回退到 HttpOnly cookie.

    这样前端可用 credentials: 'include' 自动带 cookie,
    同时保留对脚本 / 第三方工具的 Bearer 兼容.
    """
    token = ""
    if creds and creds.credentials:
        token = creds.credentials
    if not token:
        token = request.cookies.get("pilkb_token") or ""
    if not token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing token")
    payload = decode_token(token)
    if not payload or not payload.get("sub"):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    sub = payload["sub"]

    # 吊销校验: payload 里的 ver 必须等于 DB 当前 token_version。老 token (无 ver)
    # 按 0 处理, 注册用户初始就是 0, 向后兼容; 用户不在 DB (如 dev-token 的
    # dev-user) 不校验 —— 那类 token 本来就不对应账号。
    current = token_version.get_current_version(sub)
    if current is not None and payload.get("ver", 0) != current:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token revoked")
    return sub
