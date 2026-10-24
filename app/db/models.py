"""用户模型 (P2-12: 把内存 _USERS 迁移到 SQLite)."""
from __future__ import annotations

from sqlalchemy import Column, DateTime, Integer, String, func

from app.db.base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(128), unique=True, index=True, nullable=False)
    email = Column(String(256), nullable=False)
    password_hash = Column(String(256), nullable=False)
    # token 吊销: 登出时 +1, 该用户名下所有旧 JWT 的 ver 对不上即 401
    # (见 app/core/token_version.py)。
    token_version = Column(Integer, nullable=False, default=0, server_default="0")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
