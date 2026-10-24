"""数据库初始化入口."""
from __future__ import annotations

from app.db.base import Base
from app.db.models import User  # noqa: F401
from app.db.session import engine, SessionLocal


def create_tables() -> None:
    """根据模型建表 (应用启动时调用), 并给旧库补齐后加的列.

    create_all 不会给已存在的表加列: 老库升级 (如 users 新增 token_version) 若
    不迁移, 首次读写就在运行时炸 no such column。项目没有 alembic, 这里做最小
    迁移 —— 检查到缺列就 ALTER TABLE ADD COLUMN ... NOT NULL DEFAULT 0
    (sqlite / postgres 都接受这个形式, 且存量行自动填默认值)。
    """
    from sqlalchemy import inspect, text

    Base.metadata.create_all(bind=engine)
    insp = inspect(engine)
    if "users" not in insp.get_table_names():
        return
    columns = {c["name"] for c in insp.get_columns("users")}
    if "token_version" not in columns:
        with engine.begin() as conn:
            conn.execute(
                text("ALTER TABLE users ADD COLUMN token_version INTEGER NOT NULL DEFAULT 0")
            )


def drop_tables() -> None:
    Base.metadata.drop_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


__all__ = ["Base", "User", "engine", "SessionLocal", "create_tables", "drop_tables", "get_db"]
