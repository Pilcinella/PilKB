"""数据库连接与会话 (P2-12)."""
from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine, make_url
from sqlalchemy.orm import sessionmaker

from app.core.config import settings

# 项目根目录 (本文件位于 app/db/session.py, parents[2] 即根目录).
_BACKEND_ROOT = Path(__file__).resolve().parents[2]


def _resolve_database_url(url: str) -> str:
    """把 sqlite 的相对路径固化为相对项目根目录的绝对路径, 并建好父目录.

    默认配置是 `sqlite:///./data/pilkb.db`, 这个路径**相对于进程 CWD**: 只有 CWD
    恰好是项目根目录时才指向预期位置. 一旦 CWD 变化 (Docker WORKDIR 不同、
    systemd/supervisor 启动、测试里 monkeypatch.chdir), sqlite 要么在错误位置
    新建库, 要么因父目录不存在直接抛
    `OperationalError: unable to open database file`.

    这里统一按项目根目录解析, 让 CWD 不再影响数据库位置. PostgreSQL 等非
    sqlite URL 以及 `:memory:` 原样返回.
    """
    try:
        parsed = make_url(url)
    except Exception:
        return url

    if not parsed.drivername.startswith("sqlite"):
        return url

    db = parsed.database
    # 绝对路径 / :memory: / 空 -> 无需处理
    if not db or db.startswith("/") or db == ":memory:":
        return url

    abs_db = (_BACKEND_ROOT / db).resolve()
    try:
        abs_db.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        # 只读文件系统等场景建目录会失败; 保留原始 URL, 让 sqlite 自己报错.
        return url
    return parsed.set(database=str(abs_db)).render_as_string(hide_password=False)


DATABASE_URL = _resolve_database_url(settings.database_url)

_connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}

engine: Engine = create_engine(
    DATABASE_URL,
    connect_args=_connect_args,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# 注意: 不要提供"运行时重建 engine"的辅助函数. `from app.db import SessionLocal`
# 这类导入时绑定拿到的是旧 sessionmaker 对象, 重新绑定 session.SessionLocal 之后
# auth.py / get_db() 里已有的引用不会跟着更新, 调用方会以为切了库其实还在写旧库.
# 真要换库就在启动时通过 DATABASE_URL 指定, 或重启进程.
