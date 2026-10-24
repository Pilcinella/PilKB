"""FastAPI application entry."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response

from app import __version__
from app.api import auth, chat, documents, health, history, kbs
from app.core.config import get_settings
from app.core.logging import logger
from app.core.security import InsecureJWTConfigError, validate_production_security
from app.db import create_tables


def _parse_cors_origins(raw: str) -> list[str]:
    """解析 CORS 白名单: 支持逗号分隔字符串或 JSON 数组."""
    s = (raw or "").strip()
    if not s:
        return []
    if s.startswith("["):
        try:
            import json as _json
            parsed = _json.loads(s)
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        except Exception:
            return []
    return [x.strip() for x in s.split(",") if x.strip()]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 通过 get_settings() 实时读取 (而不是模块级缓存), 这样部署时改 env
    # 也能正确触发 fail-fast.
    cfg = get_settings()
    try:
        validate_production_security(cfg)
    except InsecureJWTConfigError as e:
        logger.critical(f"Refusing to start: {e}")
        raise

    # P2-12: 启动时创建用户表 (SQLite 持久化 _USERS 替代内存 dict)
    create_tables()

    # 冷启动预热: 在单线程的启动阶段把 chromadb 的首次引导 (建 sqlite / 跑迁移 /
    # 建 tenant / 建 collection) 做掉, 运行期的请求就都走"已就绪"的热路径.
    # 为什么需要: 引导在并发下不安全 (实测 4 线程同时首次触达有 25%~43% 抛
    # ValueError / OperationalError), 而运行期确实存在并发首触 —— 改写产生多条
    # query 时 pipeline 会 asyncio.gather 并发检索同一个冷 store, 两个不同 kb 的
    # 并发上传也会同时触达共享的 persist 目录.
    # 预热只是把窗口挪走, 真正的并发保护在 ChromaStore._ensure 的模块级锁.
    # 失败不阻断启动: 首次请求仍会再试一次.
    try:
        from app.rag.retriever import get_retriever

        get_retriever("default").vector_store.count()
        logger.info("Chroma warm-up done (kb=default)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Chroma warm-up skipped: {e}")

    # 存储一致性自检: 逐个 kb 比较 BM25 内存 docs 与向量库的 chunk 数.
    #
    # 为什么需要: 这是两份独立存储, 任何一边出问题都不会抛异常 —— 检索照常返回
    # 结果 (BM25 路静默为空, 退化为纯向量检索), 接口全是 200, 只是父文档召回不再
    # 生效、喂给 LLM 的上下文缺段. 没有基线对照时几乎无法定位, 所以在启动时用日志
    # 明确暴露出来.
    #
    # 只告警不自动修复: 重建 BM25 需要先把根因 (多进程写入) 解决掉, 见 Dockerfile
    # 的 --workers 说明; 否则重建出来的 BM25 只是让两份都可能残缺的数据彼此一致.
    # 修复入口是手动端点 POST /documents/rebuild-bm25 (从向量库回灌), 不在这里
    # 自动调. 失败不阻断启动.
    try:
        from app.core.kb_registry import list_kb_ids
        from app.rag.retriever import check_store_consistency

        for kb_id in list_kb_ids():
            st = check_store_consistency(kb_id)
            if st["consistent"]:
                logger.info(f"store check ok (kb={kb_id}, chunks={st['vector_count']})")
            else:
                logger.error(
                    f"store 不一致 (kb={kb_id}): bm25={st['bm25_count']} "
                    f"vector={st['vector_count']} —— 检索仍可用(退化为纯向量路), "
                    f"但父文档召回(连坐)对该 kb 已失效, 上下文会缺段. "
                    f"修复: POST /api/v1/documents/rebuild-bm25?kb_id={kb_id} "
                    f"(从向量库回灌), 或解决根因后重新入库该 kb 的文档."
                )
    except Exception as e:  # noqa: BLE001
        logger.warning(f"store consistency check skipped: {e}")

    logger.info(f"PilKB v{__version__} starting... env={cfg.app_env}")
    yield
    logger.info("PilKB shutting down.")


class SecurityHeadersMiddleware:
    """统一安全响应头.

    P2-11: 当使用 HttpOnly cookie 存 JWT 时, XSS 仍能读出页面内容, 因此需要 CSP
    限制脚本来源; 同时打开 HSTS / X-Content-Type-Options 等基础 header.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def _send(message):
            if message["type"] == "http.response.start":
                headers = message.get("headers") or []
                # 防止 MIME 嗅探
                headers.append((b"x-content-type-options", b"nosniff"))
                # XSS 过滤 (legacy 但无害)
                headers.append((b"x-frame-options", b"DENY"))
                headers.append((b"x-xss-protection", b"1; mode=block"))
                # HSTS (仅 prod + https). dev 环境不强制.
                cfg = get_settings()
                if cfg.app_env == "prod":
                    headers.append((b"strict-transport-security", b"max-age=31536000; includeSubDomains"))
                # CSP: 默认 self only; 允许 eval (一些前端库可能用) 但不允许 inline script.
                # 生产请根据实际前端静态资源域名收紧 script-src / style-src / connect-src.
                headers.append((
                    b"content-security-policy",
                    b"default-src 'self'; script-src 'self' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; connect-src 'self';",
                ))
                # 禁止在请求 / 响应中携带来源 URL, 降低 referrer 泄漏风险.
                headers.append((b"referrer-policy", b"strict-origin-when-cross-origin"))
                message["headers"] = headers
            await send(message)

        await self.app(scope, receive, _send)


app = FastAPI(
    title="PilKB - 通用企业级 RAG 知识库平台",
    description="Multi-Agent + MCP + Hybrid Search + Full-Stack",
    version=__version__,
    lifespan=lifespan,
)

# 安全: 严禁 allow_origins=["*"] 与 allow_credentials=True 同时使用 (违反 CORS 规范).
# origin 严格走 settings.cors_allowed_origins 白名单.
_cors_origins = _parse_cors_origins(get_settings().cors_allowed_origins)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 全局安全响应头
app.add_middleware(SecurityHeadersMiddleware)

app.include_router(health.router)
app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(kbs.router)
app.include_router(chat.router)
app.include_router(history.router)


if __name__ == "__main__":
    import uvicorn

    from app.core.config import settings

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=(settings.app_env == "dev"),
    )