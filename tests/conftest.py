"""共享 fixtures: 把 app 加入 sys.path 让 import 跑得通."""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

import pytest

# 让 `from app.xxx import ...` 跑得通, 不依赖 src layout
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# pytest_configure() 里建出来的临时数据目录, 会话结束时清理.
_TMP_DATA_ROOT: str | None = None


def _cleanup_tmp_data_root() -> None:
    global _TMP_DATA_ROOT
    if _TMP_DATA_ROOT and os.path.isdir(_TMP_DATA_ROOT):
        shutil.rmtree(_TMP_DATA_ROOT, ignore_errors=True)
    _TMP_DATA_ROOT = None


def pytest_configure(config):  # noqa: ARG001
    """把测试用到的持久化目录全部重定向到临时目录.

    为什么必须在 pytest_configure 里做:
      `app.core.config` 在模块级就 `settings = get_settings()`, 而
      `app.db.session` 又在模块级按 settings 建 engine. 两者都发生在
      **第一次 import app.* 时**, 也就是测试模块被收集/导入的那一刻.
      pytest_configure 早于收集, 所以这里是唯一能赶在 app 被 import 之前
      把环境变量铺好的钩子.

    不隔离的后果 (都是真实踩过的):
      1. 测试直接写开发者的 `data/pilkb.db`, 每次跑测试都往里塞
         alice/bob/... 这些测试账号;
      2. 反过来, 测试还依赖"库是干净的"——同一个文件第二次跑就会因为
         "用户已存在" 全线 400, 单测不可重复执行;
      3. bm25/chroma/uploads 的 pkl 和上传文件同样散落在 data/.

    逃生口: 显式设置 PILKB_TEST_DATA_DIR 或对应的 *_DIR / DATABASE_URL
    环境变量时, 尊重调用方的选择, 不覆盖.
    """
    global _TMP_DATA_ROOT

    override_root = os.environ.get("PILKB_TEST_DATA_DIR")
    data_root = Path(override_root) if override_root else Path(tempfile.mkdtemp(prefix="pilkb_testdata_"))
    data_root.mkdir(parents=True, exist_ok=True)

    if not override_root:
        _TMP_DATA_ROOT = str(data_root)
        atexit.register(_cleanup_tmp_data_root)

    def _setdefault(env_key: str, value: str) -> None:
        # 已显式指定的不动; 否则指到临时目录.
        os.environ.setdefault(env_key, value)

    # sqlite 用正斜杠, 避免 Windows 反斜杠在 URL 里被当转义.
    db_path = (data_root / "pilkb_test.db").as_posix()
    _setdefault("DATABASE_URL", f"sqlite:///{db_path}")
    # 整棵数据目录一起搬走: kb_registry.json 不走下面的 *_PERSIST_DIR 配置,
    # 而是直接挂在 DATA_DIR 上。只搬那几个子目录的话, 测试仍会写开发者真实的
    # data/kb_registry.json, 一次 pytest 就把本地知识库注册表冲掉。
    _setdefault("PILKB_DATA_DIR", str(data_root))
    for key, sub in (
        ("FILE_STORAGE_DIR", "files"),
        ("CHROMA_PERSIST_DIR", "chroma"),
        ("BM25_PERSIST_DIR", "bm25"),
        ("HISTORY_STORAGE_DIR", "history"),
    ):
        _setdefault(key, (data_root / sub).as_posix())
    # agent 记忆的 SQLite checkpoint: 不重定向的话, 固定 thread_id 的记忆测试
    # (test_agent_memory 的 "alice:sess1" 等) 会把上一轮 pytest 的 checkpoint
    # 留在开发者的 data/ 里, 下一次跑 "初始 checkpoint 为空" 的断言全挂,
    # 而且真实后端的会话记忆被测试数据污染.
    _setdefault("AGENT_CHECKPOINT_DB", (data_root / "agent_checkpoints.db").as_posix())

    # 测试会话统一走 chroma 分支: client fixture patch 的是 ChromaStore. 本地
    # .env 常设 VECTOR_STORE_BACKEND=numpy (Windows chromadb 崩溃的 workaround),
    # 若让它渗进来, "fake" 测试会在背后悄悄构建**真实** NumpyVectorStore, 把
    # chunk 持久化到上面的会话级共享目录 —— 同名 kb 的跨测试残留让计数类断言
    # 变成顺序依赖 (实测: kb_alice 的 vector_count 在某个前序测试后永远非 0).
    # 显式导出该变量的调用方不受影响 (setdefault 尊重已有值); 需要 numpy 后端
    # 的测试直接 monkeypatch settings 单例属性, 与本项无关.
    _setdefault("VECTOR_STORE_BACKEND", "chroma")

    # 同一泄漏模式的第二例: 本地 .env 常设 RERANK_BACKEND=api (免装本地模型),
    # 渗进测试后 get_reranker() 返回 APIReranker —— test_reranker_cache 只 mock
    # sentence_transformers, CrossEncoder 构造计数恒为 0 而失败; 更糟的是
    # rerank() 会拿 .env 里的真实 key 打线上请求 (计费 + flaky)。测试统一走
    # local 分支; 需要 api 后端的测试显式导出该变量即可 (setdefault 尊重已有值)。
    _setdefault("RERANK_BACKEND", "local")


def pytest_unconfigure(config):  # noqa: ARG001
    _cleanup_tmp_data_root()


@pytest.fixture(scope="session", autouse=True)
def _ensure_db_schema():
    """保证测试库的 schema 就绪, 让每个测试文件都能独立跑.

    为什么要这个 fixture:
        `create_tables()` 只在 FastAPI 的 lifespan 里被调用, 而 `TestClient(app)`
        **必须作为上下文管理器使用才会触发 lifespan** (`with TestClient(app) as c`)。
        写成 `return TestClient(app)` 的 fixture 不会建表 —— 这类测试之所以平时
        能过, 纯粹是因为同一次 pytest 会话里前面某个文件用过 `with TestClient(app)`,
        其副作用把表建好了。于是测试之间产生了隐式顺序依赖: 单独跑该文件、
        或调整文件收集顺序, 就会全线 `OperationalError: no such table: users`
        (test_upload_resilience.py 单独跑 4 个全挂, 全量跑只挂 3 个, 就是这么来的)。

        建 schema 属于测试基础设施职责, 不该依赖"某个测试恰好触发了应用启动",
        所以在这里统一做一次, 一次性解开所有测试文件的这层耦合。
    """
    from app.db import create_tables

    create_tables()
    yield


@pytest.fixture(autouse=True)
def _isolate_process_state():
    """按测试隔离进程级单例/缓存, 解开"结果取决于收集顺序"的耦合.

    - rate_limit: 限流状态是进程级 dict. 不清的话, 前面测试用掉的配额会挤占
      后面的测试 (全量跑时 test_kb_acl / test_chat_sse_contract 的 register
      会被前面文件的注册挤成 429, 单独跑却通过).
    - retriever._INSTANCE_CACHE: HybridRetriever 持有 store/embedding 实例,
      跨测试残留会让某个测试读到上一个测试环境创建的实例 (test_document_delete
      还把 reset patch 成 no-op, 残留问题在那里被放大). 清掉后每个测试从
      自己的环境重新构建; embedding/reranker 模型有各自的单例缓存, 不会重载.
    - agents.graph._GRAPH_CACHE: 带 checkpointer 的图持有 AsyncSqliteSaver,
      其 asyncio.Lock 绑定首次使用它的 loop —— pytest-asyncio 每个用例一个
      新 loop, 缓存的图跨用例复用会 RuntimeError "bound to a different event
      loop". 清掉后各用例拿全新 saver (checkpoint 在 sqlite 里, 不受影响).
    """
    from app.core import rate_limit as _rl
    from app.rag import retriever as _rt

    _rl.reset_for_tests()
    _rt.reset_retriever_cache()
    from app.agents.graph import reset_agent_graph_cache as _reset_graph

    _reset_graph()
    yield
    _rt.reset_retriever_cache()
    _reset_graph()


@pytest.fixture(scope="session")
def settings():
    """从 .env.example 强制加载一份测试用 settings."""
    from app.core.config import Settings

    return Settings(_env_file=ROOT / ".env.example")
