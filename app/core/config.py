"""Application configuration loaded from environment variables.

.env.example 中列出的所有 key 都在这里集中映射, 业务代码通过 settings.xxx 访问。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# 项目根目录 (本文件位于 app/core/config.py, parents[2] 即根目录).
BACKEND_ROOT = Path(__file__).resolve().parents[2]
# 所有运行时数据统一落在项目根目录的 data/ 下.
#
# 允许用 PILKB_DATA_DIR 整体覆盖: `kb_registry.json` 这类文件不走
# `*_PERSIST_DIR` 系列配置, 而是直接拼在 DATA_DIR 上, 所以只重定向那几个
# PERSIST_DIR 是拦不住的 —— 跑一次 pytest 就会把开发者本地的知识库注册表
# 冲掉 (表现为重启后端后"知识库不存在", 但 bm25/vector 文件还在磁盘上)。
DATA_DIR = Path(os.environ.get("PILKB_DATA_DIR") or (BACKEND_ROOT / "data"))


def _to_abs_path(value: str) -> str:
    """把相对路径按项目根目录解析成绝对路径.

    `./data/chroma` 这类配置是相对**进程 CWD** 的: CWD 一变 (Docker WORKDIR 不同、
    systemd 启动、测试 monkeypatch.chdir), 数据就会落到意料之外的位置, 或者因为
    父目录不存在直接 IOError。这里统一锚定到项目根目录, 让 CWD 不再影响存储位置。
    """
    if not value:
        return value
    p = Path(value)
    if p.is_absolute():
        return str(p)
    return str((BACKEND_ROOT / p).resolve())


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ===== App =====
    app_env: Literal["dev", "prod", "test"] = "dev"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_log_level: str = "INFO"
    # 仅用于本地 / CI 演示，显式开启时不请求外部模型或加载 torch 模型。
    fake_llm: bool = Field(default=False, validation_alias="PILKB_FAKE_LLM")
    fake_embedding: bool = Field(default=False, validation_alias="PILKB_FAKE_EMBEDDING")

    # ===== LLM (三个 provider 可并存) =====
    llm_provider: Literal["deepseek", "qwen", "openai"] = "deepseek"

    deepseek_api_key: Optional[str] = None
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"

    qwen_api_key: Optional[str] = None
    qwen_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    qwen_model: str = "qwen-plus"

    openai_api_key: Optional[str] = None
    openai_base_url: str = "https://api.openai.com/v1"
    openai_model: str = "gpt-4o-mini"
    # 部分 OpenAI 兼容 provider (如 MiniMax) 需要在请求 body 里加私有字段
    # (例如 {"thinking": {"type": "disabled"}} 关掉 M3 的思考模式).
    # JSON 字符串, 解析失败或为空则不传. 与 provider 名无关, 避免在代码里写死分支.
    openai_extra_body: Optional[str] = None

    # 单次 LLM 请求的超时 (秒) 与 SDK 自动重试次数, 对所有 provider 生效.
    # 不设超时的话 openai SDK 默认最长挂 600s, 而 to_thread 卸载的线程不可取消,
    # provider 半死不活时线程池会被挂起调用占满.
    llm_request_timeout: float = Field(default=120.0, gt=0, le=600)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    # 兼容旧字段
    @property
    def default_llm_provider(self) -> str:
        return self.llm_provider

    @property
    def default_llm_model(self) -> str:
        return {
            "deepseek": self.deepseek_model,
            "qwen": self.qwen_model,
            "openai": self.openai_model,
        }[self.llm_provider]

    # ===== Embedding / Rerank =====
    # 后端: local (sentence-transformers 本地加载) | api (OpenAI 兼容的 /embeddings 接口).
    # 换后端 / 换模型等于换向量空间: bge-small-zh 是 512 维, bge-m3 是 1024 维,
    # 已有的向量 collection 必须重建, 否则查询会因维度不匹配直接报错.
    embedding_backend: Literal["local", "api"] = "local"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    embedding_device: str = "cpu"
    # --- embedding_backend=api 时使用 (示例: 硅基流动) ---
    # 注意 api 后端读的是 EMBEDDING_API_MODEL (不是上面的 EMBEDDING_MODEL):
    # local 用 bge-small-zh 省资源, api 用 bge-m3 拿长上下文 (1024 维 / 8192 token;
    # bge-small-zh 会把超过 512 token 的输入静默截断, 后半段内容进不了向量表示).
    embedding_api_model: str = "BAAI/bge-m3"
    embedding_api_base: str = "https://api.siliconflow.cn/v1"
    embedding_api_key: str = ""
    # 单次请求最多送几条文本 / 单条文本字符上限 (超长截断).
    embedding_api_batch: int = Field(default=16, ge=1, le=128)
    embedding_max_chars: int = Field(default=8000, ge=100, le=30000)

    # rerank 后端: local (CrossEncoder 本地加载) | api (Cohere 风格的 /rerank 接口).
    # 两者输出的分数量纲不同 (本地是原始 logits, 服务端是归一化到 0~1 的
    # relevance_score), 但都只用于同一批候选内部的相对排序, 不影响检索结果.
    rerank_backend: Literal["local", "api"] = "local"
    rerank_model: str = "BAAI/bge-reranker-base"
    # --- rerank_backend=api 时使用 ---
    # bge-reranker-v2-m3 与本地 bge-reranker-base 同系列, 服务端版本更大且支持
    # 多语言; 换 api 后端后本机不再需要 torch / sentence-transformers.
    rerank_api_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_api_base: str = "https://api.siliconflow.cn/v1"
    rerank_api_key: str = ""
    # 单条候选送进 rerank 前的字符上限 (本地 CrossEncoder 也会按模型 max_length 截断).
    rerank_api_max_chars: int = Field(default=2000, ge=100, le=30000)
    rerank_api_timeout: float = Field(default=60.0, gt=0, le=600)
    # 失败重试次数. 只对 429 / 5xx / 网络错误重试 —— 4xx 是请求本身的问题,
    # 重试只会把同一个错误再犯一遍.
    rerank_api_max_retries: int = Field(default=3, ge=1, le=10)

    # ===== Storage =====
    database_url: str = "sqlite:///./data/pilkb.db"
    redis_url: str = "redis://localhost:6379/0"
    chroma_persist_dir: str = "./data/chroma"
    # 上传文件落盘目录。默认 ./data/uploads —— 必须和上传链路历史上实际使用的
    # 目录保持一致, 否则改配置驱动的那一刻, 之前上传的文件会全部"找不到"
    # (原先 documents.py 里硬编码的就是 data/uploads)。改这个值前先迁移已有文件。
    file_storage_dir: str = "./data/uploads"
    # BM25 倒排索引的持久化目录 (文件名 bm25_{kb_id}.pkl).
    bm25_persist_dir: str = "./data"
    # 会话历史的持久化目录 (文件名 {user}.jsonl).
    history_storage_dir: str = "./data/history"
    # Agent 会话记忆 (LangGraph checkpointer) 的 SQLite 文件. 与上面的目录一样
    # 锚定到项目根: Docker 里落在 /app/data, 已被 ./data volume 覆盖, 不需要
    # 额外挂载. conftest 会把它重定向到临时目录, 测试互不污染.
    agent_checkpoint_db: str = "./data/agent_checkpoints.db"

    # 存储类路径一律锚定到项目根目录, 避免随 CWD 漂移 (见 _to_abs_path).
    _normalize_dirs = field_validator(
        "chroma_persist_dir", "file_storage_dir", "bm25_persist_dir", "history_storage_dir",
        "agent_checkpoint_db",
    )(_to_abs_path)
    # 向量库后端: chroma (默认, 生产推荐) | numpy (纯 Python, 用于 chroma 在当前平台无法运行的兜底).
    # chromadb 在 Windows + Anaconda Python 3.12 上会 ACCESS_VIOLATION, 此时可临时切 numpy.
    vector_store_backend: Literal["chroma", "numpy"] = "chroma"

    # ===== Auth =====
    jwt_secret: str = "change-me-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    # 登出/吊销后旧 token 的最长残留校验时间 (进程内缓存 TTL, 秒; 0 = 每次直查 DB)
    auth_token_cache_ttl_sec: float = 30.0

    # ===== Knowledge Base ACL =====
    # 公共只读知识库白名单. pydantic-settings 对 list 默认要求 JSON 格式,
    # 这里用 str, 由 kb_registry 自行解析 (支持逗号分隔或 JSON 数组).
    # 这些 kb_id 任何登录用户可读, 不可写入; 上传文档时若 kb 不存在会自动创建并归属当前用户.
    # 默认含 "default" 以兼容 demo.
    public_kb_ids: str = "default"

    # 管理员用户名白名单 (逗号分隔或 JSON 数组, 由 kb_registry 解析).
    # 用途: 公共 kb 的 BM25 索引重建 (POST /documents/rebuild-bm25). 公共 kb 不归
    # 任何用户所有, 重建又是全量读 + 落盘的 O(N) 操作, 因此不放给任意访客 ——
    # 私有 kb 的 owner 无需在此列 (owner 天然可重建自己的 kb).
    # 默认空 = 无人是管理员, 公共 kb 只能由运维手工调接口 (或用 ADMIN_USERS 授权).
    # 本地演示可设为 "dev-user" (dev-token 只在 APP_ENV=dev 颁发).
    admin_users: str = ""

    # ===== CORS =====
    # 允许跨域的前端 origin 白名单 (逗号分隔 / JSON 数组).
    # 严禁 "*" + credentials=True (违反 CORS 规范, 也是明显的安全反模式).
    # 多个 origin 用逗号分隔, 例如 "http://localhost:3000,https://app.example.com"
    cors_allowed_origins: str = "http://localhost:3000"

    # ===== Trusted Proxy =====
    # 反代场景下, 只有直连 peer 的 IP 在这份名单里 (逗号分隔; "*" = 全部信任,
    # 仅限本机调试) 才采信 X-Forwarded-For。默认留空 = 完全不信任 XFF, 用直连
    # 地址 —— 否则客户端可以伪造 XFF 换 IP, 绕过登录爆破锁定与 chat 限流的 ip 维度.
    trusted_proxies: str = ""

    # ===== Rate Limit =====
    # 同一账号 login 失败 5 次 / 5 分钟, 锁定 15 分钟 (返回 429).
    rate_limit_login_window_sec: int = 300
    rate_limit_login_fail_max: int = 5
    rate_limit_login_lock_sec: int = 900
    # chat 接口每用户每分钟请求上限 (默认 20).
    rate_limit_chat_per_min: int = 20
    # /auth/register 按 IP 维度的固定窗口 (register 无认证可依, 只能按 IP):
    # 不设限可以无限注册填爆 users 表.
    rate_limit_register_per_min: int = 5
    # /documents/upload 按用户维度的固定窗口: 单文件大小上限挡不住海量小文件.
    rate_limit_upload_per_min: int = 10
    # 是否启用限流 (测试时可关掉以免 flaky)
    rate_limit_enabled: bool = True

    # 每用户可创建的私有 kb 数上限 (ensure_kb_for_write 自动建 kb 时校验):
    # 每个 kb 都会在 retriever 实例缓存里常驻一份 BM25 docs + collection,
    # 无上限可被批量建 kb 耗尽内存.
    max_kbs_per_user: int = Field(default=10, ge=1)

    # ===== MCP =====
    mcp_enabled: bool = True
    mcp_servers_config: str = "./mcp_servers.json"
    mcp_transport: str = "stdio"
    mcp_server_name: str = "pilkb"

    # ===== Observability =====
    langfuse_enabled: bool = False
    langfuse_public_key: Optional[str] = None
    langfuse_secret_key: Optional[str] = None
    langfuse_host: str = "https://cloud.langfuse.com"

    # ===== RAG 切分 / 检索参数 =====
    chunk_size: int = Field(default=500, ge=100, le=2000)
    chunk_overlap: int = Field(default=80, ge=0, le=400)
    top_k_vector: int = 20
    top_k_bm25: int = 20
    top_k_final: int = 5
    # 每条召回结果拼进 LLM 上下文的字符上限, RAG 主链路 / 评估链路 / Agent 工具
    # 三条路径共用, 保证行为一致。
    #
    # 这个值原先在三处各写死一次 (tools.py 300, chat.py 600, pipeline.py 600),
    # 都小于默认 adaptive_chunk_size, 于是长 chunk 的后半段 —— 很多文档正好把
    # "定价 / 结论 / 注意事项" 放在文末 —— 会被静默切掉。后果不是报错, 而是模型
    # 含糊其辞: Agent 链路答"未找到相关信息"、calculator 拿不到数字所以永不触发,
    # 而 RAG 主链路又会抱怨"原文此处被截断"。这类"答案似是而非"的问题排查成本
    # 极高, 因为链路全是 200 OK。
    context_snippet_chars: int = Field(default=2000, ge=200, le=20000)

    # ===== RAG 质量调优 (来自 RAG 知识库工程实践) =====
    # 1) 文档清洗: 占 RAG 质量调优 ~60% 精力, 42 条正则规则去除噪声.
    cleaner_enabled: bool = True
    # 2) 自适应切分: 短文档整篇保留为 1 个 chunk; 长文档按 adaptive_chunk_size 切.
    adaptive_chunking_enabled: bool = True
    adaptive_chunk_size: int = Field(default=500, ge=100, le=8000)
    adaptive_short_doc_chars: int = Field(default=500, ge=100)
    # 3) 元数据增强 / HyDE: 入库存假设性问题 (hypothetical question) 提升召回.
    hyde_enabled: bool = True
    # 入库侧 HyDE 是"每 chunk 一次 LLM 调用", 大文档会把上传拖成分钟级并烧 token.
    # 因此只对前 N 个 chunk 走 LLM, 其余回退抽取式首句 (metadata 仍完整, 只是弱一些);
    # 设为 0 表示入库侧完全不调 LLM (纯离线入库).
    hyde_index_max_chunks: int = Field(default=20, ge=0, le=500)
    # 这 N 个 chunk 的 LLM 调用并发度, 避免串行等待也避免打爆 provider 限流.
    hyde_index_concurrency: int = Field(default=4, ge=1, le=32)
    # 查询侧 HyDE: 把用户问题扩写成一段"假设答案文档", 作为额外一路 query 参与
    # 召回 (与入库侧互补 —— 改写做词面扩展, HyDE 做语义空间对齐, 用户问句与文档
    # 陈述句式差异大时收益最明显). 默认关闭: 每次问答多一次 LLM 调用.
    hyde_query_enabled: bool = False
    # 4) 父文档召回 (连坐召回): 召回某 chunk 时一并召回同 doc 的兄弟 chunk.
    parent_recall_enabled: bool = True
    parent_recall_extra: int = Field(default=6, ge=0, le=32)
    # 5) 重排优化: 候选池 Top-60 -> 重排后 Top-25 (控制 Cross-Encoder 算力).
    rerank_candidate_pool: int = Field(default=60, ge=10, le=200)
    rerank_top_n: int = Field(default=25, ge=5, le=100)
    # 6) 查询改写 (Query Rewriting): LLM 改写/拆解问题提升召回.
    query_rewrite_enabled: bool = True

    # ===== Agent =====
    # agent 引擎: react = tool-calling 循环 (默认, 模型自主组织检索词与工具序列,
    # 结果不相关时换词重试); router = 旧的一次性路由 (LLM 输出三个布尔再确定性
    # 执行), 保留作快路径与消融对照 (tests/run_agent_ablation.py 按此开关对比).
    agent_engine: Literal["react", "router"] = "react"
    # react 引擎的工具调用轮数上限 (一轮 = 一次 reason 决策 + 一次工具执行).
    # 超出后不再询问模型, 直接用已收集的证据生成答案 —— 预算耗尽必须兜底,
    # 否则一次跑飞的循环会把 LLM 配额与请求超时一起烧穿.
    agent_max_steps: int = Field(default=4, ge=1, le=10)
    # 会话记忆: LangGraph checkpointer (SQLite 持久化, 见 agent_checkpoint_db),
    # thread_id = "{user}:{session}" 隔离. 需要 session_id 才真正启用; 关闭或
    # 无 session 时与单轮行为一致.
    agent_memory_enabled: bool = True
    # 喂给 LLM 的历史消息条数上限 (含本轮): 更早的消息留在 checkpoint, 不进上下文.
    agent_memory_max_messages: int = Field(default=20, ge=2, le=100)

    # ===== Upload =====
    # 单文件最大字节数, 防止恶意上传把磁盘填爆或拖死 embedding.
    # 默认 25MB, 可通过环境变量调整. 超出时返回 413.
    upload_max_bytes: int = Field(default=25 * 1024 * 1024, ge=1024)

    def get_llm_api_key(self, provider: str) -> Optional[str]:
        return {
            "deepseek": self.deepseek_api_key,
            "qwen": self.qwen_api_key,
            "openai": self.openai_api_key,
        }.get(provider)

    def get_llm_base_url(self, provider: str) -> str:
        return {
            "deepseek": self.deepseek_base_url,
            "qwen": self.qwen_base_url,
            "openai": self.openai_base_url,
        }.get(provider, self.openai_base_url)

    def get_llm_model(self, provider: str) -> str:
        return {
            "deepseek": self.deepseek_model,
            "qwen": self.qwen_model,
            "openai": self.openai_model,
        }.get(provider, self.openai_model)


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
