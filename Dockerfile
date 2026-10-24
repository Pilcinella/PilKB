# syntax=docker/dockerfile:1.6
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System deps: build-essential for chromadb/sentence-transformers wheels on slim;
# poppler-utils for PDF parsing; tesseract for OCR.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        poppler-utils \
        tesseract-ocr \
        tesseract-ocr-chi-sim \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt \
    && pip install "pydantic[email]"

COPY app ./app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

# --workers 1 是刻意选择, 不是保守取值. 这套存储是"进程内内存态 + 本地文件"的组合:
#
#   1) BM25 的 docs 常驻内存 (每个进程一份), 每次写入把**整表**覆盖到
#      <bm25_persist_dir>/bm25_<kb>.pkl. 多 worker 下 A 上传 X、B 上传 Y 各自覆盖,
#      磁盘上只剩 Y —— 丢更新是静默的, 重启后 BM25 永久缺文档.
#      实测: 生产镜像默认的 2 worker 每轮冷启动都有概率触发.
#   2) vector_store 的引导锁 _CHROMA_INIT_LOCK 是 threading.Lock, **跨进程无效**,
#      而 Chroma 走的是本地 sqlite (PersistentClient). 实测 2 进程同时冷启动 8 轮
#      有 1 轮出现某个 worker 持续不可用且 upsert 丢失; 4 进程时 5 轮里有 3 轮
#      大面积失败 (OperationalError / duplicate column name / StopIteration).
#
# 要恢复多 worker, 得先改存储形态 (BM25 外置, 或 Chroma 起独立服务端 + 每 worker
# 独立 BM25 文件并按需从向量库重建), 而不是直接把这个数字调大.
# main.py 的启动自检会在两份存储数量不一致时打出 error 日志.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]