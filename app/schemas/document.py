"""Document schemas.

文档上传 / 列表 / 删除三类接口的请求与响应模型定义.
"""
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


class DocumentOut(BaseModel):
    """list 接口单条文档的元信息."""

    doc_id: str = Field(..., description="文档的唯一 ID, 格式形如 doc_<12位hex>")
    filename: str = Field(..., description="上传时的原始文件名")
    kb_id: str = Field(..., description="归属的知识库 ID")
    chunks: int = Field(..., description="文档切分后得到的 chunk 总数")
    status: str = Field(..., description="处理状态; 入库成功时为 'indexed'")
    created_at: datetime = Field(..., description="入库时间 (UTC 时区)")
    metadata: dict = Field(default={}, description="附加元数据 (预留的扩展位)")


class DocumentUploadResponse(BaseModel):
    """upload 接口成功时的响应体."""

    doc_id: str = Field(..., description="由后端新分配的文档 ID")
    filename: str = Field(..., description="上传时的原始文件名")
    chunks: int = Field(..., description="实际切分得到的 chunk 数量")
    status: str = Field(default="indexed", description="恒为 'indexed'")
    message: Optional[str] = Field(default=None, description="供人阅读的提示信息")
