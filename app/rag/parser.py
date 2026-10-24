"""文档解析器 (Document Parser), 位于入库链路最前端.

`api/documents.py` 的 upload 流程中, 用户上传的原始文件必须先转成纯文本,
才能进入后续的 cleaner -> chunker -> embedding -> 存储环节; 本模块承担的正是
这一步:

    支持的格式 (详见 `SUPPORTED`):
      - .pdf   : pypdf 逐页抽取文本 (空页跳过), 页与页之间以空行衔接.
      - .md/.markdown/.txt : UTF-8 直读全文 (errors="ignore" 容忍坏字节).
      - .docx  : python-docx 按 paragraph 抽取, 段落间同样以空行衔接.
      - .png/.jpg/.jpeg : OCR 路径 (pytesseract, 中英文 chi_sim+eng);
                          系统未装 tesseract 时降级返回空串, 不使整个上传失败.

设计要点:
  - 解析属于**纯 IO / CPU 操作**, 调用方 (documents.py) 通过 `asyncio.to_thread`
    将其卸载到线程池执行, 以免阻塞 FastAPI 事件循环.
  - 单页/单段落级别的解析异常一律吞掉并按空处理, 不让个别坏页连累整篇文档.
  - 后缀不在支持范围时直接抛 ValueError, 由上游转换为 400 返回.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict

from loguru import logger


class DocumentParser:
    # 允许入库的扩展名白名单; documents.py 保存前会先校验后缀是否命中该集合.
    SUPPORTED = {".pdf", ".md", ".markdown", ".txt", ".docx", ".png", ".jpg", ".jpeg"}

    def parse(self, file_path: str) -> str:
        """依据后缀分派至对应的私有解析方法, 返回拼合后的纯文本.

        Returns:
            纯文本 (页/段之间以 "\\n\\n" 相连). 全部解析失败时返回 "".
        Raises:
            ValueError: 后缀不在 SUPPORTED 集合中.
        """
        p = Path(file_path)
        suffix = p.suffix.lower()
        if suffix not in self.SUPPORTED:
            raise ValueError(f"Unsupported file type: {suffix}")
        try:
            if suffix == ".pdf":
                return self._parse_pdf(p)
            if suffix in {".md", ".markdown"}:
                return p.read_text(encoding="utf-8", errors="ignore")
            if suffix == ".txt":
                return p.read_text(encoding="utf-8", errors="ignore")
            if suffix == ".docx":
                return self._parse_docx(p)
            if suffix in {".png", ".jpg", ".jpeg"}:
                return self._parse_image_ocr(p)
        except Exception as e:
            logger.exception(f"Parse failed: {file_path}: {e}")
            raise
        return ""

    def _parse_pdf(self, p: Path) -> str:
        """PDF 解析: pypdf 逐页调用 extract_text, 抽不出文字的页 (如纯图片页) 直接跳过.

        页与页以 "\\n\\n" 相连, 保留基本的段落/页边界, 便于 cleaner 后续处理.
        """
        from pypdf import PdfReader

        reader = PdfReader(str(p))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                texts.append(t)
        return "\n\n".join(texts)

    def _parse_docx(self, p: Path) -> str:
        """DOCX 解析: python-docx 遍历 paragraph 取文本, 空段落跳过.

        段落间以 "\\n\\n" 相连, 文档的结构层级得以保留 (标题/正文同为 paragraph).
        """
        from docx import Document

        doc = Document(str(p))
        return "\n\n".join(par.text for par in doc.paragraphs if par.text.strip())

    def _parse_image_ocr(self, p: Path) -> str:
        """图片 OCR: pytesseract 中英文识别 (lang="chi_sim+eng").

        前提是系统装有 Tesseract OCR 引擎; 缺失或导入失败时打印 warning 并返回
        空串 (选择降级而非抛错), 保证"图片解析不了"不至于让上传链路 500.
        """
        try:
            import pytesseract
            from PIL import Image

            img = Image.open(str(p))
            return pytesseract.image_to_string(img, lang="chi_sim+eng")
        except Exception as e:
            logger.warning(f"OCR unavailable: {e}; return empty")
            return ""
