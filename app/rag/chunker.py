"""Recursive text chunker (LangChain-style)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from app.core.config import settings


@dataclass
class Chunk:
    text: str
    metadata: dict


class TextChunker:
    """依段落 / 句子逐级递归切分, 相邻块之间保留 overlap.

    overlap 的两条实现规则:
    1. 常规打包路径 (靠 `buf` 累积) -- 切出当前 chunk 后, 下一轮 buf 不清零,
       而是从上一 chunk 尾部截 `chunk_overlap` 个字符起步, 使相邻 chunk
       真正交叠, 边界附近的召回因此受益.
    2. 硬切路径 (单段超过 chunk_size) -- 窗口以 `chunk_size - chunk_overlap`
       为 step 推进, 硬切出的相邻片段同样保有 overlap.

    另外: `_recursive_split` 返回的每段末尾都附着 separator, 例如 "句子。".
    常规路径上 overlap 边界可能落在 separator 内部, 但相邻 chunk 仍会交叠
    若干字符; 硬切路径没有 separator, overlap 是严格的.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
        adaptive: bool = False,
        adaptive_chunk_size: int | None = None,
        adaptive_short_doc_chars: int | None = None,
    ):
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap
        # overlap 一旦不小于 chunk_size, 硬切路径的循环将无法收敛, 此处强制收窄.
        if self.chunk_overlap >= self.chunk_size:
            self.chunk_overlap = max(0, self.chunk_size // 4)
        self.separators = ["\n\n", "\n", "。", "！", "？", ". ", "! ", "? ", " ", ""]
        # 自适应切分参数 (源自 RAG 工程实践):
        #   短文档 (len <= adaptive_short_doc_chars) 整篇并成 1 个 chunk,
        #   免做无意义的碎切; 长文档换用更大的 adaptive_chunk_size,
        #   既压低 chunk 总数, 又留住更长的上下文.
        self.adaptive = adaptive
        self.adaptive_chunk_size = adaptive_chunk_size or settings.adaptive_chunk_size
        self.adaptive_short_doc_chars = (
            adaptive_short_doc_chars or settings.adaptive_short_doc_chars
        )

    def split(self, text: str, doc_id: str = "") -> List[Chunk]:
        text = (text or "").strip()
        if not text:
            return []
        # 自适应模式: 短文档整篇保留, 长文档放大 chunk_size 后再切.
        if self.adaptive:
            if len(text) <= self.adaptive_short_doc_chars:
                chunk = Chunk(text=text, metadata={"doc_id": doc_id})
                chunk.metadata["chunk_id"] = f"{doc_id}_c0"
                chunk.metadata["chunk_index"] = 0
                return [chunk]
            # 长文档: 先临时将 chunk_size 抬到 adaptive_chunk_size, 再复用常规递归切分.
            saved = self.chunk_size
            self.chunk_size = self.adaptive_chunk_size
            try:
                return self._split_long(text, doc_id)
            finally:
                self.chunk_size = saved
        # 非自适应模式 (默认): 直接进入常规递归切分.
        return self._split_long(text, doc_id)

    def _split_long(self, text: str, doc_id: str = "") -> List[Chunk]:
        """常规递归切分, 自适应长文档与默认模式共用; overlap 规则见类 docstring."""
        splits = self._recursive_split(text, self.separators)
        # 剔除只剩分隔符的"幽灵段" (递归过程会产出 "。" 之类的空壳).
        splits = [s for s in splits if s.strip()]
        chunks: List[Chunk] = []
        buf = ""
        for s in splits:
            if len(buf) + len(s) <= self.chunk_size:
                # 1) buf 还装得下 s: 继续累积.
                buf = (buf + s).strip()
                continue
            # 2) 装不下: 先把 buf 切出去.
            if buf:
                chunks.append(Chunk(text=buf, metadata={"doc_id": doc_id}))
                # 关键点: 下一轮 buf 以上一 chunk 的尾部 overlap 为起点, 而非清空.
                buf = self._tail(buf, self.chunk_overlap)
            # 3) 此时 buf 即上一 chunk 的尾部 overlap, 再尝试吞入 s.
            if len(s) <= self.chunk_size - len(buf):
                buf = (buf + s).strip()
                continue
            # 4) s 单独已超长, 走硬切: 将 buf 前置拼到 s 上, 使"上一 chunk 末尾"
            #    与"s 的开头"同样交叠.
            piece = (buf + s) if buf else s
            buf = ""
            step = max(1, self.chunk_size - self.chunk_overlap)
            # 按窗口逐片推进, 每片都不超过 chunk_size.
            for i in range(0, len(piece), step):
                slice_text = piece[i : i + self.chunk_size]
                if not slice_text.strip():
                    continue
                chunks.append(Chunk(text=slice_text, metadata={"doc_id": doc_id}))
            # 硬切完毕, 将最后一片的尾部 overlap 留作下一段 buf 的起点.
            if chunks:
                buf = self._tail(chunks[-1].text, self.chunk_overlap)
        if buf and buf.strip():
            # 防 hard-cut 残留: buf 本就是上一 chunk 的尾部 overlap, 已被其末尾覆盖,
            # 不必再 emit 一个只含 overlap 内容的重复 chunk.
            if chunks and chunks[-1].text.endswith(buf):
                pass
            else:
                chunks.append(Chunk(text=buf.strip(), metadata={"doc_id": doc_id}))
        # 统一注入 chunk_id
        for i, c in enumerate(chunks):
            c.metadata["chunk_id"] = f"{doc_id}_c{i}"
            c.metadata["chunk_index"] = i
        return chunks

    @staticmethod
    def _tail(s: str, n: int) -> str:
        """取 s 的最后 n 个字符; n<=0 时给空串."""
        if n <= 0 or not s:
            return ""
        return s[-n:]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators or len(text) <= self.chunk_size:
            return [text]
        sep = separators[0]
        rest = separators[1:]
        if sep == "":
            # 兜底: 逐字符等宽切
            return [text[i : i + self.chunk_size] for i in range(0, len(text), self.chunk_size)]
        parts = text.split(sep)
        out: List[str] = []
        for idx, p in enumerate(parts):
            # 分隔符随段保留 (最后一段后面没有分隔符, 除外), buf 拼接后才能还原原文.
            is_last = idx == len(parts) - 1
            if is_last:
                p_with_sep = p  # 末段不补 sep, 防递归时重复追加 separator 生出幽灵段.
            else:
                p_with_sep = p + sep
            if not p_with_sep:
                continue
            if len(p_with_sep) <= self.chunk_size:
                out.append(p_with_sep)
            else:
                out.extend(self._recursive_split(p_with_sep, rest))
        return out