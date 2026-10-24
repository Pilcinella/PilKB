"""文档清洗 (Document Cleaning).

RAG 落地中被反复验证的一条规律: **六到七成的调优精力耗在清洗上**, 检索与
重排只占零头. 解析环节拿到的原始文本噪声密布——控制字符、零宽字符、页眉
页脚、脚注、OCR 碎片、Markdown/LaTeX 语法残留——原样切块入库会同时污染
向量表示、拖垮召回率。

本模块以 **42 条正则规则** 实现纯确定性清洗 (不调 LLM, 结果可复现、零成本):
  1. 不可见字符: NUL 与各类控制字符、零宽空格/连字、BOM、软连字符
  2. 排版噪声: 冗余空白与换行、行尾空格、Tab 混用
  3. 标记语言残留: HTML 标签, Markdown 链接/图片/加粗, LaTeX 命令与数学环境
  4. 结构性噪声: 页眉页脚 (`第 N 页`), 脚注标记 (`¹`/`[1]`), 目录点线 (`.....`)
  5. 半结构化噪声: 长分隔线、孤立标点行、全角空格、URL/邮箱归一化

清洗过程**幂等**: 同一文本无论 clean 多少遍, 输出保持一致。
"""
from __future__ import annotations

import re
from typing import List

# ---------------------------------------------------------------------------
# 1. 不可见与控制字符
# ---------------------------------------------------------------------------
# NUL 及 C0/C1 控制字符 (\t \n \r 三个排版用字符除外)
_RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
# 零宽类字符: 零宽空格, 零宽/词连接符, BOM, 行与段落分隔符
_RE_ZERO_WIDTH = re.compile(
    "[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e\u00ad\ufffc]"
)
# 方向/零宽标记的补充规则:\u200e \u200f 为 LTR/RTL 方向符, 会干扰中文分词, 必须剔除;
# \u200b 已在 _RE_ZERO_WIDTH 中覆盖, 这里重复匹配一次是双保险, 无副作用.
_RE_ZWSP_VARIANT = re.compile("[\u200b\u200e\u200f]")

# ---------------------------------------------------------------------------
# 2. HTML / XML 痕迹
# ---------------------------------------------------------------------------
# 留意 negative lookahead 的用意: 跳过清洗器自己生成的 <URL> / <EMAIL> 占位符.
# 不排除的话, 同一文本 clean 到第二遍时占位符会被误认成 HTML 标签删掉, 幂等性随之破坏.
_RE_HTML_TAG = re.compile(r"<(?!URL>|EMAIL>)[^>]+>")
_RE_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_RE_HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_CDATA = re.compile(r"<!\[CDATA\[.*?\]\]>", re.DOTALL)

# ---------------------------------------------------------------------------
# 3. Markdown / 富文本痕迹
# ---------------------------------------------------------------------------
_RE_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")  # 链接 [文本](url) 只留文本
_RE_MD_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")  # 图片 ![alt](url) 整体删除
_RE_MD_BOLD_ITALIC = re.compile(r"[*_]{1,3}([^*_]+)[*_]{1,3}")  # **x** -> x
_RE_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s+", re.MULTILINE)  # 行首标题井号剥掉
_RE_MD_BLOCKQUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_RE_MD_HR = re.compile(r"^\s{0,3}[-*_]{3,}\s*$", re.MULTILINE)
_RE_MD_CODE_FENCE = re.compile(r"```.*?```", re.DOTALL)
_RE_MD_INLINE_CODE = re.compile(r"`([^`]+)`")
_RE_MD_TABLE_SEP = re.compile(r"^\s*\|?[\s:|-]+\|?\s*$", re.MULTILINE)  # |---|---|
_RE_MD_TABLE_PIPE = re.compile(r"(?<!\S)\|(?!\S)")  # 孤立的表格竖线

# ---------------------------------------------------------------------------
# 4. LaTeX 痕迹
# ---------------------------------------------------------------------------
_RE_LATEX_ENV = re.compile(r"\\begin\{[^}]*\}.*?\\end\{[^}]*\}", re.DOTALL)
_RE_LATEX_CMD = re.compile(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?")
_RE_LATEX_MATH = re.compile(r"\$[^$]*\$")  # 单美元包裹的行内公式
_RE_LATEX_DISPLAY = re.compile(r"\$\$.*?\$\$", re.DOTALL)  # 双美元包裹的块级公式
# 仅匹配"整行注释", 即 `%` 位于行首 (允许前面有空白) 的行。
#
# 刻意不写成 r"%.*$": 那会让**行内任何** `%` 到行尾的内容都被连带删除。业务
# 文档中 "按原价 85% 计费；购买 20 个账号……" 之类的百分数随处可见, 这条规则
# 一旦放宽, 整段商务条款会被无声抹去——不报错、不告警, 症状只是检索时 chunk
# 里的句子断在半截、模型答"原文此处被截断", 而排查时链路清一色 200。
# 因此行内 `%` 全部保留, 代价是漏掉少量 `code % comment` 形式的 LaTeX 注释。
_RE_LATEX_COMMENT = re.compile(r"^[ \t]*(?<!\\)%.*$", re.MULTILINE)

# ---------------------------------------------------------------------------
# 5. 结构噪声: 页眉页脚 / 脚注 / 目录
# ---------------------------------------------------------------------------
_RE_PAGE_NUM = re.compile(r"^\s*(?:第\s*\d+\s*页\s*(?:[/-]\s*\d+\s*页)?|page\s*\d+\s*(?:of\s*\d+)?)\s*$", re.IGNORECASE | re.MULTILINE)
_RE_FOOTNOTE_MARK = re.compile(r"(?<=\S)[¹²³⁴⁵⁶⁷⁸⁹⁰⑩]")  # 上标式脚注符号
_RE_FOOTNOTE_BRACKET = re.compile(r"\[\d{1,3}\](?=\s|$)")  # 方括号脚注 [12]
_RE_TOC_DOTS = re.compile(r"\.{4,}\s*\d+\s*$", re.MULTILINE)  # 目录引导点 `.....12` (re.M 使匹配落到每行行尾, 否则只命中文末)
_RE_SECTION_NUM = re.compile(r"^\s*\d+(?:\.\d+)*\s+(?=\S)", re.MULTILINE)  # 行首章节编号 `1.2.3 ` (re.M 逐行生效, 否则只作用于全文首行)

# ---------------------------------------------------------------------------
# 6. 标点与空白归一
# ---------------------------------------------------------------------------
_RE_TRAILING_SPACE = re.compile(r"[ \t]+$")  # 行尾空白
_RE_MULTI_SPACE = re.compile(r"[ \t]{2,}")  # 连续空格折叠为一个
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")  # 连续空行压缩
_RE_LEADING_SPACE_LINE = re.compile(r"^[ \t]+", re.MULTILINE)
_RE_FULLWIDTH_SPACE = re.compile(r"[\u3000]")  # 全角空格转半角
_RE_MULTI_DOT = re.compile(r"\.{4,}")  # 四个以上的连续点号
_RE_ISOLATED_PUNCT = re.compile(r"(?<=\n)[，。、；：！？,.!?;:]+\s*$", re.MULTILINE)  # 整行只剩标点 (re.M 逐行匹配行尾, 否则只看最后一行)

# URL 与邮箱统一成占位符 (同一链接的不同写法不再各自污染 chunk)
_RE_URL = re.compile(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", re.IGNORECASE)
_RE_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")

# 长分隔线 (--- / === / ***)
_RE_SEPARATOR_LINE = re.compile(r"^\s*[-=*_]{5,}\s*$", re.MULTILINE)

# 空括号对 / 空引用
_RE_EMPTY_PAREN = re.compile(r"\(\s*\)|\（\s*\）")
_RE_BROKEN_FRAGMENT = re.compile(r"[•·▪◦]+\s*$")  # 悬在行尾的项目符号

# 中文标点前的空白 (HTML 标签替换成空格后, 常见残留形如 "文字 ，")
_RE_SPACE_BEFORE_CJK_PUNCT = re.compile(r"[ \t]+(?=[，。、；：！？）】》」』])")
_RE_TAB = re.compile(r"\t+")


class DocumentCleaner:
    """确定性清洗器: 42 条正则规则, 幂等, 不依赖 LLM。

    用法:
        text = DocumentCleaner().clean(raw_text)
    """

    # 规则次序有讲究: 先剥离标记语言与结构噪声, 再做空白归一, 最后统一兜底。
    _STEPS: List[tuple] = [
        # ---- 标记语言与结构 ----
        (_RE_HTML_COMMENT, ""),
        (_RE_CDATA, ""),
        (_RE_HTML_TAG, " "),
        (_RE_HTML_ENTITY, " "),
        (_RE_MD_CODE_FENCE, " "),
        (_RE_MD_IMAGE, ""),
        (_RE_MD_LINK, r"\1"),
        (_RE_MD_INLINE_CODE, r"\1"),
        (_RE_MD_BOLD_ITALIC, r"\1"),
        (_RE_MD_TABLE_SEP, ""),
        (_RE_MD_TABLE_PIPE, " "),
        (_RE_MD_HEADING, ""),
        (_RE_MD_BLOCKQUOTE, ""),
        (_RE_MD_HR, ""),
        (_RE_LATEX_DISPLAY, " "),
        (_RE_LATEX_MATH, " "),
        (_RE_LATEX_ENV, " "),
        (_RE_LATEX_COMMENT, ""),
        (_RE_LATEX_CMD, " "),
        # ---- 控制与不可见字符 ----
        (_RE_CONTROL, ""),
        (_RE_ZERO_WIDTH, ""),
        (_RE_ZWSP_VARIANT, ""),
        (_RE_TAB, " "),
        # ---- 结构噪声 ----
        (_RE_PAGE_NUM, ""),
        (_RE_FOOTNOTE_MARK, ""),
        (_RE_FOOTNOTE_BRACKET, ""),
        (_RE_TOC_DOTS, ""),
        (_RE_SEPARATOR_LINE, ""),
        (_RE_BROKEN_FRAGMENT, ""),
        (_RE_EMPTY_PAREN, ""),
        # ---- 标点与空白归一 ----
        (_RE_FULLWIDTH_SPACE, " "),
        (_RE_URL, " <URL> "),
        (_RE_EMAIL, " <EMAIL> "),
        (_RE_TRAILING_SPACE, ""),
        (_RE_LEADING_SPACE_LINE, ""),
        (_RE_ISOLATED_PUNCT, ""),
        (_RE_MULTI_DOT, "..."),
        (_RE_MULTI_SPACE, " "),
        (_RE_SPACE_BEFORE_CJK_PUNCT, ""),
        (_RE_MULTI_NEWLINE, "\n\n"),
    ]

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def clean(self, text: str) -> str:
        if not self.enabled or not text:
            return (text or "").strip()
        out = text
        for pattern, repl in self._STEPS:
            out = pattern.sub(repl, out)
        # 收尾处理: 清理章节号前缀残留的多余空格, 并折叠首尾空白
        out = _RE_SECTION_NUM.sub(lambda m: m.group(0).rstrip() + " ", out)
        out = re.sub(r"\n[ \t]+\n", "\n\n", out)
        return out.strip()

    # 对外暴露规则总数, 供 README 与测试核对 "42 条" 这一数字。
    @property
    def rule_count(self) -> int:
        return len(self._STEPS) + 2  # 额外 +2: 收尾阶段的章节号替换与换行折叠
