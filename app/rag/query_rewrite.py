"""查询改写 (Query Rewriting).

用户问题常常口语化、有歧义、或与文档陈述句式不一致, 直接拿去检索召回差。
改写的目标:
  - **扩展 (Expansion)**: 补上同义/上下位词, 让 BM25 也能命中.
  - **拆解 (Decomposition)**: 把复合问题拆成多个子问题, 分别检索再融合.
  - **消歧 (Disambiguation)**: 把指代/缩写展开成文档里的标准说法.

实现:
  - 有 LLM 时: 让 LLM 输出 N 个改写变体 (一行一个).
  - 无 LLM (测试/离线): 走**确定性回退** —— 按中英文标点做子句切分,
    每个子句作为一条检索 query, 与原问题一起融合, 仍然能提升召回.
"""
from __future__ import annotations

import re
from typing import Callable, List, Optional

_SPLIT_PUNCT = re.compile(r"[；;。！!？?\n]+")

# 不可见字符: 零宽空格 / 零宽连字 / 词连接符 / BOM / 左右方向符 / 软连字符.
#
# 为什么必须显式列出, 不能靠下面的 _WS: `\s` **匹配不到**它们 —— \u200b 的
# Unicode 分类是 Cf (格式字符) 而不是 Zs (空格分隔符). 用户从网页或 PDF 复制
# 问题时经常带上这些字符, 原样进 embedding 与 BM25 分词.
_ZW = re.compile("[\u200b\u200c\u200d\u2060\ufeff\u200e\u200f\u00ad]")

# 括号 / 引号类噪声 -> 空格.
#
# 刻意**不含**这三类 —— 它们在查询里可能是实义符号, 删掉不可逆:
#   `<` `>`   : 比较 / 泛型符号 ("x < 5 怎么配", "List<String> 怎么写")
#   `《` `》` : 中文书名号 ("《用户协议》第 3.2 条")
#   `[` `]`   : 引用标注 ("文档[1]里说了什么")
#
# 依据: BM25 的分词器 _SEG_RE = [一-龥]+|[a-z0-9]+ 把所有标点一律当分隔符,
# 所以标点对 BM25 召回零影响, 只对 embedding 有轻微影响 —— 删它的收益极小,
# 而误伤实义符号的代价是不可逆的. 同类教训见入库侧 cleaner: 它的 LaTeX 规则
# 会把 Windows 路径 "C:\Users\test" 整个吃掉.
# 注: 原实现写的是 `\"'\"'` —— 第二组 ASCII 引号是重复的, 而中文弯引号
# (“”‘’)一个都没有, 明显是复制时丢了字符. 这里补齐并去重.
_BRACKET = re.compile(r"""["'()（）【】“”‘’`,，、]""")

_WS = re.compile(r"\s+")


class QueryRewriter:
    """把一个问题改写成多条检索 query.

    `rewrite(query, llm=None) -> List[str]` 总是包含原始 query,
    追加 0~max_variants 条改写.
    """

    def __init__(self, enabled: bool = True, max_variants: int = 3):
        self.enabled = enabled
        self.max_variants = max_variants

    def rewrite(
        self,
        query: str,
        llm: Optional[Callable[[str], str]] = None,
        history: Optional[List[dict]] = None,
    ) -> List[str]:
        """把一个问题改写成多条检索 query.

        `history` 是**预留接口**: 元素形如 ``{"role": "user"|"assistant",
        "content": str}`` (与 history API / 前端 HistoryMessage 对齐)。**本次不
        消费它** —— 传 None 与传 ``[]`` 的行为必须完全一致 (有回归测试钉着)。

        未来的指代消解 (condensation) 会插在下面的 `cleaned` 之后、变体生成之前:
        把"它 / 那个 / 刚才说的"按历史替换成实体, 产出一个自包含的 query, 让本类
        docstring 里声明的第三条目标 (消歧) 真正落地 —— 没有历史入参时, 缩写可以
        展开, 指代无法消解。
        """
        query = (query or "").strip()
        if not query:
            return []
        if not self.enabled:
            return [query]
        # 去不可见字符 + 括号类噪声 + 折叠多余空白, 但保留句末标点
        # (问号等是 query 的一部分).
        cleaned = _WS.sub(" ", _BRACKET.sub(" ", _ZW.sub("", query))).strip()
        if not cleaned:
            # 整个问题都是括号 / 不可见字符 (例如 "（（））"), 清洗后什么都不剩.
            # 返回空列表而不是 [""] —— 后者是 truthy, 会让
            # pipeline.retrieve_and_rerank 的 "空问题 / 纯标点" 兜底失效,
            # 把一个空字符串送进 embedding 和 BM25.
            return []
        # 未来的指代消解插在这里: cleaned = self._condense(cleaned, history, llm)
        # 必须在变体生成之前 —— 变体是围绕"自包含 query"展开的, 拿带指代的原句去
        # 扩写只会把歧义放大到多条 query 上.
        if llm is not None:
            try:
                variants = self._llm_rewrite(cleaned, llm)
                if variants:
                    out = [cleaned] + variants
                    # 去重保序
                    seen = set()
                    uniq = []
                    for v in out:
                        if v and v not in seen:
                            seen.add(v)
                            uniq.append(v)
                    return uniq[: 1 + self.max_variants]
            except Exception:
                pass
        # 确定性回退: 子句切分 (复合问题拆解)
        return self._fallback_split(cleaned)

    @staticmethod
    def _llm_rewrite(query: str, llm: Callable[[str], str]) -> List[str]:
        prompt = (
            "你是检索系统的 Query Rewriter. 给定用户问题, 生成最多 3 条"
            "检索友好的改写 (同义扩展 / 子问题拆解 / 标准说法), 每行一条, "
            "不要编号, 不要解释:\n\n" + query
        )
        resp = (llm(prompt) or "").strip()
        out: List[str] = []
        for line in resp.splitlines():
            line = line.strip().lstrip("0123456789.-、) ").strip()
            if line:
                out.append(line)
        return out

    def _fallback_split(self, query: str) -> List[str]:
        """无 LLM 回退: 按标点拆子句, 原句 + 各子句分别检索后融合."""
        parts = [p.strip() for p in _SPLIT_PUNCT.split(query) if p.strip()]
        # 单个子句就没必要拆了
        if len(parts) <= 1:
            return [query]
        out = [query]
        for p in parts:
            if len(p) >= 4 and p not in out:  # 太短的子句意义不大
                out.append(p)
        return out[: 1 + self.max_variants]
