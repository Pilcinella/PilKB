"""测试: 查询改写 QueryRewriter (LLM + 确定性回退)."""
from __future__ import annotations

from app.rag.query_rewrite import QueryRewriter


def test_rewrite_includes_original():
    qw = QueryRewriter(enabled=True)
    out = qw.rewrite("保修期多久？", llm=None)
    assert out[0] == "保修期多久？"


def test_fallback_splits_compound_question():
    # 无 LLM 时, 复合问题按标点拆子句, 原句 + 子句分别检索.
    qw = QueryRewriter(enabled=True, max_variants=3)
    out = qw.rewrite("你们支持哪些支付方式？配送要几天？", llm=None)
    assert len(out) >= 2
    assert "你们支持哪些支付方式？配送要几天？" in out
    assert "配送要几天" in out


def test_single_clause_no_split():
    qw = QueryRewriter(enabled=True)
    out = qw.rewrite("保修多久", llm=None)
    # 单子句不拆, 只返回原句
    assert out == ["保修多久"]


def test_llm_rewrite_returns_variants():
    def fake_llm(prompt: str) -> str:
        return "保修期限是多久\n支付方式有哪些\n如何申请售后"

    qw = QueryRewriter(enabled=True, max_variants=3)
    out = qw.rewrite("你们的产品售后政策", llm=fake_llm)
    # 原句 + 3 条改写
    assert out[0] == "你们的产品售后政策"
    assert "保修期限是多久" in out
    assert "支付方式有哪些" in out
    assert "如何申请售后" in out


def test_disabled_returns_only_original():
    qw = QueryRewriter(enabled=False)
    out = qw.rewrite("任意问题？子句二？", llm=None)
    assert out == ["任意问题？子句二？"]


def test_deduplicates_variants():
    def fake_llm(prompt: str) -> str:
        return "保修期多久？\n保修期多久？"  # 重复

    qw = QueryRewriter(enabled=True)
    out = qw.rewrite("保修期多久？", llm=fake_llm)
    # 去重后不应有重复项
    assert len(out) == len(set(out))


# ---------------------------------------------------------------------------
# 查询清洗: 去掉什么 / 保留什么
#
# 判据来自 BM25 的分词器 (_SEG_RE = [一-龥]+|[a-z0-9]+): 标点一律是分隔符,
# 对 BM25 召回零影响. 所以清洗只该动"不可见字符"和"括号引号类噪声",
# 任何字母/数字/实义符号都不能碰 —— 删了就是不可逆的信息损失.
# ---------------------------------------------------------------------------

# 从网页 / PDF 复制问题时会带上的不可见字符.
_INVISIBLE = "\u200b\u200c\u200d\u2060\ufeff\u200e\u200f\u00ad"


def test_strips_invisible_characters():
    """零宽字符必须被去掉.

    回归护栏: `\\s+` 匹配不到它们 (\u200b 的 Unicode 分类是 Cf 而非 Zs),
    原实现只做空白折叠, 这些字符会原样进 embedding 和 BM25 分词.
    """
    qw = QueryRewriter(enabled=True)
    out = qw.rewrite("保修期\u200b是多久\ufeff？", llm=None)
    assert out == ["保修期是多久？"]
    assert not any(ch in out[0] for ch in _INVISIBLE)


def test_preserves_meaningful_symbols():
    """`<` `>` `《` `》` `[` `]` 是实义符号, 不能被当噪声删掉.

    对照实验 (入库侧 cleaner 的实际行为):
      "C:\\Users\\test 这个目录在哪" --cleaner--> "C: 这个目录在哪"
    LaTeX 规则把路径整个吃掉了. 查询侧不能重蹈覆辙.
    """
    qw = QueryRewriter(enabled=True)
    cases = [
        "如果 x < 5 应该怎么设置",
        "List<String> 这个类型怎么写",
        "《用户协议》第 3.2 条怎么说的",
        "文档[1]里说了什么",
        r"C:\Users\test 这个目录在哪",
        "report_final_v2.docx 在哪",
        "a_b_c 和 abc 一样吗",
    ]
    for q in cases:
        assert q in qw.rewrite(q, llm=None), f"{q!r} 被清洗破坏了"


def test_strips_brackets_and_quotes():
    """括号 / 引号类噪声仍然要去掉, 但句末标点必须保留."""
    qw = QueryRewriter(enabled=True)
    assert qw.rewrite("（必读）“安装说明”怎么用？", llm=None) == ["必读 安装说明 怎么用？"]
    # 问号是意图信号, 也是 _fallback_split 的切分点, 不能被吃掉.
    assert qw.rewrite("支持哪些支付？", llm=None) == ["支持哪些支付？"]


def test_pure_punctuation_returns_empty():
    """整个问题都是括号 / 不可见字符时, 返回 [] 而不是 [""].

    回归护栏: 返回 [""] 会让 pipeline.retrieve_and_rerank 的
    "空问题 / 纯标点" 兜底失效 ([""] 是 truthy), 空字符串被送进 embedding.
    """
    qw = QueryRewriter(enabled=True)
    assert qw.rewrite("（（））", llm=None) == []
    assert qw.rewrite("\u200b\u200b", llm=None) == []


def test_invisible_chars_do_not_break_clause_split():
    """不可见字符夹在标点之间时, 子句切分仍要正常工作."""
    qw = QueryRewriter(enabled=True, max_variants=3)
    out = qw.rewrite("支持哪些支付\u200b？配送要几天？", llm=None)
    assert "支持哪些支付？配送要几天？" in out
    assert "配送要几天" in out
