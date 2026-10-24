# -*- coding: utf-8 -*-
"""清洗器对百分号的处理 (回归)。

背景: `_RE_LATEX_COMMENT` 最初写成 `r"%.*$"`, 本意是剔除 LaTeX 注释, 实际却
把**行内任意** `%` 到行尾的内容整段删光。业务文档中 `85%` / `20%` 这样的百分比
比比皆是, 于是"年度促销按原价的 85% 收费；如客户购买 20 个账号……"会被削成
"年度促销按原价的 85"。

其危害在于**静默**: 入库不报错, 检索全是 200, 只在模型侧表现为 chunk 里的
句子断在半截, 模型答"原文此处似乎不完整 / 被截断", 排查代价极高。
"""
from __future__ import annotations

import pytest

from app.rag.cleaner import DocumentCleaner


@pytest.fixture
def cleaner():
    return DocumentCleaner(enabled=True)


@pytest.mark.parametrize("text", [
    "年度促销活动按原价的 85% 收费。",
    "折扣为 85%，仅限年付用户。",
    "增长率 20%，环比提升 15.5%。",
    "毛利率 12.5%；净利率 8%。",
    "占比 30% 的用户贡献了 70% 的收入。",
])
def test_percent_inside_line_is_preserved(cleaner, text):
    """行内百分比之后的内容必须原样保留。"""
    assert cleaner.clean(text) == text


def test_percent_does_not_swallow_following_sentence(cleaner):
    line = "企业版年费为 1280 元。年度促销活动按原价的 85% 收费；如客户购买 20 个账号，则总价等于折后单价乘以账号数量。"
    out = cleaner.clean(line)
    assert "85%" in out
    assert "20 个账号" in out
    assert "折后单价乘以账号数量" in out


def test_latex_comment_at_line_start_is_still_removed(cleaner):
    """整行 LaTeX 注释仍须清除 —— 这本就是该规则的职责。"""
    out = cleaner.clean("正文内容\n% 这是一条 LaTeX 注释\n后续正文")
    assert "LaTeX" not in out
    assert "正文内容" in out
    assert "后续正文" in out


def test_indented_latex_comment_removed(cleaner):
    out = cleaner.clean("正文\n   % 缩进的注释\n结尾")
    assert "缩进的注释" not in out
    assert "结尾" in out


def test_escaped_percent_not_treated_as_comment(cleaner):
    out = cleaner.clean(r"折扣为 85\% 起")
    assert "85" in out


def test_idempotent(cleaner):
    text = "按原价的 85% 收费；买 20 个账号按折后价计算。\n% 注释\n正文"
    once = cleaner.clean(text)
    assert cleaner.clean(once) == once
