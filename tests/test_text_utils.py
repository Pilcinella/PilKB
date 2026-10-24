# -*- coding: utf-8 -*-
"""app.core.text_utils 单测: 思考块清洗 (只剥开头)。

覆盖点:
1. 开头的完整 `<think>...</think>` (含连续多个) 被整块剥离。
2. 开头未闭合 `<think>` (模型吐一半就结束) 从标签处截断到结尾。
3. 开头的自闭合 `<thinking/>` / 孤立 `</think>` 只删标签本身。
4. **正文开始后的同类标签一律保留** —— 讲解标签格式的回答不能被误删
   (旧版中段清洗会把 `<think` 之后的内容整段吞掉)。
5. 前缀撞车的 `<thinkers>` 不算思考标签。
6. 流式: 标签被切分到多个 chunk 时也不能漏 / 不能误伤。
7. 流式: 流结束仍停在思考态时, 剩余内容全部丢弃。
8. 普通文本不被误伤; 流式与一次性清洗结果一致。
"""
from __future__ import annotations

import pytest

from app.core.text_utils import ThinkBlockFilter, strip_think


# ---------- strip_think (一次性清洗) ----------

def test_strip_complete_think_block():
    raw = "<think>The user asks about X. Let me check [1].</think>\n\n答案是 42。"
    assert strip_think(raw) == "答案是 42。"


def test_strip_multiple_leading_blocks():
    raw = "<think>step 1</think><think>step 2</think>正文"
    assert strip_think(raw) == "正文"


def test_strip_multiline_think_block():
    raw = "<think>\nline1\nline2\n</think>正文在这里"
    assert strip_think(raw) == "正文在这里"


def test_strip_unclosed_leading_block():
    # 模型吐了开头但没闭合 (流式中途断 / 被截断): 从标签处到结尾全丢弃。
    raw = "<think>thinking forever without closing"
    assert strip_think(raw) == ""


def test_strip_leading_stray_close_tag():
    raw = "</think>正文"
    assert strip_think(raw) == "正文"


def test_strip_leading_self_closing_tag():
    # 自闭合只删标签本身; 旧版会被未闭合正则整段吞掉, 那是个 bug。
    raw = "<thinking/>可见内容"
    assert strip_think(raw) == "可见内容"


def test_strip_other_vendor_tags():
    raw = "<thinking>内部</thinking>可见内容"
    assert strip_think(raw) == "可见内容"


def test_strip_case_insensitive():
    raw = "<THINK>hidden</THINK>shown"
    assert strip_think(raw) == "shown"


def test_body_tags_are_user_content():
    # 正文里合法出现标签字面量 (讲解推理模型格式的回答): 原样保留。
    raw = "MiniMax-M3 会先输出 <think>推理过程</think> 再给答案"
    assert strip_think(raw) == raw


def test_body_unclosed_tag_not_truncated():
    # 旧版: 未闭合正则从 <think 吞到结尾, 后半篇答案整段消失。
    raw = "标签格式是 <think 后面跟内容"
    assert strip_think(raw) == raw


def test_prefix_collision_not_a_tag():
    assert strip_think("<thinkers>hello") == "<thinkers>hello"


def test_strip_leaves_normal_text_untouched():
    raw = "## 标题\n- 列表项 [1]\n\n普通回答。"
    assert strip_think(raw) == raw


def test_strip_handles_empty_and_non_str():
    assert strip_think("") == ""
    assert strip_think(None) is None
    assert strip_think(123) == 123


# ---------- ThinkBlockFilter (流式清洗) ----------

def _feed_all(chunks):
    f = ThinkBlockFilter()
    out = "".join(f.feed(c) for c in chunks)
    out += f.flush()
    return out


def test_stream_whole_block_in_one_chunk():
    out = _feed_all(["<think>secret</think>", "可见"])
    assert out == "可见"
    assert "secret" not in out


def test_stream_leading_open_tag_split_across_chunks():
    # 开头的 `<think>` 被切成 `<th` + `ink>`: 逐 chunk replace 会漏, 状态机不会。
    out = _feed_all(["<th", "ink>secret</think>", "后面"])
    assert out == "后面"
    assert "secret" not in out


def test_stream_close_tag_split_across_chunks():
    out = _feed_all(["<think>secret</thi", "nk>", "可见文本"])
    assert out == "可见文本"
    assert "secret" not in out


def test_stream_char_by_char():
    # 最极端: 一个字符一个 chunk。
    raw = "<think>secret</think>B"
    out = _feed_all(list(raw))
    assert out == "B"
    assert "secret" not in out


def test_stream_leading_block_then_answer():
    out = _feed_all(["<think>hmm", " more", "</think>", "答案是 42"])
    assert out == "答案是 42"


def test_stream_body_started_keeps_mid_text_think():
    # 正文已开始后出现的思考标签是用户内容: 透传, 与 strip_think 一致。
    out = _feed_all(["好的，", "<think>hmm", " more", "</think>", "答案是 42"])
    assert out == "好的，<think>hmm more</think>答案是 42"


def test_stream_unclosed_leading_think_discards_all():
    assert _feed_all(["<think>thinking forever"]) == ""


def test_stream_prefix_collision_not_swallowed():
    # `<think` 是 `<thinkers>` 的前缀: 不能因此把后面的正文全吞掉。
    assert _feed_all(list("<thinkers>hello")) == "<thinkers>hello"


def test_stream_no_think_passthrough():
    chunks = ["PilKB", " 是一个", " RAG 平台", "。"]
    assert _feed_all(chunks) == "PilKB 是一个 RAG 平台。"


@pytest.mark.parametrize("raw", [
    "<think>a</think>b",
    "<think>a</think><think>b</think>c",
    "<think/>z",
    "</think>w",
    "<think>unclosed",
    "no think here",
    "a<think>b</think>c",       # 正文里的标签: 保留
    "<thinkers>x</thinkers>y",  # 前缀撞车: 保留
])
def test_stream_matches_strip_think(raw):
    """流式过滤器的结果应与一次性清洗一致 (同一份输入)。"""
    assert _feed_all([raw]) == strip_think(raw)
