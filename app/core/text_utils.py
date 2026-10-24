# -*- coding: utf-8 -*-
"""模型输出清洗工具。

背景: 部分推理模型 (如 MiniMax-M3 / DeepSeek-R1 系列) 会把内部思考过程以
`<think>...</think>` 块的形式吐在 `content` 里, 而不是放进独立的
`reasoning_content` 字段。如果原样透传, 用户会在回答正文开头看到一大段英文
内心独白, 录屏演示里尤其明显。

`OPENAI_EXTRA_BODY` 可以在部分平台上关掉思考, 但:
1. 不是所有 provider / 网关都支持该字段;
2. 字段拼错或被 shell 吃掉引号时静默失效;
3. 换成别的 provider 又要重配一遍。

所以这里做**兜底清洗**: 不管上游有没有关掉, 出网前统一剥掉思考块。

支持的标签 (各厂商叫法不一, 统一按"块"处理, 连内容一起删):
`<think>` / `<thinking>` / `<reasoning>` / `<reflection>`

只剥**开头**的思考块: 思考块几乎总在输出最前面, 而正文里合法出现这些标签的
字面量 (如"推理模型会输出 <think>...</think>") 是用户内容 —— 中段清洗会误删,
未闭合的 `<think` 甚至会把后半篇答案整段吞掉。

两个入口:
- `strip_think(text)`: 一次性清洗 (非流式 / 落库前)。
- `ThinkBlockFilter`: 流式逐 token 清洗, 能处理标签被切分到多个 chunk
  的情况 (例如先收到 `<th`, 再收到 `ink>`)。
"""
from __future__ import annotations

import re

# 各厂商的思考标签名 (不带尖括号)。
THINK_TAGS = ("think", "thinking", "reasoning", "reflection")

# 开头的完整块 `<tag ...>...</tag>`: 连内容一起删除。可能有连续多个 (分步推理),
# strip_think 里循环剥; `^\s*` 锚定只匹配输出开头。
_LEAD_BLOCK_RES = tuple(
    re.compile(rf"^\s*<{t}\b[^>]*>.*?</{t}\s*>", re.IGNORECASE | re.DOTALL) for t in THINK_TAGS
)
# 开头的自闭合 `<think/>` / 孤立闭标签 `</think>`: 只删标签本身。注意不能匹配
# 裸开标签 `<think>` —— 那是未闭合思考块的开头, 要交给 _LEAD_OPEN_RES 截断。
_LEAD_STRAY_RES = tuple(
    re.compile(rf"^\s*(?:</{t}\b[^>]*>|<{t}\b[^>]*/\s*>)", re.IGNORECASE) for t in THINK_TAGS
)
# 开头的未闭合块 `<tag ...>...`: 模型吐一半就结束 / 被截断, 从标签处到结尾全删。
# 必须放在完整块 / 自闭合之后判断 —— 它会吞掉剩余全部内容。
_LEAD_OPEN_RES = tuple(
    re.compile(rf"^\s*<{t}\b[^>]*>.*", re.IGNORECASE | re.DOTALL) for t in THINK_TAGS
)


def strip_think(text: str) -> str:
    """剥离模型输出**开头**的思考块, 返回清洗后的正文。

    - 开头的完整 `<think>...</think>` (可连续多个): 标签连同内容整块移除。
    - 开头的自闭合 `<think/>` / 孤立闭标签 `</think>`: 只移除标签本身。
    - 开头的未闭合 `<think ...>`: 从该标签到结尾全部移除。
    - 正文开始后出现的同类标签一律视为用户内容, 原样保留 —— 否则讲解这些
      标签格式的回答会被误删, 甚至被未闭合块截断。
    - 非字符串输入: 原样返回, 不抛异常 (调用点大多是宽松透传)。
    """
    if not isinstance(text, str) or not text:
        return text
    out = text
    # 完整块可能连续多个, 循环剥到剥不动为止。
    while True:
        for regex in _LEAD_BLOCK_RES:
            new = regex.sub("", out, count=1)
            if new != out:
                out = new
                break
        else:
            break
    while True:
        for regex in _LEAD_STRAY_RES:
            new = regex.sub("", out, count=1)
            if new != out:
                out = new
                break
        else:
            break
    for regex in _LEAD_OPEN_RES:
        out = regex.sub("", out, count=1)
    return out.strip()


class ThinkBlockFilter:
    """流式思考块过滤器。

    用法::

        f = ThinkBlockFilter()
        for chunk in stream:
            visible = f.feed(getattr(chunk, "content", "") or "")
            if visible:
                yield visible
        tail = f.flush()
        if tail:
            yield tail

    状态机 (与 strip_think 的"只剥开头"语义一致):
    - 开头态: 还没放行过非空白正文。开头的思考标签按思考块处理; 只放行过
      空白不算正文 —— 其后仍可能是思考块。
    - 思考态: 丢弃所有文本, 直到真正的闭标签才切回开头态。
    - 透传态: 已放行过非空白正文, 之后的一切 (含同类标签) 都是用户内容,
      原样放行不再扫描。
    - 因为开/闭标签可能被切到两个 chunk 之间, 每次只消费"确定安全"的部分,
      把可能为标签前缀的尾部留在 buffer 里等下一个 chunk。
    """

    # 最长的闭标签是 `</reflection>` (13 字符); 多留一点余量。
    _RESERVE = 16

    def __init__(self, tags=THINK_TAGS) -> None:
        self._tags = frozenset(t.lower() for t in tags)
        self._open_tokens = tuple("<" + t for t in tags)
        self._close_tokens = tuple("</" + t for t in tags)
        self._buf = ""
        self._in_think = False
        # 是否已放行过非空白正文 —— 一旦 True, 后续输入全部透传。
        self._body_started = False

    def _safe_len(self, text: str) -> int:
        """返回 buffer 中可以安全消费的长度 (保留疑似标签前缀的尾巴)。"""
        if len(text) <= self._RESERVE:
            return 0
        return len(text) - self._RESERVE

    @staticmethod
    def _find_earliest(text: str, tokens):
        """找出 tokens 中最早出现的位置, 返回 (index, matched_token); 没有则 (-1, None)。"""
        best_idx, best_tok = -1, None
        lowered = text.lower()
        for tok in tokens:
            idx = lowered.find(tok)
            if idx != -1 and (best_idx == -1 or idx < best_idx):
                best_idx, best_tok = idx, tok
        return best_idx, best_tok

    @staticmethod
    def _tag_name(raw: str) -> str:
        """从标签体 (`think` / `thinking x=1` / `thinkers`) 提取标签名。"""
        return raw.split(None, 1)[0].rstrip("/")

    def feed(self, chunk: str) -> str:
        """喂入一个片段, 返回应当展示给用户的文本。"""
        if not chunk:
            return ""
        self._buf += chunk
        out: list[str] = []

        while self._buf:
            if self._in_think:
                # 思考态: 找真闭标签。
                idx, tok = self._find_earliest(self._buf, self._close_tokens)
                if idx == -1:
                    # 还没收到闭标签: 丢弃除尾部保留区以外的全部内容。
                    n = self._safe_len(self._buf)
                    if n == 0:
                        break
                    self._buf = self._buf[n:]
                    continue
                gt = self._buf.find(">", idx)
                if gt == -1:
                    # 闭标签还没收全 (`</think` 但 `>` 在下一个 chunk): 丢掉它之前的
                    # 思考内容, 把疑似闭标签的尾巴留在 buffer 里, 等收全再切状态.
                    # 直接切回正文态会把后面那个 `>` 当成正文漏给用户。
                    self._buf = self._buf[idx:]
                    break
                if self._tag_name(self._buf[idx + 2 : gt]) not in self._tags:
                    # `</thinkers>` 之类前缀撞车的字符串是思考内容: 丢掉继续等真闭标签。
                    self._buf = self._buf[gt + 1 :]
                    continue
                # 收到真闭标签: 丢弃思考内容, 切回开头态。
                self._buf = self._buf[gt + 1 :]
                self._in_think = False
                continue

            if self._body_started:
                # 正文已开始: 同类标签都是用户内容 (如讲解标签格式的回答), 全部透传。
                out.append(self._buf)
                self._buf = ""
                break

            # 开头态: 找最早出现的开/闭标签。
            idx, tok = self._find_earliest(self._buf, self._open_tokens + self._close_tokens)
            if idx == -1:
                # 没有标签: 消费掉除尾部保留区以外的全部内容。
                n = self._safe_len(self._buf)
                if n == 0:
                    break
                head = self._buf[:n]
                self._buf = self._buf[n:]
                out.append(head)
                # 放行过非空白正文才算"正文开始"。
                if head.strip():
                    self._body_started = True
                continue
            if self._buf[:idx].strip():
                # 首个标签之前已有正文: 不是开头的思考块, 标签按用户内容透传。
                self._body_started = True
                continue
            gt = self._buf.find(">", idx)
            if gt == -1:
                # 标签没收全: 留在 buffer 等收全再判断 (可能是 <thinkers> 这类正文)。
                self._buf = self._buf[idx:]
                break
            if tok in self._close_tokens:
                # 开头的孤立闭标签 (流被截断后只补了 `</think>`): 只丢标签本身。
                if self._tag_name(self._buf[idx + 2 : gt]) not in self._tags:
                    self._body_started = True
                    continue
                self._buf = self._buf[gt + 1 :]
                continue
            raw = self._buf[idx + 1 : gt]
            if raw.rstrip().endswith("/"):
                # 自闭合 `<think/>`: 只丢标签本身。
                self._buf = self._buf[gt + 1 :]
                continue
            if self._tag_name(raw) not in self._tags:
                # `<thinkers>` 之类前缀撞车: 不是思考标签, 按正文透传。
                self._body_started = True
                continue
            # 真开标签: 跳过标签本身, 进入思考态。
            self._buf = self._buf[gt + 1 :]
            self._in_think = True

        return "".join(out)

    def flush(self) -> str:
        """流结束时调用: 吐出 buffer 里剩余的非思考文本。

        若流结束时仍停在思考态, 说明思考块没闭合, 剩余内容全部丢弃 —— 这与
        `strip_think` 对"未闭合块"的处理保持一致。
        """
        rest = "" if self._in_think else self._buf
        self._buf = ""
        self._in_think = False
        return rest
