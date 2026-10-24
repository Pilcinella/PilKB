"use client";
import { Suspense, useEffect, useRef, useState } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeHighlight from "rehype-highlight";
import {
  Source,
  appendMessage,
  createSession,
  ensureDevToken,
  getSession,
  getStoredKbId,
  setStoredKbId,
  streamChat,
  DEMO_KB,
} from "@/lib/api";
import { useShell } from "@/components/AppShell";
import KbPill from "@/components/KbPill";
import PillSelect from "@/components/PillSelect";
import { IconCpu, IconGear, IconSend } from "@/components/icons";

type Msg = {
  role: "user" | "assistant";
  content: string;
  sources: Source[];
  /** 独立于 content 的错误信息; 非空时渲染成错误卡片而不是糊在正文里 */
  error?: string;
  /**
   * 服务端下发的提示 (notice 帧), 如"本次未使用知识库"。
   * 必须与 content 分开: 正文会被送进 RAGAS 打分和会话历史, 提示混进去会污染两者。
   */
  notice?: string;
  /** 正在流式输出(用于显示光标) */
  streaming?: boolean;
  /**
   * 思考期指示: 收到 source/notice 且尚未产出首字时置 true, 首个 token 到达即清除。
   * 这段空档里推理模型在后台思考(首 token 可能耗时数十秒), 不给状态会像"卡住"。
   */
  thinking?: boolean;
};

// 快捷问题: 对齐已上传的键盘文档 (`data/uploads/doc_2c637cbe29e0.md`,
// A87 Pro 产品手册) 的内容, 保证开箱即问就能拿到有引用的真实回答,
// 而不是"未找到相关信息"。
const SUGGESTIONS = [
  "A87 Pro 支持哪几种连接方式？怎么切换？",
  "如何用组合键调节 RGB 灯光亮度和切换灯效？",
  "热插拔轴体怎么更换？兼容哪些轴？",
  "键盘连接不上或按键失灵，该怎么排查？",
];

/** 只更新最后一条消息(流式增量写入的目标) */
function patchLast(msgs: Msg[], fn: (m: Msg) => Msg): Msg[] {
  if (msgs.length === 0) return msgs;
  const i = msgs.length - 1;
  return msgs.map((m, idx) => (idx === i ? fn(m) : m));
}

/** 输入框随内容增高 (DeepSeek 式): 上限由 CSS max-height 兜底 */
function autoGrow(el: HTMLTextAreaElement | null) {
  if (!el) return;
  el.style.height = "auto";
  el.style.height = `${Math.min(el.scrollHeight, 180)}px`;
}

function ChatView() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const shell = useShell();
  // 侧栏恢复历史会话以 URL ?s=<session_id> 驱动: 刷新 / 返回都能回到同一会话
  const sidParam = searchParams.get("s");

  const [token, setToken] = useState("");
  // 选中的知识库: localStorage 里的上次选择, 兜底 DEMO_KB. 只能在 effect 里
  // 初始化 (localStorage 是浏览器 API, SSR 阶段不可用).
  const [kbId, setKbId] = useState(DEMO_KB);
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<"rag" | "agent">("rag");
  const [topK, setTopK] = useState(5);
  const [topKOpen, setTopKOpen] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [loading, setLoading] = useState(false);
  const [ready, setReady] = useState(false);
  const [copiedIdx, setCopiedIdx] = useState<number | null>(null);

  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const topKRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setKbId(getStoredKbId() || DEMO_KB);
    ensureDevToken().then((t) => {
      setToken(t);
      setReady(true);
    });
  }, []);

  // 恢复历史会话: 侧栏点选 → /chat?s=<id> → 拉详情回填消息流与 kb/mode.
  // token 异步就绪, 依赖里带上它让本 effect 重跑一次.
  useEffect(() => {
    if (!sidParam || !token) return;
    let cancelled = false;
    (async () => {
      try {
        const detail = await getSession(token, sidParam);
        if (cancelled) return;
        setSessionId(detail.session_id);
        setMessages(
          detail.messages.map((m) => ({
            role: m.role,
            content: m.content,
            sources: m.sources ?? [],
          })),
        );
        if (detail.kb_id) {
          setKbId(detail.kb_id);
          setStoredKbId(detail.kb_id);
        }
        setMode(detail.mode === "agent" ? "agent" : "rag");
        shell.setActiveSessionId(detail.session_id);
      } catch (e: any) {
        if (!cancelled)
          setMessages((m) => [
            ...m,
            {
              role: "assistant",
              content: "",
              sources: [],
              error: `恢复会话失败: ${e?.message ?? e}`,
            },
          ]);
      }
    })();
    return () => {
      cancelled = true;
    };
    // shell 只用其中稳定的 setter (setActiveSessionId), 不把 api 对象纳入依赖
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sidParam, token]);

  // "开启新对话" (顶栏/侧栏按钮): nonce 自增 → 清空当前消息与会话
  const firstNonce = useRef(true);
  useEffect(() => {
    if (firstNonce.current) {
      firstNonce.current = false;
      return;
    }
    setSessionId(null);
    setMessages([]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [shell.resetNonce]);

  // 新消息到达时滚到底部
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);

  // 输入框高度跟随内容
  useEffect(() => {
    autoGrow(inputRef.current);
  }, [q]);

  // top_k 弹层: 点击外部关闭
  useEffect(() => {
    if (!topKOpen) return;
    function onDoc(e: MouseEvent) {
      if (topKRef.current && !topKRef.current.contains(e.target as Node)) setTopKOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    return () => document.removeEventListener("mousedown", onDoc);
  }, [topKOpen]);

  async function ensureSession(firstQuestion: string) {
    if (sessionId) return sessionId;
    if (!token) return null;
    // 标题取首个问题前 24 字 (后端不会自动改标题), 侧栏列表里能认出这条会话。
    // 显式带 kb_id: 否则后端会落成 "default", 与实际检索用的知识库不一致。
    const r = await createSession(token, {
      title: firstQuestion.trim().slice(0, 24) || "新会话",
      mode,
      kb_id: kbId,
    });
    setSessionId(r.session_id);
    // 上报给外壳: 侧栏高亮当前会话并刷新列表
    shell.setActiveSessionId(r.session_id);
    shell.refreshSessions();
    return r.session_id;
  }

  /**
   * 发一轮问答。
   * @param addUser 重新生成时为 false —— 用户消息已在历史里, 只需重跑助手那一轮
   */
  async function runAsk(text: string, addUser: boolean) {
    const trimmed = text.trim();
    if (!trimmed || !token) return;

    // createSession 401/网络失败以前会直接抛出且无人接住 —— 用户点发送后
    // 界面毫无反应, 错误只躺在控制台。落到消息流里给个可见的失败卡片。
    let sid: string | null = null;
    try {
      sid = await ensureSession(trimmed);
    } catch (e: any) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "", sources: [], error: e?.message ?? String(e) },
      ]);
      return;
    }
    if (!sid) {
      // token 为空: 生产环境 (dev-token 403) 未登录, 或获取 token 失败.
      // 之前这里静默 return, 用户点发送毫无反应。落到消息流给可见提示。
      setMessages((m) => [
        ...m,
        {
          role: "assistant",
          content: "",
          sources: [],
          error: "未登录或登录已过期：请点击侧栏底部的「登录 / 注册」后重试。",
        },
      ]);
      return;
    }

    if (addUser) {
      setMessages((m) => [...m, { role: "user", content: trimmed, sources: [] }]);
    }
    setMessages((m) => [
      ...m,
      { role: "assistant", content: "", sources: [], streaming: true },
    ]);
    setLoading(true);

    if (addUser) {
      try {
        await appendMessage(token, sid, {
          role: "user",
          content: trimmed,
          ts: Date.now() / 1000,
        });
      } catch {
        /* 落库失败不影响对话本身 */
      }
    }

    let collected = "";
    // 用局部变量收集引用: 原实现读 asstMsg.sources, 而那是个初始空数组常量,
    // 流式写回的是 state 里的对象, 结果入库的 sources 永远为空。
    const collectedSources: Source[] = [];
    let errMsg = "";

    try {
      for await (const evt of streamChat({
        token,
        question: trimmed,
        session_id: sid,
        kb_id: kbId,
        mode,
        top_k: topK,
      })) {
        if (evt.type === "token") {
          const t = (evt.data as any)?.text ?? "";
          collected += t;
          setMessages((m) =>
            patchLast(m, (it) => ({ ...it, content: it.content + t, thinking: false })),
          );
        } else if (evt.type === "source") {
          const src = evt.data as Source;
          collectedSources.push(src);
          setMessages((m) =>
            patchLast(m, (it) => ({
              ...it,
              sources: [...it.sources, src],
              // 首字未出时收到 source: 检索已完成、模型在生成, 切到"正在思考"
              thinking: it.content === "",
            })),
          );
        } else if (evt.type === "notice") {
          const n = (evt.data as any)?.text ?? "";
          if (n)
            setMessages((m) =>
              patchLast(m, (it) => ({ ...it, notice: n, thinking: it.content === "" })),
            );
        } else if (evt.type === "error") {
          errMsg = (evt.data as any)?.msg ?? "服务端返回错误";
        }
      }
    } catch (e: any) {
      errMsg = e?.message ?? String(e);
    } finally {
      setLoading(false);
      setMessages((m) =>
        patchLast(m, (it) => ({
          ...it,
          streaming: false,
          // 兜底清除: 错误 / 非流式模式(无 TokenEvent) / 空回答都走到这里
          thinking: false,
          error: errMsg || undefined,
        })),
      );
      // 只有成功产出内容才落库, 避免把错误文本写进历史
      if (!errMsg && collected) {
        try {
          await appendMessage(token, sid, {
            role: "assistant",
            content: collected,
            sources: collectedSources,
            ts: Date.now() / 1000,
          });
        } catch {
          /* ignore */
        }
      }
      // 侧栏列表的消息数/更新时间以服务端为准, 顺手刷新
      shell.refreshSessions();
    }
  }

  function ask() {
    const text = q;
    if (!text.trim()) return;
    setQ("");
    runAsk(text, true);
  }

  function onKeydown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    // 仿 DeepSeek: Enter 发送, Shift+Enter 换行 (兼容 ⌘/Ctrl+Enter)。
    // 输入法组词中的 Enter (isComposing) 不发送, 否则中文没法断词选字。
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      ask();
    }
  }

  /** 重新生成: 回溯到该助手消息之前最近的用户提问, 截断后重跑 */
  function regenerate(idx: number) {
    let question = "";
    for (let i = idx - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        question = messages[i].content;
        break;
      }
    }
    if (!question) return;
    setMessages((m) => m.slice(0, idx));
    runAsk(question, false);
  }

  async function copyText(text: string, idx: number) {
    try {
      await navigator.clipboard.writeText(text);
      setCopiedIdx(idx);
      setTimeout(() => setCopiedIdx(null), 1500);
    } catch {
      /* 剪贴板不可用(如非安全上下文)时静默失败 */
    }
  }

  /**
   * 切换知识库. kb 绑定在会话元数据上 (建会话时写入), 中途换库继续沿用旧会话
   * 会让侧栏显示错误的库名, 也会把两个库的回答混进同一串对话 —— 换库即开
   * 新会话, 行为与"开启新对话"按钮一致.
   */
  function switchKb(next: string) {
    if (!next || next === kbId) return;
    setKbId(next);
    setStoredKbId(next);
    setSessionId(null);
    setMessages([]);
    shell.setActiveSessionId(null);
    // 若正处在恢复的历史会话上 (?s=...), 摘掉参数避免刷新后回连旧会话
    router.replace("/chat");
  }

  // 初始化阶段用骨架屏, 比一行"正在准备"更接近成品观感
  if (!ready) {
    return (
      <div className="ds-chat">
        <div className="ds-msgs">
          <div className="ds-msgs-inner">
            <div className="card">
              <div className="skeleton skeleton-line w-45" />
              <div className="skeleton skeleton-line w-90" />
              <div className="skeleton skeleton-line w-70" />
            </div>
          </div>
        </div>
      </div>
    );
  }

  const canSend = !!q.trim() && !loading;

  return (
    <div className={"ds-chat" + (messages.length === 0 ? " empty" : "")}>
      <div className="ds-msgs">
        <div className="ds-msgs-inner">
          {messages.length === 0 && (
            <div className="ds-hero">
              <div className="ds-hero-title">
                你好，我是 <span className="grad">PilKB</span> 助手
              </div>
              <div className="ds-hero-sub">
                基于你的知识库完成检索、重排与生成，引用可溯源。免登录，直接开始提问。
              </div>
              <div className="suggs">
                {SUGGESTIONS.map((s) => (
                  <button
                    key={s}
                    className="sugg"
                    onClick={() => {
                      setQ(s);
                      inputRef.current?.focus();
                    }}
                  >
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="ds-msg-user-row">
                <div className="ds-bubble-user">{m.content}</div>
              </div>
            ) : (
              <div key={i} className="ds-msg-assistant">
                <div className="ds-ava">P</div>
                <div className="ds-msg-body">
                  <div className="md">
                    <ReactMarkdown
                      remarkPlugins={[remarkGfm]}
                      rehypePlugins={[[rehypeHighlight, { ignoreMissing: true }]]}
                      components={{
                        a: ({ ...props }) => (
                          <a {...props} target="_blank" rel="noopener noreferrer" />
                        ),
                      }}
                    >
                      {m.content}
                    </ReactMarkdown>
                    {m.streaming && <span className="stream-cursor" />}
                  </div>
                  {/* source 已到、首字未出: 推理模型正在思考, 状态行比裸三点更像"在干活" */}
                  {m.thinking && !m.error && (
                    <div className="thinking-line">
                      <span className="typing">
                        <i />
                        <i />
                        <i />
                      </span>
                      正在思考…
                    </div>
                  )}
                  {/* 流式尚未产出首字时显示三点, 避免看起来"卡住了" */}
                  {!m.thinking && m.content === "" && !m.error && (
                    <span className="typing">
                      <i />
                      <i />
                      <i />
                    </span>
                  )}

                  {m.notice && (
                    <div className="notice-box">
                      <span className="n-icon">ⓘ</span>
                      <span>{m.notice}</span>
                    </div>
                  )}

                  {m.error && (
                    <div className="error-box">
                      <span className="e-icon">⚠</span>
                      <div className="e-body">
                        <div className="e-msg">{m.error}</div>
                        <div className="e-sub">
                          可能是模型服务未配置或网络异常，可重试一次。
                        </div>
                      </div>
                      <button
                        className="ghost"
                        style={{ padding: "5px 12px", fontSize: 12 }}
                        onClick={() => regenerate(i)}
                      >
                        重试
                      </button>
                    </div>
                  )}

                  {m.sources.length > 0 && (
                    <details className="sources">
                      <summary>引用 {m.sources.length} 条</summary>
                      {m.sources.map((s, j) => (
                        <div key={j} className="source">
                          <span className="score">{Number(s.score).toFixed(3)}</span>{" "}
                          <span className="text">{s.content}</span>
                        </div>
                      ))}
                    </details>
                  )}

                  {/* 消息操作: 悬停显形 */}
                  {!m.streaming && m.content && (
                    <div className="msg-actions">
                      <button
                        className={"icon-btn" + (copiedIdx === i ? " copied" : "")}
                        title="复制"
                        onClick={() => copyText(m.content, i)}
                      >
                        {copiedIdx === i ? "✓" : "⧉"}
                      </button>
                      <button
                        className="icon-btn"
                        title="重新生成"
                        disabled={loading}
                        onClick={() => regenerate(i)}
                      >
                        ⟳
                      </button>
                    </div>
                  )}
                </div>
              </div>
            ),
          )}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="ds-composer-zone">
        <div className="ds-composer">
          <textarea
            ref={inputRef}
            rows={1}
            placeholder="给 PilKB 发送消息"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={onKeydown}
          />
          <div className="ds-composer-bar">
            {/* 左: 知识库 / 模式 (对应 DeepSeek 的"深度思考/联网搜索"胶囊位) */}
            <KbPill token={token} kbId={kbId} onChange={switchKb} />
            <PillSelect
              value={mode}
              onChange={(v) => setMode(v as "rag" | "agent")}
              icon={<IconCpu size={15} />}
              items={[
                { value: "rag", label: "RAG 模式" },
                { value: "agent", label: "Agent 自动路由" },
              ]}
            />
            <div className="ds-composer-gap" />
            {/* 右: top_k 齿轮 + 发送 */}
            <div className="topk-pop-wrap" ref={topKRef}>
              <button
                type="button"
                className={"ds-icobtn" + (topKOpen ? " on" : "")}
                title="检索设置 (top_k)"
                onClick={() => setTopKOpen((o) => !o)}
              >
                <IconGear size={17} />
              </button>
              {topKOpen && (
                <div className="topk-pop">
                  <div className="topk-head">检索设置</div>
                  <div className="topk-row">
                    <span>top_k（召回片段数）</span>
                    <span className="topk-val mono">{topK}</span>
                  </div>
                  <input
                    type="range"
                    min={1}
                    max={20}
                    step={1}
                    value={topK}
                    onChange={(e) => setTopK(Number(e.target.value))}
                  />
                  <div className="topk-presets">
                    {[3, 5, 8, 10].map((n) => (
                      <button
                        key={n}
                        type="button"
                        className={"chip-btn" + (n === topK ? " on" : "")}
                        onClick={() => setTopK(n)}
                      >
                        {n}
                      </button>
                    ))}
                  </div>
                  <div className="muted">值越大召回越全，但生成越慢</div>
                </div>
              )}
            </div>
            <button
              type="button"
              className="ds-send"
              onClick={ask}
              disabled={!canSend}
              title="发送 (Enter)"
            >
              {loading ? <span className="ds-send-spinner" /> : <IconSend size={18} />}
            </button>
          </div>
        </div>
        <div className="ds-hint">内容由 AI 生成，仅供参考 · Enter 发送，Shift + Enter 换行</div>
      </div>
    </div>
  );
}

export default function ChatPage() {
  // useSearchParams 需要上层 Suspense 兜底, 否则 next build 静态预渲染会报错
  return (
    <Suspense fallback={null}>
      <ChatView />
    </Suspense>
  );
}
