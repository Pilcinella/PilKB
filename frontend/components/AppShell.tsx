"use client";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import ThemeToggle from "@/components/ThemeToggle";
import {
  SessionMeta,
  clearStoredAuth,
  deleteSession,
  ensureDevToken,
  getStoredUser,
  listSessions,
} from "@/lib/api";
import {
  IconChevronsLeft,
  IconLogout,
  IconNewChat,
  IconPanelLeft,
  IconPlus,
  IconSearch,
  IconTrash,
  IconUpload,
  IconUser,
  IconX,
} from "@/components/icons";
import LoginModal from "@/components/LoginModal";

/**
 * DeepSeek 式应用外壳: 可收起侧栏 (新对话 / 搜索 / 历史会话列表) + 顶栏
 * (3 个圆形图标按钮 + 对话/检索滑块)。放在根 layout 里包裹所有路由。
 *
 * shell 与 chat 页通过 context 共享会话状态:
 * - openSession: 侧栏点选历史会话 → 跳 /chat?s=<id>, chat 页拉详情回填
 * - setActiveSessionId: chat 页自建会话后上报, 侧栏据此高亮
 * - resetNonce: "开启新对话"自增, chat 页据此清空消息
 */
export type ShellApi = {
  activeSessionId: string | null;
  setActiveSessionId: (id: string | null) => void;
  resetNonce: number;
  newChat: () => void;
  openSession: (id: string) => void;
  refreshSessions: () => void;
};

const ShellContext = createContext<ShellApi | null>(null);

export function useShell(): ShellApi {
  const ctx = useContext(ShellContext);
  if (!ctx) throw new Error("useShell 必须在 AppShell 内使用");
  return ctx;
}

/** 会话按更新时间分组 (DeepSeek: 今天 / 昨天 / 7 天内 …) */
function fmtGroup(ts: number): string {
  if (!ts) return "更早";
  const day = (x: Date) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = (day(new Date()) - day(new Date(ts * 1000))) / 86400000;
  if (diff <= 0) return "今天";
  if (diff === 1) return "昨天";
  if (diff <= 7) return "7 天内";
  if (diff <= 30) return "30 天内";
  return "更早";
}

export default function AppShell({ children }: { children: ReactNode }) {
  const router = useRouter();
  const pathname = usePathname();
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [token, setToken] = useState("");
  const [user, setUser] = useState("");
  const [sessions, setSessions] = useState<SessionMeta[] | null>(null);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [resetNonce, setResetNonce] = useState(0);
  const [search, setSearch] = useState("");
  const [loginOpen, setLoginOpen] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  // 视图由路由决定: 滑块开关就是 chat/sources 两个路由之间的导航,
  // 刷新 / 分享链接时状态不丢。
  const view = pathname?.startsWith("/sources") ? "sources" : "chat";

  const refreshSessions = useCallback(() => {
    if (!token) return;
    listSessions(token)
      .then((list) =>
        setSessions([...list].sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0))),
      )
      .catch(() => {
        /* 列表失败不阻塞主界面, 下次展开侧栏会重试 */
      });
  }, [token]);

  useEffect(() => {
    // dev 环境: dev-token 成功后 localStorage 里已写入 "dev-user";
    // 生产环境: dev-token 403, getStoredUser() 为空 = 未登录, 侧栏显示登录按钮.
    // 不能在这兜底 "dev-user" —— 那会把生产环境的未登录态伪装成已登录.
    ensureDevToken().then((t) => {
      setToken(t);
      setUser(getStoredUser());
    });
  }, []);

  useEffect(() => {
    refreshSessions();
  }, [refreshSessions]);

  // 每次展开侧栏刷新一次, 捕捉其它入口产生的变化
  useEffect(() => {
    if (sidebarOpen) refreshSessions();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sidebarOpen]);

  // 窄屏默认收起 (移动端侧栏是遮罩式, 默认展开会盖住内容)
  useEffect(() => {
    if (window.innerWidth < 900) setSidebarOpen(false);
  }, []);

  const newChat = useCallback(() => {
    setActiveSessionId(null);
    setResetNonce((n) => n + 1);
    router.push("/chat");
  }, [router]);

  const openSession = useCallback(
    (id: string) => {
      setActiveSessionId(id);
      router.push(`/chat?s=${encodeURIComponent(id)}`);
    },
    [router],
  );

  const api = useMemo<ShellApi>(
    () => ({ activeSessionId, setActiveSessionId, resetNonce, newChat, openSession, refreshSessions }),
    [activeSessionId, resetNonce, newChat, openSession, refreshSessions],
  );

  async function onDelete(id: string) {
    if (!token) return;
    if (!confirm("确定要删除这个会话吗？")) return;
    try {
      await deleteSession(token, id);
      // 删的是正在看的会话: 顺带开一个新对话, 免得界面停在已不存在的会话上
      if (activeSessionId === id) {
        setActiveSessionId(null);
        setResetNonce((n) => n + 1);
        router.push("/chat");
      }
      refreshSessions();
    } catch {
      /* 删除失败: 列表保留原样 */
    }
  }

  function focusSearch() {
    setSidebarOpen(true);
    // 展开动画进行中 input 可能尚未可见, 隔两帧再聚焦
    requestAnimationFrame(() => requestAnimationFrame(() => searchRef.current?.focus()));
  }

  function logout() {
    clearStoredAuth();
    // 页面组件 state 里还留着旧 token, 整页重载最干净 (cookie 由后端清)
    window.location.reload();
  }

  // 会话按搜索词过滤 + 按时间顺序分组 (列表已按 updated_at 降序, 相邻聚合即可)
  const kw = search.trim().toLowerCase();
  const filtered = (sessions ?? []).filter(
    (s) => !kw || (s.title || "").toLowerCase().includes(kw),
  );
  const groups: { label: string; items: SessionMeta[] }[] = [];
  for (const s of filtered) {
    const label = fmtGroup(s.updated_at);
    const last = groups[groups.length - 1];
    if (last && last.label === label) last.items.push(s);
    else groups.push({ label, items: [s] });
  }

  return (
    <ShellContext.Provider value={api}>
      <div className={"ds-shell" + (sidebarOpen ? "" : " side-closed")}>
        <aside className="ds-sidebar">
          <div className="ds-sidebar-inner">
            <div className="ds-side-head">
              <Link href="/chat" className="brand ds-brand">
                <span className="logo-mark">P</span>
                <span className="brand-name">
                  Pil<span className="grad">KB</span>
                </span>
              </Link>
              <button
                type="button"
                className="ds-icobtn"
                onClick={() => setSidebarOpen(false)}
                title="收起侧栏"
              >
                <IconChevronsLeft />
              </button>
            </div>

            <div className="ds-side-new">
              <button type="button" className="ds-newchat" onClick={newChat}>
                <IconPlus size={16} />
                开启新对话
              </button>
              <Link href="/upload" className="ds-side-upload" title="上传文档到知识库">
                <IconUpload size={15} />
                上传文档
              </Link>
            </div>

            <div className="ds-side-search">
              <IconSearch size={15} />
              <input
                ref={searchRef}
                placeholder="搜索对话"
                value={search}
                onChange={(e) => setSearch(e.target.value)}
              />
              {search && (
                <button
                  type="button"
                  className="ds-icobtn mini"
                  title="清空"
                  onClick={() => {
                    setSearch("");
                    searchRef.current?.focus();
                  }}
                >
                  <IconX size={13} />
                </button>
              )}
            </div>

            <nav className="ds-side-list">
              {sessions === null && (
                <div className="ds-side-empty muted">正在加载会话…</div>
              )}
              {sessions !== null && filtered.length === 0 && (
                <div className="ds-side-empty muted">
                  {kw ? "没有匹配的对话" : "暂无历史对话，发起第一次提问吧"}
                </div>
              )}
              {groups.map((g) => (
                <div key={g.label} className="ds-side-group">
                  <div className="ds-side-grouplabel">{g.label}</div>
                  {g.items.map((s) => (
                    <div
                      key={s.session_id}
                      className={"ds-conv" + (s.session_id === activeSessionId ? " active" : "")}
                    >
                      <button
                        type="button"
                        className="ds-conv-btn"
                        onClick={() => openSession(s.session_id)}
                        title={s.title}
                      >
                        <span className="t">{s.title || "(未命名会话)"}</span>
                      </button>
                      <button
                        type="button"
                        className="ds-conv-del"
                        title="删除会话"
                        onClick={() => onDelete(s.session_id)}
                      >
                        <IconTrash size={13} />
                      </button>
                    </div>
                  ))}
                </div>
              ))}
            </nav>

            <div className="ds-side-foot">
              {user ? (
                <>
                  <span className="ds-avatar">{user.slice(0, 1).toUpperCase()}</span>
                  <span className="ds-user-name" title={user}>
                    {user}
                  </span>
                  <button type="button" className="ds-icobtn" title="退出登录" onClick={logout}>
                    <IconLogout size={16} />
                  </button>
                </>
              ) : (
                <button
                  type="button"
                  className="ds-login-btn"
                  onClick={() => setLoginOpen(true)}
                >
                  <IconUser size={16} />
                  登录 / 注册
                </button>
              )}
              <ThemeToggle />
            </div>
          </div>
        </aside>
        {sidebarOpen && <div className="ds-backdrop" onClick={() => setSidebarOpen(false)} />}

        <div className="ds-main">
          <header className="ds-top">
            {/* DeepSeek: 侧栏展开时这 3 个按钮让位给侧栏自身的控件, 只保留占位 */}
            <div className={"ds-topbtns" + (sidebarOpen ? " gone" : "")}>
              <button
                type="button"
                className="ds-icobtn"
                onClick={() => setSidebarOpen(true)}
                title="打开侧栏"
              >
                <IconPanelLeft />
              </button>
              <button type="button" className="ds-icobtn" onClick={focusSearch} title="搜索对话">
                <IconSearch />
              </button>
              <button type="button" className="ds-icobtn" onClick={newChat} title="开启新对话">
                <IconNewChat />
              </button>
            </div>

            <div className="ds-switch" role="tablist" aria-label="视图切换">
              <span className={"ds-switch-thumb" + (view === "sources" ? " right" : "")} />
              <button
                type="button"
                role="tab"
                aria-selected={view === "chat"}
                className={view === "chat" ? "on" : ""}
                onClick={() => router.push("/chat")}
              >
                对话
              </button>
              <button
                type="button"
                role="tab"
                aria-selected={view === "sources"}
                className={view === "sources" ? "on" : ""}
                onClick={() => router.push("/sources")}
              >
                检索
              </button>
            </div>

            {user ? (
              <span className="status-pill ds-status" title="当前登录用户">
                <i className="dot" />
                {user}
              </span>
            ) : (
              <button
                type="button"
                className="status-pill ds-status"
                title="未登录"
                onClick={() => setLoginOpen(true)}
              >
                <i className="dot dot-warn" />
                未登录 · 点击登录
              </button>
            )}
          </header>
          <div className="ds-content">{children}</div>
        </div>
      </div>
      {loginOpen && (
        <LoginModal
          onClose={() => setLoginOpen(false)}
          onDone={() => window.location.reload()}
        />
      )}
    </ShellContext.Provider>
  );
}
