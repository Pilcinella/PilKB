"use client";
import { useEffect, useState } from "react";
import {
  SessionDetail,
  deleteSession,
  ensureDevToken,
  getStoredUser,
  getSession,
  listSessions,
} from "@/lib/api";

function fmtTime(ts: number): string {
  if (!ts) return "-";
  const d = new Date(ts * 1000);
  return d.toLocaleString("zh-CN", { hour12: false });
}

export default function HistoryPage() {
  const [token, setToken] = useState("");
  const [user, setUser] = useState("");
  const [sessions, setSessions] = useState<SessionDetail[] | null>(null);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [error, setError] = useState("");
  const [ready, setReady] = useState(false);

  async function refresh() {
    if (!token) return;
    try {
      const list = await listSessions(token);
      const details: SessionDetail[] = await Promise.all(
        list.map((m) => getSession(token, m.session_id)),
      );
      setSessions(details);
      if (!activeId && details.length) setActiveId(details[0].session_id);
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }

  useEffect(() => {
    ensureDevToken().then((t) => { setToken(t); setUser(getStoredUser()); setReady(true); });
  }, []);

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  async function onDelete(id: string) {
    if (!confirm("确定要删除这个会话吗？")) return;
    try {
      await deleteSession(token, id);
      setActiveId(null);
      refresh();
    } catch (e: any) {
      setError(e?.message ?? String(e));
    }
  }

  if (!ready) {
    return (
      <div className="ds-scrollpage">
        <div className="ds-page">
          <div className="card">
            <div className="muted">正在初始化...</div>
          </div>
        </div>
      </div>
    );
  }

  const active = sessions?.find((s) => s.session_id === activeId);

  return (
    <div className="ds-scrollpage">
      <div className="ds-page history-page">
      <div className="page-head">
        <h1>
          会话历史 <span className="grad">History</span>
        </h1>
        <p className="muted">{user} · 共 {sessions?.length ?? 0} 个会话</p>
      </div>

      {error && <div className="err">{error}</div>}

      <div className="hist-layout">
        <aside className="hist-list card">
          {sessions === null && <div className="muted">正在加载...</div>}
          {sessions && sessions.length === 0 && <div className="muted">暂无会话, 去 Chat 发起第一次提问吧</div>}
          {sessions?.map((s) => (
            <button
              key={s.session_id}
              className={"hist-item" + (s.session_id === activeId ? " active" : "")}
              onClick={() => setActiveId(s.session_id)}
            >
              <div className="title">{s.title}</div>
              <div className="meta muted">
                {s.message_count} 条 · {fmtTime(s.updated_at)}
              </div>
            </button>
          ))}
        </aside>

        <section className="hist-detail card">
          {!active && <div className="muted">点选左侧会话查看详情</div>}
          {active && (
            <>
              <div className="row" style={{ justifyContent: "space-between" }}>
                <h3 style={{ margin: 0 }}>{active.title}</h3>
                <button className="ghost" onClick={() => onDelete(active.session_id)}>删除会话</button>
              </div>
              <div className="muted mono" style={{ fontSize: 11 }}>{active.session_id}</div>
              <div className="msgs">
                {active.messages.map((m, i) => (
                  <div key={i} className={"bubble " + m.role}>
                    <div className="role">{m.role === "user" ? "用户" : "助手"}</div>
                    <div className="content">{m.content}</div>
                    {m.sources && m.sources.length > 0 && (
                      <details className="sources">
                        <summary>引用 {m.sources.length} 条</summary>
                        {m.sources.map((s, j) => (
                          <div key={j} className="source">{s.content}</div>
                        ))}
                      </details>
                    )}
                  </div>
                ))}
                {active.messages.length === 0 && <div className="muted">暂无消息</div>}
              </div>
            </>
          )}
        </section>
      </div>
    </div>
    </div>
  );
}
