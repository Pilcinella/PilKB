// 统一的 API 客户端, 自动带 token (HttpOnly cookie + credentials: include)
// 防御性 trim：NEXT_PUBLIC_API_BASE 若被复制粘贴带了尾随空格 / 换行，模板字面量拼出
// "http://localhost:8000 /api/v1/..." 会让 fetch 抛 TypeError。
// 同时去掉尾随 /，避免 "http://localhost:8000//api/v1/..." 这种双斜杠。
const API = (process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000")
  .trim()
  .replace(/\/+$/, "");

// 本地演示默认知识库（已预置示例文档，免登录即可对话）
// 可用 NEXT_PUBLIC_DEMO_KB 覆盖: 录屏 / 多套环境演示时指向独立知识库,
// 避免把开发时随手上传的文档(个人简历等)混进演示画面。
export const DEMO_KB = (process.env.NEXT_PUBLIC_DEMO_KB || "kb-demo")
  .trim();

export type Source = {
  doc_id?: string;
  chunk_id: string;
  content: string;
  score: number;
  metadata?: Record<string, unknown>;
};

export type ChatMode = "rag" | "agent";

export type SessionMeta = {
  session_id: string;
  title: string;
  kb_id: string;
  mode: string;
  created_at: number;
  updated_at: number;
  message_count: number;
};

export type HistoryMessage = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  ts: number;
};

export type SessionDetail = SessionMeta & {
  messages: HistoryMessage[];
};

// P2-11: 优先用 HttpOnly cookie 传输 JWT; 同时保留 Bearer header 兼容.
// fetch 设 credentials: 'include', 让浏览器自动带上后端写的 cookie.
// 对已有 token (例如从 localStorage 迁移 / dev-token 手动输入) 仍走 Authorization.
function authHeaders(token?: string): HeadersInit {
  const h: HeadersInit = {};
  if (token) {
    h["Authorization"] = `Bearer ${token}`;
  }
  return h;
}

const fetchInit: RequestInit = { credentials: "include" };

async function jsonOrThrow<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const text = await res.text();
    throw new Error(`${res.status} ${res.statusText}: ${text}`);
  }
  return res.json() as Promise<T>;
}

// ----- Auth -----
export async function register(username: string, email: string, password: string): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/register`, {
    ...fetchInit,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, email, password }),
  }));
}

export async function login(username: string, password: string): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/login`, {
    ...fetchInit,
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  }));
}

export async function logout(): Promise<void> {
  await fetch(`${API}/api/v1/auth/logout`, {
    ...fetchInit,
    method: "POST",
  });
}

export async function devToken(): Promise<{ access_token: string }> {
  return jsonOrThrow(await fetch(`${API}/api/v1/auth/dev-token`, { ...fetchInit, method: "POST" }));
}

// ----- Chat (SSE 流式) -----
export async function* streamChat(params: {
  token?: string;
  question: string;
  session_id?: string;
  kb_id?: string;
  top_k?: number;
  mode?: ChatMode;
}): AsyncGenerator<{ type: string; data: unknown }> {
  const res = await fetch(`${API}/api/v1/chat/stream`, {
    ...fetchInit,
    method: "POST",
    headers: { "Content-Type": "application/json", ...authHeaders(params.token) },
    body: JSON.stringify({
      question: params.question,
      session_id: params.session_id,
      kb_id: params.kb_id ?? DEMO_KB,
      top_k: params.top_k ?? 5,
      mode: params.mode ?? "rag",
    }),
  });
  if (!res.body) throw new Error("SSE: no body");
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += dec.decode(value, { stream: true });
    const lines = buf.split("\n\n");
    buf = lines.pop() || "";
    for (const chunk of lines) {
      const line = chunk.trim();
      if (!line.startsWith("data:")) continue;
      const payload = line.slice(5).trim();
      if (!payload) continue;
      try {
        yield JSON.parse(payload);
      } catch {
        // ignore malformed line
      }
    }
  }
}

// ----- Knowledge bases -----
export type KbInfo = {
  kb_id: string;
  owner: string;
  is_public: boolean;
  can_write: boolean;
};

/** 列出当前用户可访问的知识库 (自己的 + 公共白名单). */
export async function listKbs(token: string): Promise<KbInfo[]> {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/kbs`, { ...fetchInit, headers: authHeaders(token) }),
  );
}

/** 显式创建知识库; 校验 / 归属 / 每用户上限由后端裁决 (400/403 抛 Error). */
export async function createKb(
  token: string,
  kb_id: string,
): Promise<{ status: "created" | "exists"; kb_id: string; owner: string; is_public: boolean }> {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/kbs`, {
      ...fetchInit,
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify({ kb_id }),
    }),
  );
}

/** 删除整个知识库 (向量库 + BM25 索引 + 注册表一并清除; 仅 owner / 公共库管理员). 不可恢复. */
export async function deleteKb(token: string, kb_id: string) {
  const r = await jsonOrThrow<{ status: string; kb_id: string }>(
    await fetch(`${API}/api/v1/kbs/${encodeURIComponent(kb_id)}`, {
      ...fetchInit,
      method: "DELETE",
      headers: authHeaders(token),
    }),
  );
  // 广播给所有已挂载的知识库下拉 (chat/sources/upload 各有一个 KbPill):
  // 删掉的库必须立即从选项里消失, 否则用户点到它只会收到 404.
  // 与 pilkb:auth 同一套 window 事件模式 (组件之间没有父子关系).
  if (typeof window !== "undefined") window.dispatchEvent(new Event("pilkb:kbs"));
  return r;
}

// ----- Documents -----
export async function uploadDocument(token: string, file: File, kb_id = DEMO_KB) {
  const fd = new FormData();
  fd.append("file", file);
  return jsonOrThrow<{ doc_id: string; filename: string; chunks: number; status: string; message?: string }>(
    await fetch(`${API}/api/v1/documents/upload?kb_id=${encodeURIComponent(kb_id)}`, {
      ...fetchInit,
      method: "POST",
      headers: authHeaders(token),
      body: fd,
    }),
  );
}

export type KbDocument = {
  doc_id: string;
  filename: string;
  user: string;
  chunks: number;
};

export type KbStats = {
  kb_id: string;
  bm25_count: number;
  vector_count: number;
  /** 两份索引是否一致; false 表示 BM25 与向量库脱钩 (关键词检索已降级). */
  consistent: boolean;
  /** 当前用户能否触发重建 (私有 kb 要 owner, 公共 kb 要管理员). */
  can_rebuild: boolean;
  /** 当前用户能否写入该 kb (公共 kb / 非 owner 为 false, 此时不显示删除入口). */
  can_write: boolean;
  /** kb 内已入库的文档 (按 doc_id 聚合, 来自向量库). */
  documents: KbDocument[];
};

export async function listDocuments(token: string, kb_id = DEMO_KB): Promise<KbStats> {
  return jsonOrThrow<KbStats>(
    await fetch(`${API}/api/v1/documents/list?kb_id=${encodeURIComponent(kb_id)}`, {
      ...fetchInit,
      headers: authHeaders(token),
    }),
  );
}

/** 删除文档及其全部 chunk (破坏性操作, 调用方负责二次确认). */
export async function deleteDocument(token: string, doc_id: string, kb_id = DEMO_KB) {
  return jsonOrThrow<{ status: string; doc_id: string; kb_id: string; chunks_deleted: number }>(
    await fetch(
      `${API}/api/v1/documents/${encodeURIComponent(doc_id)}?kb_id=${encodeURIComponent(kb_id)}`,
      { ...fetchInit, method: "DELETE", headers: authHeaders(token) },
    ),
  );
}

/** 从向量库侧回灌重建 BM25 索引. 幂等; 权限不足时 403. */
export async function rebuildBm25(token: string, kb_id = DEMO_KB) {
  return jsonOrThrow<{ status: string; kb_id: string; chunks_before: number; chunks_restored: number }>(
    await fetch(`${API}/api/v1/documents/rebuild-bm25?kb_id=${encodeURIComponent(kb_id)}`, {
      ...fetchInit,
      method: "POST",
      headers: authHeaders(token),
    }),
  );
}

// ----- History -----
export async function listSessions(token: string): Promise<SessionMeta[]> {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history`, { ...fetchInit, headers: authHeaders(token) }),
  );
}

export async function getSession(token: string, session_id: string): Promise<SessionDetail> {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history/${session_id}`, { ...fetchInit, headers: authHeaders(token) }),
  );
}

export async function createSession(token: string, payload: { title?: string; kb_id?: string; mode?: string } = {}) {
  return jsonOrThrow<{ session_id: string }>(
    await fetch(`${API}/api/v1/history`, {
      ...fetchInit,
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify(payload),
    }),
  );
}

export async function appendMessage(token: string, session_id: string, msg: HistoryMessage) {
  return jsonOrThrow<{ session_id: string; message_count: number }>(
    await fetch(`${API}/api/v1/history/${session_id}/append`, {
      ...fetchInit,
      method: "POST",
      headers: { "Content-Type": "application/json", ...authHeaders(token) },
      body: JSON.stringify(msg),
    }),
  );
}

export async function deleteSession(token: string, session_id: string) {
  return jsonOrThrow(
    await fetch(`${API}/api/v1/history/${session_id}`, {
      ...fetchInit,
      method: "DELETE",
      headers: authHeaders(token),
    }),
  );
}

export const API_BASE = API;

// ----- 客户端 token 存储 (与 AuthBar 共享) -----
// 免登录演示: 直接把后端 /auth/dev-token 颁发的 token 存到 localStorage,
// 之后所有请求用 Authorization: Bearer 带上 (跨域下比 HttpOnly cookie 更可靠).
const USER_KEY = "pilkb.user";
const TOKEN_KEY = "pilkb.token";

export function getStoredToken(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(TOKEN_KEY) || "";
}

export function getStoredUser(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(USER_KEY) || "";
}

export function setStoredToken(token: string, user: string): void {
  if (typeof window === "undefined") return;
  if (token) localStorage.setItem(TOKEN_KEY, token);
  localStorage.setItem(USER_KEY, user);
  window.dispatchEvent(new Event("pilkb:auth"));
}

export function clearStoredAuth(): void {
  if (typeof window === "undefined") return;
  localStorage.removeItem(USER_KEY);
  localStorage.removeItem(TOKEN_KEY);
  window.dispatchEvent(new Event("pilkb:auth"));
  // cookie 由后端 /auth/logout 清除; 这里也尝试触发一下 (可选)
  fetch(`${API_BASE}/api/v1/auth/logout`, { ...fetchInit, method: "POST" }).catch(() => {});
}

// ----- KB 选择持久化 (与 token 存储同模式) -----
// 用户上次选的知识库, 跨页面 / 刷新后沿用; 未选择时各页回退 DEMO_KB.
const KB_KEY = "pilkb.kb_id";

export function getStoredKbId(): string {
  if (typeof window === "undefined") return "";
  return localStorage.getItem(KB_KEY) || "";
}

export function setStoredKbId(kbId: string): void {
  if (typeof window === "undefined") return;
  if (kbId) localStorage.setItem(KB_KEY, kbId);
}

// 缓存 token 的过期预检: 后端签发的 JWT 恒带 exp, 页面加载时本地就能判断,
// 不必等第一个请求 401. base64url 要换回标准字母表才能 atob; 非 ASCII 的 sub
// 会解成乱码, 但多字节序列里不会出现 ASCII 引号/反斜杠, 只读 exp 不受影响。
// 解析失败按已过期处理, 走重新获取。
function cachedTokenExpired(token: string): boolean {
  try {
    const b64 = token.split(".")[1].replace(/-/g, "+").replace(/_/g, "/");
    const payload = JSON.parse(atob(b64));
    return typeof payload.exp === "number" && payload.exp * 1000 <= Date.now();
  } catch {
    return true;
  }
}

// 免登录: 自动向后端拿 dev token 并缓存, 已缓存且未过期则直接返回. 失败返回空串.
// 之前只要有缓存就复用且不校验 —— token 过了有效期(默认 24h)后每个请求都 401,
// 页面永远不自愈, 只能手动清 localStorage. 注: 缓存的若是注册用户的过期 token,
// 这里会静默换成 dev-user (演示优先); 正式账号体系上线时应改为引导重新登录。
export async function ensureDevToken(): Promise<string> {
  if (typeof window === "undefined") return "";
  const existing = localStorage.getItem(TOKEN_KEY);
  if (existing && !cachedTokenExpired(existing)) return existing;
  if (existing) localStorage.removeItem(TOKEN_KEY);
  try {
    const resp = await devToken();
    setStoredToken(resp.access_token, "dev-user");
    return resp.access_token;
  } catch (e) {
    console.error("[pilkb] 自动获取 dev token 失败:", e);
    return "";
  }
}
