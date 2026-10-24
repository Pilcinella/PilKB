"use client";
import { useEffect, useRef, useState } from "react";
import { devToken, login, register, setStoredToken } from "@/lib/api";

/**
 * 登录/注册弹窗 (居中卡片). 生产环境 (APP_ENV != dev) 后端不颁发 dev-token,
 * 自动免登录不可用, 这里是正式账号的唯一入口.
 *
 * 成功后由调用方 (AppShell) 整页 reload: 页面组件在挂载时就把 token 装进了
 * state, 不重载它们读不到新 token. 不能靠 pilkb:auth 事件驱动 reload ——
 * ensureDevToken 在每次挂载时也会 setStoredToken 并广播该事件, 会死循环.
 */
export default function LoginModal({
  onClose,
  onDone,
}: {
  onClose: () => void;
  /** 登录/注册成功后的动作 (AppShell 用来 reload) */
  onDone: () => void;
}) {
  const [tab, setTab] = useState<"login" | "register">("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [email, setEmail] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);
  const firstRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    firstRef.current?.focus();
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") onClose();
    }
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  async function submit() {
    if (busy || !username.trim() || !password) return;
    setErr("");
    setBusy(true);
    try {
      if (tab === "login") {
        const resp = await login(username.trim(), password);
        setStoredToken(resp.access_token, username.trim());
      } else {
        if (!email.trim()) throw new Error("注册需要填写邮箱");
        const resp = await register(username.trim(), email.trim(), password);
        setStoredToken(resp.access_token, username.trim());
      }
      onDone();
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  /** 开发模式免登录: 生产环境后端 403, 报错文案里能看出原因 */
  async function quickDev() {
    if (busy) return;
    setErr("");
    setBusy(true);
    try {
      const resp = await devToken();
      setStoredToken(resp.access_token, "dev-user");
      onDone();
    } catch (e: any) {
      setErr(`dev-token 获取失败: ${e?.message ?? e}`);
    } finally {
      setBusy(false);
    }
  }

  function onKeydown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.nativeEvent.isComposing) submit();
  }

  return (
    <div
      className="login-mask"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="login-card" role="dialog" aria-modal="true" aria-label="登录 PilKB">
        <div className="login-title">
          Pil<span className="grad">KB</span>
        </div>
        <div className="login-sub">
          {tab === "login" ? "登录以继续使用知识库" : "创建新账号"}
        </div>

        <div className="login-tabs" role="tablist">
          <button
            type="button"
            role="tab"
            aria-selected={tab === "login"}
            className={tab === "login" ? "on" : ""}
            onClick={() => {
              setTab("login");
              setErr("");
            }}
          >
            登录
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "register"}
            className={tab === "register" ? "on" : ""}
            onClick={() => {
              setTab("register");
              setErr("");
            }}
          >
            注册
          </button>
        </div>

        <input
          ref={firstRef}
          type="text"
          placeholder="用户名"
          value={username}
          onChange={(e) => setUsername(e.target.value)}
          onKeyDown={onKeydown}
        />
        {tab === "register" && (
          <input
            type="email"
            placeholder="邮箱"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            onKeyDown={onKeydown}
          />
        )}
        <input
          type="password"
          placeholder="密码"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          onKeyDown={onKeydown}
        />

        {err && <div className="login-err">{err}</div>}

        <button
          className="login-submit"
          onClick={submit}
          disabled={busy || !username.trim() || !password || (tab === "register" && !email.trim())}
        >
          {busy ? "…" : tab === "login" ? "登录" : "注册并登录"}
        </button>
        <button type="button" className="login-dev" onClick={quickDev} disabled={busy}>
          开发模式免登录进入（仅本地）
        </button>
      </div>
    </div>
  );
}
