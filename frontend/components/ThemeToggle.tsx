"use client";
import { useEffect, useState } from "react";

type Theme = "dark" | "light";

/**
 * 深 / 浅色主题切换。
 *
 * 初始值为何写死 "dark" 而非直接读 DOM:
 * SSR 阶段没有 document 可用, 首帧必须给一个确定值, 否则服务端与客户端渲染
 * 结果不一致会触发 hydration mismatch。真实主题由 layout.tsx 的 inline script
 * 在首帧绘制前写入 data-theme, 本组件在 useEffect 中再把它同步进本地 state。
 */
export default function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>("dark");

  useEffect(() => {
    const cur = document.documentElement.getAttribute("data-theme");
    if (cur === "light" || cur === "dark") setTheme(cur);
  }, []);

  function toggle() {
    const next: Theme = theme === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    try {
      localStorage.setItem("pilkb-theme", next);
    } catch {
      /* 隐私模式下 localStorage 可能禁止写入, 忽略即可 */
    }
    setTheme(next);
  }

  return (
    <button
      type="button"
      className="theme-toggle"
      onClick={toggle}
      title={theme === "dark" ? "切到浅色模式" : "切到深色模式"}
      aria-label="深浅色主题切换"
    >
      {theme === "dark" ? "☾" : "☀"}
    </button>
  );
}
