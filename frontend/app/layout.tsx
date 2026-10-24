import "./globals.css";
import type { Metadata } from "next";
import { Inter } from "next/font/google";
import AppShell from "@/components/AppShell";

// next/font 构建期即下载 Inter 并自托管到 /_next/static/fonts,
// 运行时不再访问 Google Fonts (国内网络亦稳定), 且自动附带 font-display: swap.
const inter = Inter({
  subsets: ["latin"],
  variable: "--font-inter",
  display: "swap",
});

export const metadata: Metadata = {
  title: "PilKB · 企业知识库",
  description: "Multi-Agent + MCP + Hybrid Search",
};

// 首帧绘制前同步写入 data-theme, 免得刷新瞬间深浅色闪烁.
// 选 inline script 而非 useEffect: 后者要等 hydration 完成才执行, 挡不住闪烁.
const themeInitScript = `
(function () {
  try {
    var t = localStorage.getItem("pilkb-theme");
    if (t !== "light" && t !== "dark") t = "dark";
    document.documentElement.setAttribute("data-theme", t);
  } catch (e) {
    document.documentElement.setAttribute("data-theme", "dark");
  }
})();
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="zh-CN" className={inter.variable} data-theme="dark" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: themeInitScript }} />
      </head>
      <body>
        <AppShell>{children}</AppShell>
      </body>
    </html>
  );
}
