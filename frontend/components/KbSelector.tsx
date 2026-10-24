"use client";
import { useCallback, useEffect, useState } from "react";
import { KbInfo, createKb, listKbs } from "@/lib/api";

/**
 * 知识库选择器: 下拉列出"我的 + 公共"知识库, 附内联新建入口.
 *
 * 列表来自 GET /api/v1/kbs, 新建走 POST /api/v1/kbs —— 校验 / 归属 / 每用户
 * 上限全部由后端裁决, 这里只展示错误信息, 不做前端预检.
 *
 * 当前 kbId 不在列表里时 (比如 DEMO_KB 被别的用户占用、或列表加载失败)
 * 仍作为额外选项保留: 静默切到别的库会让检索来源悄悄变化, 比显示一个
 * 可能 403 的选项更糟; 用户点它触发请求时后端会给出明确的 403.
 */
export default function KbSelector({
  token,
  kbId,
  onChange,
}: {
  token: string;
  kbId: string;
  onChange: (kbId: string) => void;
}) {
  const [kbs, setKbs] = useState<KbInfo[]>([]);
  const [creating, setCreating] = useState(false);
  const [newId, setNewId] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState("");

  const refresh = useCallback(async () => {
    if (!token) return;
    try {
      setKbs(await listKbs(token));
      setErr("");
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    }
  }, [token]);

  useEffect(() => {
    refresh();
  }, [refresh]);

  async function submitNew() {
    const id = newId.trim();
    if (!id || !token) return;
    setBusy(true);
    setErr("");
    try {
      await createKb(token, id);
      await refresh();
      onChange(id);
      setNewId("");
      setCreating(false);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setBusy(false);
    }
  }

  const known = kbs.some((k) => k.kb_id === kbId);

  return (
    <span className="row" style={{ gap: 6 }}>
      <select
        value={kbId}
        onChange={(e) => onChange(e.target.value)}
        title="知识库 (检索 / 上传目标)"
        style={{ maxWidth: 180 }}
      >
        {!known && <option value={kbId}>{kbId}（不在列表）</option>}
        {kbs.map((k) => (
          <option key={k.kb_id} value={k.kb_id}>
            {k.kb_id}
            {k.is_public ? "（公共）" : ""}
          </option>
        ))}
      </select>
      {creating ? (
        <>
          <input
            type="text"
            placeholder="字母/数字/_/-"
            value={newId}
            autoFocus
            onChange={(e) => setNewId(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submitNew()}
            style={{ width: 140, flex: "0 0 auto" }}
          />
          <button onClick={submitNew} disabled={busy || !newId.trim()}>
            {busy ? "..." : "创建"}
          </button>
          <button
            className="ghost"
            onClick={() => {
              setCreating(false);
              setNewId("");
              setErr("");
            }}
          >
            取消
          </button>
        </>
      ) : (
        <button className="ghost" onClick={() => setCreating(true)} title="新建知识库">
          ＋ 新建
        </button>
      )}
      {err && (
        <span className="err" title={err}>
          ⚠
        </span>
      )}
    </span>
  );
}
