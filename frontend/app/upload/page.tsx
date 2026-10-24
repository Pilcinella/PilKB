"use client";
import { Suspense, useEffect, useRef, useState } from "react";
import { useSearchParams } from "next/navigation";
import {
  DEMO_KB,
  deleteDocument,
  deleteKb,
  ensureDevToken,
  getStoredKbId,
  listDocuments,
  rebuildBm25,
  setStoredKbId,
  uploadDocument,
  type KbStats,
} from "@/lib/api";
import KbPill from "@/components/KbPill";
import { IconPlus, IconX } from "@/components/icons";

type DocRow = {
  filename: string;
  chunks: number;
  status: string;
  doc_id: string;
  message?: string;
};

// 与后端 kb_registry._KB_ID_RE 同一套白名单 (交给用户控制的 kb_id 走严格校验),
// 前端先挡一道, 后端仍是最终裁决.
const NEW_KB_RE = /^[A-Za-z0-9_-]{1,64}$/;

function UploadView() {
  // ?new=1 直接进入"新建知识库"模式 —— chat / sources 页的"新建知识库"入口
  // 跳转带上; URL 记住模式, 刷新不丢. 模式内切换用 replaceState 同步地址栏,
  // 但不经过 Next 路由 (searchParams 不变), 因此不会与本 effect 打架.
  const searchParams = useSearchParams();
  const wantNew = searchParams.get("new") === "1";
  const [mode, setMode] = useState<"existing" | "new">(wantNew ? "new" : "existing");

  const [token, setToken] = useState("");
  // 上传目标知识库: 与 chat / sources 页共用 localStorage 里的选择, 兜底 DEMO_KB.
  const [kbId, setKbId] = useState(DEMO_KB);
  // 新建模式的库名. 刻意不提前调 createKb: 后端 ensure_kb_for_write 会在首个
  // 文档入库时自动建库并归属当前用户 —— "只命名不上传"因此永远不会产生空库.
  const [newKbId, setNewKbId] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [rows, setRows] = useState<DocRow[]>([]);
  const [stats, setStats] = useState<KbStats | null>(null);
  const [err, setErr] = useState("");
  const [ready, setReady] = useState(false);
  // 建库成功的反馈 (ok-box), 与 err 分开: 成功路径不该是红的
  const [createdMsg, setCreatedMsg] = useState("");
  // 重建索引: 二次确认 -> 执行 -> 反馈. confirming 是内联确认态 (不用
  // window.confirm: 全量重建是 O(N) 操作, 值得一次明确的确认动作,
  // 但原生弹窗在演示里太糙).
  const [confirming, setConfirming] = useState(false);
  const [rebuilding, setRebuilding] = useState(false);
  const [rebuildMsg, setRebuildMsg] = useState("");
  // 删除: 记录处于确认态的行 (一次只确认一个, 避免多行同时待确认)
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const [deleting, setDeleting] = useState(false);
  // 删除整个知识库: 与文档删除同款内联二次确认 (不可恢复, 多一道防误触)
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  const [deletingKb, setDeletingKb] = useState(false);
  const [deletedMsg, setDeletedMsg] = useState("");
  // refresh 的请求序号, 用于丢弃快速切换知识库时的过期响应
  const refreshSeq = useRef(0);

  const nameOk = NEW_KB_RE.test(newKbId.trim());

  useEffect(() => {
    setMode(wantNew ? "new" : "existing");
  }, [wantNew]);

  useEffect(() => {
    setKbId(getStoredKbId() || DEMO_KB);
    ensureDevToken().then((t) => { setToken(t); setReady(true); });
  }, []);

  async function refresh() {
    if (!token) return;
    // 请求序号: 快速切换知识库时, 旧库的慢响应不能覆盖新库的结果 (否则界面
    // 会显示"误差: 列表属于上一个库, 错误属于当前库").
    const seq = ++refreshSeq.current;
    // 每次刷新先清掉上一次的错误: 切换知识库后, 上一个库的 404/403 会一直
    // 挂在页面底部 (实测切到别的库后 "知识库不存在: kb-demo2" 仍在), 看起来
    // 像新库也坏了.
    setErr("");
    try {
      const s = await listDocuments(token, kbId);
      if (seq !== refreshSeq.current) return;
      setStats(s);
    } catch (e: any) {
      if (seq !== refreshSeq.current) return;
      // 失败时清掉旧列表: 否则界面同时显示"另一个库的文档"和一条错误,
      // 用户会以为这些文档属于当前(加载失败的)库.
      setStats(null);
      setErr(e?.message ?? String(e));
    }
  }

  useEffect(() => {
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, kbId]);

  async function doDelete(doc_id: string) {
    if (!token || deleting) return;
    setDeleting(true);
    setErr("");
    setRebuildMsg("");
    try {
      await deleteDocument(token, doc_id, kbId);
      setPendingDelete(null);
      // 同步"本次上传结果"里那一行的状态: 否则下方表格仍显示 indexed,
      // 用户会怀疑到底删掉没有 (那是操作历史, 不会随着列表刷新而变)
      setRows((rs) =>
        rs.map((r) =>
          r.doc_id === doc_id ? { ...r, status: "deleted", message: "已从知识库删除" } : r,
        ),
      );
      await refresh(); // 列表以服务端为准, 不在本地做乐观删除
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setDeleting(false);
    }
  }

  async function doRebuild() {
    if (!token || rebuilding) return;
    setRebuilding(true);
    setErr("");
    setRebuildMsg("");
    try {
      const r = await rebuildBm25(token, kbId);
      setRebuildMsg(
        `索引已重建: 恢复 ${r.chunks_restored} 个 chunk（重建前 ${r.chunks_before} 个）`,
      );
      setConfirming(false);
      await refresh(); // 让告警条随真实状态消失, 而不是本地猜测
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setRebuilding(false);
    }
  }

  /** 切换 上传到已有库 / 新建知识库, 并把模式同步到地址栏 (刷新/后退不丢) */
  function switchMode(next: "existing" | "new") {
    setMode(next);
    setErr("");
    setCreatedMsg("");
    try {
      window.history.replaceState(null, "", next === "new" ? "/upload?new=1" : "/upload");
    } catch {
      /* replaceState 失败只影响 URL 同步, 不影响功能 */
    }
  }

  async function doDeleteKb() {
    if (!token || deletingKb) return;
    setDeletingKb(true);
    setErr("");
    setRebuildMsg("");
    setCreatedMsg("");
    // 删除进行中不要有刷新请求回来把 stats 写回去
    refreshSeq.current++;
    try {
      await deleteKb(token, kbId);
      setConfirmingDelete(false);
      setDeletedMsg(`知识库 ${kbId} 已删除（向量库 + BM25 索引一并清除）。`);
      // 已删的库不再展示旧列表; 回落到公共演示库, [token, kbId] effect 会重新拉取.
      // 注意: deleteKb 内部已广播 pilkb:kbs, 所有 KbPill 会重拉列表, 已删库
      // 不会再作为选项出现.
      setStats(null);
      setStoredKbId(DEMO_KB);
      setKbId(DEMO_KB);
    } catch (e: any) {
      setErr(e?.message ?? String(e));
    } finally {
      setDeletingKb(false);
    }
  }

  async function submit() {
    if (!files.length || !token) return;
    const target = mode === "new" ? newKbId.trim() : kbId;
    if (mode === "new" && !NEW_KB_RE.test(target)) return;
    setBusy(true);
    setErr("");
    setCreatedMsg("");
    const next: DocRow[] = [];
    for (const f of files) {
      try {
        const r = await uploadDocument(token, f, target);
        next.push({ filename: r.filename, chunks: r.chunks, status: r.status, doc_id: r.doc_id, message: r.message });
      } catch (e: any) {
        next.push({ filename: f.name, chunks: 0, status: "error", doc_id: "", message: e?.message ?? String(e) });
      }
    }
    const okCount = next.filter((r) => r.status !== "error").length;
    setRows((r) => [...next, ...r]);
    setFiles([]);
    setBusy(false);
    setRebuildMsg("");
    setConfirming(false);
    setPendingDelete(null);

    if (mode === "new") {
      if (okCount > 0) {
        // 首个文档入库成功 => 知识库已被后端创建. 选中它 (chat/sources 也会
        // 沿用 localStorage 的选择), 回到常规模式; 文档表/统计随 [token, kbId]
        // effect 自动刷新 —— 这里不手动调 refresh, 闭包里的 kbId 还是旧值.
        setStoredKbId(target);
        setKbId(target);
        setNewKbId("");
        setMode("existing");
        setCreatedMsg(`知识库 ${target} 已创建，${okCount} 个文档入库成功。`);
        try {
          window.history.replaceState(null, "", "/upload");
        } catch {
          /* ignore */
        }
      } else {
        setErr("没有文档入库成功（失败原因见下方“本次上传结果”），可调整文件后重试。");
      }
      return;
    }
    refresh();
  }

  function drop(e: React.DragEvent<HTMLDivElement>) {
    e.preventDefault();
    const list = Array.from(e.dataTransfer.files || []);
    setFiles((prev) => [...prev, ...list]);
  }

  if (!ready) {
    return (
      <div className="ds-scrollpage">
        <div className="ds-page">
          <div className="card">
            <div className="muted">正在准备...</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="ds-scrollpage">
      <div className="ds-page upload-page">
      <div className="page-head">
        <h1>
          上传文档 <span className="grad">Upload</span>
        </h1>
        <p className="muted">拖拽文件即可入库：解析 → 切块 → BM25 + 向量双写；新建知识库也在这里完成</p>
      </div>
      <div className="card pop">
        <h3>拖拽文档到下方区域</h3>
        <div
          className="dropzone"
          onDragOver={(e) => e.preventDefault()}
          onDrop={drop}
          onClick={() => document.getElementById("file-input")?.click()}
        >
          <input
            id="file-input"
            type="file"
            multiple
            hidden
            onChange={(e) => setFiles(Array.from(e.target.files || []))}
          />
          <span className="muted">支持 PDF / Markdown / DOCX / TXT / 图片</span>
        </div>
        {files.length > 0 && (
          <ul className="filelist">
            {files.map((f, i) => (
              <li key={i}>
                {f.name} <span className="muted">{(f.size / 1024).toFixed(1)} KB</span>
              </li>
            ))}
          </ul>
        )}

        {/* 上传目标: 已有库下拉 + 新建入口 | 新建模式的命名输入 */}
        {mode === "new" ? (
          <div className="newkb-box">
            <div className="row" style={{ flexWrap: "wrap" }}>
              <input
                type="text"
                className="newkb-name"
                placeholder="新知识库 ID（字母/数字/_/-，1-64 位）"
                value={newKbId}
                autoFocus
                onChange={(e) => setNewKbId(e.target.value)}
              />
              <button className="ghost" onClick={() => switchMode("existing")}>
                <IconX size={14} /> 取消新建
              </button>
            </div>
            {newKbId.trim() !== "" && !nameOk && (
              <div className="err" style={{ marginTop: 8 }}>
                知识库 ID 仅允许字母、数字、下划线与连字符，长度 1–64 位
              </div>
            )}
            <div className="muted" style={{ marginTop: 8 }}>
              知识库随首个文档入库而创建并归属你；只命名不上传则不会创建。
            </div>
          </div>
        ) : (
          <div className="row" style={{ marginTop: 12, flexWrap: "wrap" }}>
            <KbPill
              token={token}
              kbId={kbId}
              onChange={(next) => {
                if (!next || next === kbId) return;
                // 切换目标库: 清掉上一个库留下的错误/成功提示, 它们与当前库无关
                setErr("");
                setDeletedMsg("");
                setCreatedMsg("");
                setConfirmingDelete(false);
                setKbId(next);
                setStoredKbId(next);
              }}
            />
            <button className="ghost" onClick={() => switchMode("new")}>
              <IconPlus size={14} /> 新建知识库
            </button>
          </div>
        )}

        <div className="row" style={{ marginTop: 12, flexWrap: "wrap" }}>
          <button
            onClick={submit}
            disabled={
              busy ||
              files.length === 0 ||
              (mode === "new" && !nameOk) ||
              (mode === "existing" && stats !== null && !stats.can_write)
            }
          >
            {busy
              ? "上传中..."
              : mode === "new"
                ? `上传并创建知识库（${files.length} 个文档）`
                : `上传 ${files.length} 个文件`}
          </button>
          {mode === "existing" && stats && (
            <span className="muted">
              当前知识库: BM25 {stats.bm25_count} chunks · 向量 {stats.vector_count} chunks
            </span>
          )}
        </div>

        {/* 公共 kb 只读: 后端 ensure_kb_for_write 一律 403, 与其让用户点出
            一排失败行, 不如直接禁用按钮并说明去哪传. stats 为 null (列表尚未
            返回 / 失败) 时不拦, 由后端兜底裁决, 错误会显示在"本次上传结果"里. */}
        {mode === "existing" && stats && !stats.can_write && (
          <div className="muted" style={{ marginTop: 8 }}>
            当前是只读知识库（公共库），上传请切换到自己的知识库或点"新建知识库"。
          </div>
        )}

        {/* 索引不一致: 两份存储 chunk 数不等 (pkl 损坏被隔离 / 多进程丢更新).
            检索仍可用 (退化为纯向量路), 所以用温和语气而非报错 —— 但必须让
            用户知道"关键词类问题会答不准"这个后果. */}
        {mode === "existing" && stats && !stats.consistent && (
          <div className="index-warn">
            {/* ⚠ 带 VS16 才是 emoji 呈现; 单个 U+26A0 在缺字体的环境下会渲染成方块 */}
            <span className="iw-icon">{"⚠️"}</span>
            <div className="iw-body">
              <div className="iw-title">搜索索引不完整</div>
              <div className="iw-sub">
                关键词索引 {stats.bm25_count} 条 / 向量索引 {stats.vector_count} 条。
                语义检索不受影响，但编号、专有名词这类精确匹配的问题可能搜不到内容。
              </div>
            </div>
            <div className="iw-actions">
              {stats.can_rebuild ? (
                confirming ? (
                  <>
                    <button onClick={doRebuild} disabled={rebuilding}>
                      {rebuilding ? "重建中..." : "确认重建"}
                    </button>
                    <button
                      className="ghost"
                      onClick={() => setConfirming(false)}
                      disabled={rebuilding}
                    >
                      取消
                    </button>
                  </>
                ) : (
                  <button className="ghost" onClick={() => setConfirming(true)}>
                    重建搜索索引
                  </button>
                )
              ) : (
                <span className="muted">请联系管理员重建</span>
              )}
            </div>
          </div>
        )}
        {rebuildMsg && <div className="ok-box">{rebuildMsg}</div>}
        {createdMsg && <div className="ok-box">{createdMsg}</div>}
        {deletedMsg && <div className="ok-box">{deletedMsg}</div>}
        {err && <div className="err">{err}</div>}
      </div>

      {stats && (
        <div className="card">
          <div className="row" style={{ justifyContent: "space-between", alignItems: "baseline" }}>
            <h3 style={{ margin: 0 }}>
              知识库文档 <span className="muted">共 {stats.documents.length} 篇</span>
            </h3>
            {/* 删除整个知识库: 与重建同一权限口径 (can_rebuild = 私有库 owner / 公共库
                管理员), 后端 DELETE /kbs 复用同一判据; 内联二次确认, 删除不可恢复 */}
            {stats.can_rebuild &&
              (confirmingDelete ? (
                <span className="row">
                  <button onClick={doDeleteKb} disabled={deletingKb}>
                    {deletingKb ? "删除中..." : `确认删除 (${kbId})`}
                  </button>
                  <button
                    className="ghost"
                    onClick={() => setConfirmingDelete(false)}
                    disabled={deletingKb}
                  >
                    取消
                  </button>
                </span>
              ) : (
                <button className="ghost danger" onClick={() => setConfirmingDelete(true)}>
                  删除知识库
                </button>
              ))}
          </div>
          {stats.documents.length === 0 ? (
            <div className="muted">该知识库还没有文档，拖拽文件到上方即可入库。</div>
          ) : (
            <table className="tbl">
              <thead>
                <tr>
                  <th>文件名</th>
                  <th>chunks</th>
                  <th>doc_id</th>
                  {stats.can_write && <th>操作</th>}
                </tr>
              </thead>
              <tbody>
                {stats.documents.map((d) => (
                  <tr key={d.doc_id}>
                    <td>{d.filename || <span className="muted">(未命名)</span>}</td>
                    <td>{d.chunks}</td>
                    <td className="mono">{d.doc_id}</td>
                    {stats.can_write && (
                      <td>
                        {pendingDelete === d.doc_id ? (
                          <span className="row">
                            <button onClick={() => doDelete(d.doc_id)} disabled={deleting}>
                              {deleting ? "删除中..." : `确认删除 (${d.chunks})`}
                            </button>
                            <button
                              className="ghost"
                              onClick={() => setPendingDelete(null)}
                              disabled={deleting}
                            >
                              取消
                            </button>
                          </span>
                        ) : (
                          <button
                            className="ghost"
                            onClick={() => setPendingDelete(d.doc_id)}
                            disabled={deleting}
                          >
                            删除
                          </button>
                        )}
                      </td>
                    )}
                  </tr>
                ))}
              </tbody>
            </table>
          )}
          {!stats.can_write && stats.documents.length > 0 && (
            <div className="muted" style={{ marginTop: 8 }}>
              只读知识库：公共知识库不开放删除，如需修改请联系管理员。
            </div>
          )}
        </div>
      )}

      {rows.length > 0 && (
        <div className="card">
          <h3>本次上传结果</h3>
          <table className="tbl">
            <thead>
              <tr>
                <th>文件名</th>
                <th>chunks</th>
                <th>状态</th>
                <th>doc_id</th>
                <th>消息</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr
                  key={i}
                  className={r.status === "error" ? "err-row" : r.status === "deleted" ? "muted-row" : ""}
                >
                  <td>{r.filename}</td>
                  <td>{r.chunks}</td>
                  <td>{r.status}</td>
                  <td className="mono">{r.doc_id || "-"}</td>
                  <td className="muted">{r.message}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
    </div>
  );
}

export default function UploadPage() {
  // useSearchParams 需要上层 Suspense 兜底, 否则 next build 静态预渲染会报错
  return (
    <Suspense fallback={null}>
      <UploadView />
    </Suspense>
  );
}
