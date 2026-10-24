"use client";
import { useCallback, useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { KbInfo, listKbs } from "@/lib/api";
import PillSelect, { type PillItem } from "@/components/PillSelect";
import { IconDatabase, IconPlus } from "@/components/icons";

/**
 * 知识库胶囊下拉 (composer / 检索页 / 上传页共用): 列表来自 GET /api/v1/kbs.
 *
 * "新建知识库"不再就地创建 —— 建库与传文档天然是一件事 (空库没有意义),
 * 入口统一跳转 /upload?new=1: 在上传页里命名 + 上传一体完成, 且只在首个
 * 文档入库时才真正建库 (不上传文档不建库)。
 *
 * 当前 kbId 不在列表里时 (比如 DEMO_KB 被别的用户占用、或列表加载失败)
 * 仍作为额外选项保留: 静默切到别的库会让检索来源悄悄变化, 比显示一个
 * 可能 403 的选项更糟; 用户点它触发请求时后端会给出明确的 403.
 */
export default function KbPill({
  token,
  kbId,
  onChange,
}: {
  token: string;
  kbId: string;
  onChange: (kbId: string) => void;
}) {
  const router = useRouter();
  const [kbs, setKbs] = useState<KbInfo[]>([]);
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

  // 知识库被创建/删除后立即重拉列表 (删除发生在别的组件里, 这里收不到 props
  // 变化; lib/api.deleteKb 成功后广播 pilkb:kbs, 与 pilkb:auth 同一模式).
  useEffect(() => {
    const onChanged = () => refresh();
    window.addEventListener("pilkb:kbs", onChanged);
    return () => window.removeEventListener("pilkb:kbs", onChanged);
  }, [refresh]);

  const items: PillItem[] = [];
  if (!kbs.some((k) => k.kb_id === kbId)) {
    items.push({ value: kbId, label: `${kbId}（不在列表）` });
  }
  for (const k of kbs) {
    items.push({ value: k.kb_id, label: k.kb_id + (k.is_public ? "（公共）" : "") });
  }

  return (
    <PillSelect
      value={kbId}
      onChange={onChange}
      items={items}
      icon={<IconDatabase size={15} />}
      title="知识库（检索目标）"
      menuFooter={
        <div className="pill-foot">
          <button
            type="button"
            className="pill-item"
            onClick={() => router.push("/upload?new=1")}
          >
            <IconPlus size={14} />
            新建知识库（去上传文档）
          </button>
          {err && (
            <div className="pill-err" title={err}>
              {"⚠️"} {err}
            </div>
          )}
        </div>
      }
    />
  );
}
