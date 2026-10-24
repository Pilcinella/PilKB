"use client";
import { useEffect, useRef, useState, type ReactNode } from "react";
import { IconChevronDown } from "@/components/icons";

export type PillItem = { value: string; label: string };

/**
 * 胶囊形下拉: 仿 DeepSeek 输入框下方的"深度思考/联网搜索"按钮形态, 但功能是选择。
 * 菜单**向下弹出**; composer 固定在视口底部、下方放不下时, 把整个输入区上移让位
 * (测菜单实际高度后对 .ds-composer-zone 施加 translateY)。非 composer 场景
 * (如 sources 页的检索条) 下方空间充足, 直接向下弹, 不做位移。
 * menuFooter 用于挂自定义尾部内容 (如知识库的"新建"入口), 在菜单容器内,
 * 不会触发点击外部关闭。
 */
export default function PillSelect({
  value,
  onChange,
  items,
  icon,
  title,
  menuFooter,
}: {
  value: string;
  onChange: (v: string) => void;
  items: PillItem[];
  icon?: ReactNode;
  title?: string;
  menuFooter?: ReactNode;
}) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    function onDoc(e: MouseEvent) {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) setOpen(false);
    }
    function onKey(e: KeyboardEvent) {
      if (e.key === "Escape") setOpen(false);
    }
    document.addEventListener("mousedown", onDoc);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDoc);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // 向下弹出 + 让位: 菜单渲染后量高度, 视口底部装不下就把 composer 整体上移。
  // 用 ResizeObserver 而不是只量一次 —— "新建知识库"的输入行展开时菜单会长高。
  // 关闭时清掉位移, composer 带过渡滑回原位。
  useEffect(() => {
    if (!open) return;
    const zone = rootRef.current?.closest<HTMLElement>(".ds-composer-zone");
    const menu = menuRef.current;
    if (!zone || !menu) return;
    const apply = () => {
      const btnBottom = rootRef.current?.getBoundingClientRect().bottom ?? 0;
      const spaceBelow = window.innerHeight - btnBottom;
      const need = menu.offsetHeight + 24 - spaceBelow; // 24 = 菜单间隙 + 底部余量
      zone.style.transform = need > 0 ? `translateY(${-need}px)` : "";
    };
    apply();
    const ro = new ResizeObserver(apply);
    ro.observe(menu);
    return () => {
      ro.disconnect();
      zone.style.transform = "";
    };
  }, [open]);

  const current = items.find((i) => i.value === value);

  return (
    <div className={"pill-select" + (open ? " open" : "")} ref={rootRef}>
      <button type="button" className="pill-btn" title={title} onClick={() => setOpen((o) => !o)}>
        {icon}
        <span className="pill-label">{current?.label ?? value}</span>
        <IconChevronDown size={14} className={"pill-caret" + (open ? " up" : "")} />
      </button>
      {open && (
        <div className="pill-menu" role="listbox" ref={menuRef}>
          {items.map((it) => (
            <button
              type="button"
              key={it.value}
              role="option"
              aria-selected={it.value === value}
              className={"pill-item" + (it.value === value ? " active" : "")}
              onClick={() => {
                onChange(it.value);
                setOpen(false);
              }}
            >
              {it.label}
            </button>
          ))}
          {menuFooter}
        </div>
      )}
    </div>
  );
}
