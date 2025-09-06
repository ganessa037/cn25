import React, { useEffect, useRef, useState } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import {
  LayoutDashboard,
  FileText,
  Wallet,
  Bell,
  Car,
  Pin,
  PinOff,
} from "lucide-react";
import clsx from "clsx";

type Item = { id: string; label: string; icon: React.ComponentType<any>; path: string };

const SECTIONS: Item[] = [
  { id: "analytics",     label: "Analytics",     icon: LayoutDashboard, path: "/dashboard" },
  { id: "documents",     label: "Documents",     icon: FileText,        path: "/dashboard/documents" },
  { id: "expenses",      label: "My Expenses",   icon: Wallet,          path: "/dashboard/expenses" },
  { id: "notifications", label: "Notifications", icon: Bell,            path: "/dashboard/notifications" },
  { id: "vehicles",      label: "My Vehicle",    icon: Car,             path: "/dashboard/vehicles" },
];

// 明确 props 类型（和 AppLayout 对齐）
export type SidebarProps = {
  pinned?: boolean;                               // 父组件控制是否固定
  onPinnedChange?: (pinned: boolean) => void;     // 通知父组件切换固定
};

export default function Sidebar({ pinned = false, onPinnedChange }: SidebarProps) {
  const [hovered, setHovered] = useState(false);
  const location = useLocation();
  const navigate = useNavigate();
  const edgeRef = useRef<HTMLDivElement>(null);

  // 靠左“热区”触发浮出
  useEffect(() => {
    const el = edgeRef.current;
    if (!el) return;
    const enter = () => setHovered(true);
    const leave = () => setHovered(false);
    el.addEventListener("mouseenter", enter);
    el.addEventListener("mouseleave", leave);
    return () => {
      el.removeEventListener("mouseenter", enter);
      el.removeEventListener("mouseleave", leave);
    };
  }, []);

  // 侧边栏展开状态：固定 或 悬停其一即可展开
  const open = pinned || hovered;

  const activeByPath = (path: string) => {
    // /dashboard 精确匹配，其他用 startsWith
    if (path === "/dashboard") return location.pathname === "/dashboard";
    return location.pathname.startsWith(path);
  };

  return (
    <>
      {/* 左侧 8px 热区，用于“靠近自动浮出” */}
      <div ref={edgeRef} className="fixed left-0 top-0 h-full w-2 z-40" />

      <aside
        className={clsx(
          "fixed top-20 left-0 z-50",
          "transition-all duration-300",
          open ? "translate-x-0" : "-translate-x-72"
        )}
        aria-label="Sidebar"
      >
        <div
          className={clsx(
            "w-72 h-[calc(100vh-6rem)]",
            "rounded-r-2xl border border-white/10",
            "bg-white/10 backdrop-blur-xl",
            "shadow-2xl shadow-black/30",
            "p-4 flex flex-col gap-4"
          )}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
        >
          {/* 顶部：标题 + Pin 按钮 */}
          <div className="flex items-center justify-between px-1">
            <div className="text-white/80 font-semibold tracking-wide">Navigation</div>
            <button
              onClick={() => onPinnedChange?.(!pinned)}
              className={clsx(
                "h-9 w-9 rounded-lg grid place-content-center",
                "border border-white/10",
                "bg-white/10 hover:bg-white/15 text-white/80 hover:text-white",
                "transition"
              )}
              aria-label={pinned ? "Unpin sidebar" : "Pin sidebar"}
              title={pinned ? "Unpin" : "Pin"}
            >
              {pinned ? <PinOff className="w-4 h-4" /> : <Pin className="w-4 h-4" />}
            </button>
          </div>

          {/* 导航列表 */}
          <nav className="flex-1 space-y-1 overflow-y-auto pr-2">
            {SECTIONS.map(({ id, label, icon: Icon, path }) => {
              const active = activeByPath(path);
              return (
                <button
                  key={id}
                  onClick={() => navigate(path)}
                  className={clsx(
                    "w-full flex items-center gap-3 px-3 py-2 rounded-lg text-left",
                    "text-white/80 hover:text-white hover:bg-white/10",
                    active && "bg-white/15 text-white"
                  )}
                >
                  <span className="w-8 h-8 rounded-xl bg-white/10 flex items-center justify-center">
                    <Icon className="w-4 h-4" />
                  </span>
                  <span className="truncate">{label}</span>
                </button>
              );
            })}
          </nav>

          {/* 底部：帮助/版权之类（可选） */}
          <div className="text-[12px] text-white/50 px-1">
            © {new Date().getFullYear()} KeretaKu
          </div>
        </div>
      </aside>
    </>
  );
}