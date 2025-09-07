import * as React from "react";
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

/** Navigation items (Settings removed; Analytics added) */
type NavItem = {
  id: string;
  label: string;
  path: string;
  icon: React.ComponentType<React.SVGProps<SVGSVGElement>>;
};

const NAV: NavItem[] = [
  { id: "vehicles",      label: "Vehicles",      path: "/dashboard/vehicles",     icon: Car },
  { id: "documents",     label: "Documents",     path: "/dashboard/documents",    icon: FileText },
  { id: "expenses",      label: "Expenses",      path: "/dashboard/expenses",     icon: Wallet },
  { id: "notifications", label: "Notifications", path: "/dashboard/notifications", icon: Bell },

  // Replaces the old “Settings” slot
  { id: "analytics",     label: "Analytics",     path: "/dashboard/analytics",     icon: LayoutDashboard },
];

export type SidebarProps = {
  /** Pin state from AppLayout */
  pinned?: boolean;
  onPinnedChange?: (p: boolean) => void;

  /** Backward-compat props (safe to ignore if not passed) */
  setPinned?: (p: boolean) => void;
  peek?: boolean;
  onMouseEnter?: () => void;
  onMouseLeave?: () => void;

  /** Aligns the sidebar right under the Header; default matches h-14 + my-2 */
  offsetTop?: number;

  /** Widths from AppLayout; used for sizing and translate distance */
  collapsedWidth?: number;   // not used for transform; content shift handled in AppLayout
  expandedWidth?: number;    // used here; default 280
};

export default function Sidebar(props: SidebarProps) {
  const {
    pinned: pinnedProp = false,
    onPinnedChange,
    setPinned,
    peek,
    onMouseEnter,
    onMouseLeave,
    offsetTop = 72,
    expandedWidth = 280,
  } = props;

  const location = useLocation();
  const navigate = useNavigate();

  // Local hover with a small “grace close” delay to avoid flicker
  const [hover, setHover] = React.useState(false);
  const closeTimer = React.useRef<number | null>(null);

  const setHoverSafe = (v: boolean) => {
    if (closeTimer.current) {
      window.clearTimeout(closeTimer.current);
      closeTimer.current = null;
    }
    if (!v) {
      closeTimer.current = window.setTimeout(() => setHover(false), 120);
    } else {
      setHover(true);
    }
  };

  React.useEffect(() => () => {
    if (closeTimer.current) window.clearTimeout(closeTimer.current);
  }, []);

  // Use parent-provided peek if present; otherwise our own hover
  const peekOpen = typeof peek === "boolean" ? peek : hover;
  const isOpen = pinnedProp || peekOpen;

  const togglePin = () => {
    const next = !pinnedProp;
    onPinnedChange?.(next);
    setPinned?.(next);
  };

  const active = (path: string) =>
    path === "/dashboard"
      ? location.pathname === "/dashboard"
      : location.pathname.startsWith(path);

  return (
    // Wrapper owns both hot-zone and panel to prevent boundary flicker
    <div
      className="fixed left-0 bottom-0 z-50"
      style={{ top: offsetTop }}
      onMouseEnter={() => { setHoverSafe(true); onMouseEnter?.(); }}
      onMouseLeave={() => { setHoverSafe(false); onMouseLeave?.(); }}
    >
      {/* Hot-zone, disabled while open to avoid ping-pong */}
      <div
        aria-hidden
        className="absolute left-0 top-0"
        style={{
          width: 10,
          height: `calc(100vh - ${offsetTop}px)`,
          pointerEvents: isOpen ? "none" : "auto",
        }}
      />

      {/* Sidebar panel */}
      <aside
        className="relative transition-transform duration-300 ease-out"
        style={{
          width: expandedWidth,
          height: `calc(100vh - ${offsetTop}px)`,
          transform: `translateX(${isOpen ? 0 : -expandedWidth}px)`,
          willChange: "transform",
        }}
        aria-label="Sidebar"
      >
        <div
          className={clsx(
            "h-full rounded-r-2xl border border-white/15",
            "bg-white/10 backdrop-blur-lg",
            "shadow-xl shadow-black/30",
            "p-4 flex flex-col gap-4"
          )}
        >
          {/* Brand + pin */}
          <div className="flex items-center justify-between px-1">
            <div className="flex items-center gap-2">
              <div className="inline-flex items-center justify-center w-8 h-8 rounded-xl bg-white/20 border border-white/25 text-white font-semibold">
                🛠️
              </div>
              <div className="text-white/85 font-semibold tracking-wide">Menu</div>
            </div>
            <button
              onClick={togglePin}
              className="h-9 w-9 rounded-lg grid place-content-center border border-white/15 bg-white/10 hover:bg-white/15 text-white/85 hover:text-white transition"
              aria-label={pinnedProp ? "Unpin sidebar" : "Pin sidebar"}
              title={pinnedProp ? "Unpin" : "Pin"}
            >
              {pinnedProp ? <PinOff className="w-4 h-4" /> : <Pin className="w-4 h-4" />}
            </button>
          </div>

          {/* Nav */}
          <nav className="flex-1 space-y-1 overflow-y-auto pr-2">
            {NAV.map(({ id, label, path, icon: Icon }) => (
              <button
                key={id}
                onClick={() => navigate(path)}
                className={clsx(
                  "w-full flex items-center gap-3 px-3 py-2 rounded-xl text-left transition border",
                  active(path)
                    ? "bg-white/20 border-white/30 text-white"
                    : "bg-white/5 border-white/10 text-white/80 hover:text-white hover:bg-white/10"
                )}
              >
                <span className="w-9 h-9 rounded-lg bg-white/12 border border-white/20 flex items-center justify-center">
                  <Icon className="w-4 h-4" />
                </span>
                <span className="truncate">{label}</span>
              </button>
            ))}
          </nav>

          <div className="text-[11px] text-white/55 px-1">
            Hover to peek. Pin to keep it open. © {new Date().getFullYear()} KeretaKu
          </div>
        </div>
      </aside>
    </div>
  );
}