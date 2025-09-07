import * as React from "react";
import { Outlet, useLocation } from "react-router-dom";
import Header from "./Header";
import Sidebar from "./Sidebar";
import Footer from "./Footer";

const COLLAPSED = 64;   // collapsed sidebar width
const EXPANDED = 280;   // expanded sidebar width
const HEADER_OFFSET = 72;

export default function AppLayout() {
  const location = useLocation();

  const [pinned, setPinned] = React.useState<boolean>(() => {
    try { return localStorage.getItem("sidebar:pinned") === "1"; } catch { return false; }
  });
  const [peek, setPeek] = React.useState(false);

  React.useEffect(() => {
    try { localStorage.setItem("sidebar:pinned", pinned ? "1" : "0"); } catch {}
  }, [pinned]);

  const isLanding = location.pathname === "/" || location.pathname === "/landing";
  const contentLeft = !isLanding && pinned ? EXPANDED : !isLanding ? COLLAPSED : 0;

  return (
    <div className="min-h-screen text-white bg-[radial-gradient(1200px_600px_at_50%_-200px,rgba(88,101,242,.35),rgba(2,8,23,1)_60%)]">
      <Header />

      {/* Hover hot-zone to peek (main pages only) */}
      {!isLanding && (
        <div
          className="fixed left-0 z-40"
          style={{
            top: HEADER_OFFSET,
            height: `calc(100vh - ${HEADER_OFFSET}px)`,
            width: 10,
          }}
          onMouseEnter={() => setPeek(true)}
          onMouseLeave={() => setPeek(false)}
        />
      )}

      {/* Sidebar (receives peek + pin) */}
      {!isLanding && (
        <Sidebar
          pinned={pinned}
          onPinnedChange={setPinned}
          peek={peek}
          onMouseEnter={() => setPeek(true)}
          onMouseLeave={() => setPeek(false)}
          collapsedWidth={COLLAPSED}
          expandedWidth={EXPANDED}
          offsetTop={HEADER_OFFSET}   // <-- new
        />
      )}

      {/* Main content shifts only when pinned */}
      <main
        className="pt-20 pb-16 transition-[margin-left] duration-300"
        style={{ marginLeft: contentLeft }}
      >
        <div className="px-3 sm:px-6">
          <Outlet />
        </div>
      </main>

      <Footer />
    </div>
  );
}