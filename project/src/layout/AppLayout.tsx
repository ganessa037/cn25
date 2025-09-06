import { Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import Header from "./Header";
import Footer from "./Footer";
import Sidebar from "./Sidebar";

export default function AppLayout() {
  const [pinned, setPinned] = useState(
    () => localStorage.getItem("sidebar:pinned") === "1"
  );
  useEffect(() => {
    localStorage.setItem("sidebar:pinned", pinned ? "1" : "0");
  }, [pinned]);

  return (
    <div className="min-h-screen text-white bg-[linear-gradient(120deg,#0b0f14,#0e1017)]">
      <Header />
      <div className="relative">
        {/* 让 Sidebar 接收 pinned 与 onPinnedChange */}
        <Sidebar pinned={pinned} onPinnedChange={setPinned} />
        <main
          className={`transition-all duration-300 pt-24 pb-16 px-6 ${
            pinned ? "ml-72" : "ml-0"
          }`}
        >
          <Outlet />
        </main>
      </div>
      <Footer />
    </div>
  );
}