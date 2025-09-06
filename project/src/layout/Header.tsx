import React from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { User as UserIcon, Settings, LogOut } from "lucide-react";

export default function Header() {
  const navigate = useNavigate();
  const loc = useLocation();

  const authed = !!localStorage.getItem("token");
  const user = (() => {
    try { return JSON.parse(localStorage.getItem("user") || "null"); } catch { return null; }
  })();

  const onDashboard = loc.pathname.startsWith("/dashboard");

  const logout = () => {
    localStorage.removeItem("token");
    localStorage.removeItem("refreshToken");
    localStorage.removeItem("user");
    navigate("/signin");
  };

  const scrollTo = (id: string) => {
    if (!onDashboard) navigate(`/dashboard#${id}`);
    setTimeout(() => {
      const el = document.getElementById(id);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 0);
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-black/80 backdrop-blur-xl border-b border-white/10">
      <div className="container mx-auto px-4 py-3">
        <div className="flex items-center justify-between">
          {/* brand */}
          <Link to="/" className="flex items-center space-x-2">
            <div className="w-8 h-8 rounded-lg bg-gradient-to-r from-blue-500 to-purple-600 flex items-center justify-center">🚗</div>
            <span className="text-white font-bold text-xl">KeretaKu</span>
          </Link>

          {/* left nav */}
          <nav className="hidden md:flex items-center gap-6">
            {!authed && (
              <>
                <a href="/#features" className="text-white/80 hover:text-white">Features</a>
                <a href="/#security" className="text-white/80 hover:text-white">Security</a>
                <a href="/#faq" className="text-white/80 hover:text-white">FAQ</a>
              </>
            )}
          </nav>

          {/* right actions */}
          <div className="flex items-center gap-2">
            {authed ? (
              <>
                <Link to="/settings" className="p-2 rounded-lg hover:bg-white/10" aria-label="Settings">
                  <Settings className="w-5 h-5 text-white" />
                </Link>
                <Link to="/profile" className="p-0.5 rounded-full hover:ring-2 ring-white/30" aria-label="Profile">
                  {user?.picture ? (
                    <img src={user.picture} alt="avatar" className="w-8 h-8 rounded-full object-cover" />
                  ) : (
                    <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
                      <UserIcon className="w-5 h-5 text-white" />
                    </div>
                  )}
                </Link>
                <button onClick={logout} className="p-2 rounded-lg hover:bg-white/10" aria-label="Sign out">
                  <LogOut className="w-5 h-5 text-white" />
                </button>
              </>
            ) : (
              <>
                <Link to="/signin" className="hidden md:inline-block px-4 py-2 rounded-lg border border-white/20 text-white/90 hover:bg-white/10">Sign In</Link>
                <Link to="/signin" className="ml-2 inline-block px-4 py-2 rounded-lg bg-gradient-to-r from-blue-600 to-purple-600 text-white font-medium hover:from-blue-700 hover:to-purple-700">Get Started</Link>
              </>
            )}
          </div>

          {/* mobile burger（保持原样占位） */}
          <button className="md:hidden text-white" aria-label="Menu">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
}