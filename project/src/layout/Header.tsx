import * as React from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { Settings, User, LogOut } from "lucide-react";
import logoUrl from "../assets/keretaku-icon.png";

/** Signed-in check (reads localStorage.user.token) */
function isAuthed(): boolean {
  try {
    return !!JSON.parse(localStorage.getItem("user") || "{}")?.token;
  } catch {
    return false;
  }
}

/** Smooth-scroll helper for landing sections */
function scrollToIdOrNavigate(id: string, navigate: (to: string) => void) {
  const el = document.getElementById(id);
  if (el) {
    el.scrollIntoView({ behavior: "smooth", block: "start" });
  } else {
    navigate(`/landing#${id}`);
  }
}

export default function Header() {
  const location = useLocation();
  const navigate = useNavigate();

  // Landing variant on "/" or "/landing"
  const isLanding = location.pathname === "/" || location.pathname.startsWith("/landing");
  const authed = isAuthed();
  const brandTo = authed && !isLanding ? "/dashboard" : "/landing";

  const goSettings = () => navigate("/dashboard/analytics"); // keep or change to your settings route
  const goProfile = () => navigate("/dashboard/profile");     // stub until implemented
  const exitToLanding = () => navigate("/landing", { replace: true });

  return (
    <header className="fixed top-0 left-0 right-0 z-50">
      <div className="mx-2 my-2 rounded-2xl bg-white/10 border border-white/15 backdrop-blur-xl shadow-2xl">
        <div className="h-14 px-3 sm:px-5 flex items-center justify-between">
          {/* Brand */}
          <Link to={brandTo} className="flex items-center gap-2 select-none">
            <span
              className="
                inline-grid place-items-center
                w-9 h-9 rounded-full
                bg-gradient-to-tr from-sky-400 to-violet-500
                p-[2px] ring-1 ring-white/30
              "
              aria-hidden
            >
              <span className="w-full h-full rounded-full bg-white/10 backdrop-blur-md grid place-items-center">
                {/* If you put the file in /public, use /keretaku-icon.png; otherwise import it as logoUrl */}
                <img
                  src={logoUrl}
                  alt=""
                  className="w-6 h-6 rounded-full object-contain"
                  draggable={false}
                />
              </span>
            </span>

            <span className="text-white font-semibold tracking-wide">MyKereta</span>
          </Link>

          {/* Center content varies by page */}
          {isLanding ? (
            <div className="hidden md:flex flex-col items-center justify-center">
              <nav className="flex items-center gap-6 text-white/90">
                <button
                  className="hover:text-white transition"
                  onClick={() => scrollToIdOrNavigate("features", navigate)}
                >
                  Features
                </button>
                <button
                  className="hover:text-white transition"
                  onClick={() => scrollToIdOrNavigate("security", navigate)}
                >
                  Security
                </button>
                <button
                  className="hover:text-white transition"
                  onClick={() => scrollToIdOrNavigate("faq", navigate)}
                >
                  FAQ
                </button>
              </nav>
            
            </div>
          ) : (
            <div /> // keep spacing symmetrical on app pages
          )}

          {/* Right controls */}
          {isLanding ? (
            // On landing we keep it minimal; optionally show Sign In
            <div className="flex items-center gap-2">
              <button
                onClick={() => navigate("/signin")}
                className="px-3 h-10 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 text-white/90"
                title="Sign in"
              >
                Sign In
              </button>
            </div>
          ) : (
            // App header (main pages)
            <div className="flex items-center gap-2">
              <button
                title="Settings"
                aria-label="Settings"
                onClick={goSettings}
                className="w-10 h-10 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 text-white/90"
              >
                <Settings className="w-5 h-5 mx-auto" />
              </button>

              <button
                title="Profile"
                aria-label="Profile"
                onClick={goProfile}
                className="w-10 h-10 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 text-white/90"
              >
                <User className="w-5 h-5 mx-auto" />
              </button>

              <button
                title="Exit to Landing"
                aria-label="Exit to Landing"
                onClick={exitToLanding}
                className="w-10 h-10 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40 text-white"
              >
                <LogOut className="w-5 h-5 mx-auto" />
              </button>
            </div>
          )}
        </div>
      </div>
    </header>
  );
}