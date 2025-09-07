import * as React from "react";
import { useLocation, useNavigate } from "react-router-dom";

function apiBase() {
  const raw =
    (import.meta as any).env?.VITE_API_URL ||
    (import.meta as any).env?.VITE_BACKEND_URL ||
    "http://127.0.0.1:3000";
  const t = String(raw).replace(/\/$/, "");
  return t.endsWith("/api") ? t : `${t}/api`;
}
function parseSearch(search: string) {
  const p = new URLSearchParams(search);
  const obj: Record<string, string> = {};
  p.forEach((v, k) => (obj[k] = v));
  return obj;
}
function parseUserParam(val?: string | null) {
  if (!val) return {};
  const candidates = [val, decodeURIComponent(val)];
  for (const c of candidates) {
    try { return JSON.parse(c); } catch {}
    try { return JSON.parse(atob(c)); } catch {}
  }
  return {};
}

export default function AuthCallback() {
  const nav = useNavigate();
  const loc = useLocation();

  React.useEffect(() => {
    const params = parseSearch(loc.search);
    const token = params.token || params.jwt || "";
    const userObj = parseUserParam(params.user);

    if (token) {
      const payload = {
        token,
        ...userObj, // { id, name, email, picture } if provided
      };
      try {
        localStorage.setItem("user", JSON.stringify(payload));
      } catch {}
      // Clean URL
      window.history.replaceState({}, "", "/dashboard");
      nav("/dashboard", { replace: true });
      return;
    }

    // Fallback: if your backend uses httpOnly cookies only (no token),
    // you can fetch /auth/me here to confirm login and build a local user.
    (async () => {
      try {
        const res = await fetch(`${apiBase()}/auth/me`, { credentials: "include" });
        if (res.ok) {
          const me = await res.json();
          try { localStorage.setItem("user", JSON.stringify({ ...me })); } catch {}
          window.history.replaceState({}, "", "/dashboard");
          nav("/dashboard", { replace: true });
        } else {
          nav("/signin", { replace: true });
        }
      } catch {
        nav("/signin", { replace: true });
      }
    })();
  }, [nav, loc.search]);

  return (
    <div className="min-h-[calc(100vh-5rem)] grid place-items-center">
      <div className="max-w-md w-full bg-white/10 border border-white/10 rounded-2xl p-6 backdrop-blur-xl shadow-xl">
        Completing sign-in…
      </div>
    </div>
  );
}