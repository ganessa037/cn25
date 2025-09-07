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

export default function SignInPage() {
  const nav = useNavigate();
  const loc = useLocation();

  // If backend redirected back to /signin?token=... handle it here
  React.useEffect(() => {
    const params = parseSearch(loc.search);
    const token = params.token || params.jwt || "";
    const user = parseUserParam(params.user);
    if (token) {
      try {
        localStorage.setItem("user", JSON.stringify({ token, ...user }));
      } catch {}
      window.history.replaceState({}, "", "/dashboard");
      nav("/dashboard", { replace: true });
    }
  }, [loc.search, nav]);

  const startGoogle = () => {
    window.location.href = `${apiBase()}/auth/google`;
  };

  return (
    <div className="min-h-[calc(100vh-5rem)] grid place-items-center">
      <div className="max-w-md w-full bg-white/10 border border-white/10 rounded-2xl p-6 backdrop-blur-xl shadow-xl text-center">
        <div className="text-2xl font-semibold">Sign in</div>
        <div className="text-white/80 mt-1">Use Google to continue</div>
        <button
          onClick={startGoogle}
          className="mt-6 w-full h-11 rounded-xl bg-white/20 hover:bg-white/25 border border-white/30 text-white font-medium"
        >
          Continue with Google
        </button>
      </div>
    </div>
  );
}