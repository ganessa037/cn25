import { useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";

export default function SignInPage() {
  const navigate = useNavigate();

  // 已有 token 就直接进仪表盘
  useEffect(() => {
    const u = JSON.parse(localStorage.getItem("user") || "null");
    if (u?.token) navigate("/dashboard", { replace: true });
  }, [navigate]);

  const startGoogle = () => {
    const base = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:3000";
    window.location.href = `${base}/api/auth/google/start`;
  };

  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="w-[680px] rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl p-10">
        <div className="flex flex-col items-center gap-6">
          <div className="h-20 w-20 rounded-2xl bg-gradient-to-br from-indigo-500 to-fuchsia-500 grid place-content-center text-4xl">🚗</div>
          <h1 className="text-4xl font-semibold">Welcome Back</h1>
          <p className="text-white/70">Sign in to continue to your dashboard</p>

          <button
            onClick={startGoogle}
            className="w-full h-16 rounded-xl bg-white/10 hover:bg-white/15 transition
                       border border-white/10 text-lg font-semibold"
          >
            Use Google
          </button>

          <Link to="/" className="text-indigo-300 hover:underline mt-4">
            Back to Home
          </Link>
        </div>
      </div>
    </div>
  );
}