import { Navigate, Outlet, useLocation } from "react-router-dom";

export default function ProtectedRoute({ children }: { children?: JSX.Element }) {
  const loc = useLocation();
  let token: string | null = null;
  try {
    token = JSON.parse(localStorage.getItem("user") || "null")?.token ?? null;
  } catch {}

  if (!token) {
    const next = encodeURIComponent(loc.pathname + loc.search + loc.hash);
    return <Navigate to={`/signin?next=${next}`} replace />;
  }
  return children ?? <Outlet />;
}