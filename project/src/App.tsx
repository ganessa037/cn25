import * as React from "react";
import { Routes, Route, Navigate, useLocation, useNavigate } from "react-router-dom";
import AppLayout from "./layout/AppLayout";

import LandingPage from "./pages/LandingPage";
import SignInPage from "./pages/SignInPage";

// Default main dashboard overview
import DashboardPage from "./pages/DashboardPage";

// Feature pages
import Analytics from "./features/Dashboard/Analytics";
import VehicleManager from "./features/Dashboard/VehicleManager";
import DocumentManager from "./features/Dashboard/DocumentManager";
import ExpenseTracker from "./features/Dashboard/ExpenseTracker";
import Notifications from "./features/Dashboard/Notifications";

/** Try to parse a JSON or base64-JSON user blob from a query param */
function parseUserParam(val?: string | null) {
  if (!val) return {};
  const candidates = [val, decodeURIComponent(val)];
  for (const c of candidates) {
    try { return JSON.parse(c); } catch {}
    try { return JSON.parse(atob(c)); } catch {}
  }
  return {};
}

/** Protected wrapper that auto-captures ?token from the URL once and then guards routes */
function Protected({ element }: { element: JSX.Element }) {
  const location = useLocation();
  const navigate = useNavigate();
  const [ready, setReady] = React.useState(false);
  const [isAuthed, setAuthed] = React.useState(false);

  React.useEffect(() => {
    // 1) Check if already signed in
    let currentToken: string | null = null;
    try {
      currentToken = JSON.parse(localStorage.getItem("user") || "{}")?.token ?? null;
    } catch {}

    // 2) If not, look for ?token in the URL once (backend redirects like /dashboard?token=...)
    if (!currentToken) {
      const usp = new URLSearchParams(location.search);
      const token = usp.get("token") || usp.get("jwt");
      const next = usp.get("next") || ""; // optional
      const userObj = parseUserParam(usp.get("user"));

      if (token) {
        try {
          localStorage.setItem("user", JSON.stringify({ token, ...userObj }));
        } catch {}
        // Clean the URL and optionally respect ?next
        const target = next || location.pathname;
        window.history.replaceState({}, "", target);
        currentToken = token;
      }
    }

    setAuthed(!!currentToken);
    setReady(true);
  }, [location.pathname, location.search]);

  if (!ready) return null; // brief, prevents flicker

  return isAuthed ? element : <Navigate to="/signin" replace />;
}

export default function App() {
  return (
    <Routes>
      {/* Public routes */}
      <Route path="/" element={<LandingPage />} />
      <Route path="/landing" element={<LandingPage />} />
      <Route path="/signin" element={<SignInPage />} />

      {/* Protected shell (Header + Sidebar + Footer) */}
      <Route element={<AppLayout />}>
        {/* DEFAULT dashboard overview */}
        <Route
          path="/dashboard"
          element={<Protected element={<DashboardPage />} />}
        />

        {/* Analytics page (review-only; CRUD stays in ExpenseTracker) */}
        <Route
          path="/dashboard/analytics"
          element={<Protected element={<Analytics />} />}
        />

        {/* Other feature pages */}
        <Route
          path="/dashboard/vehicles"
          element={<Protected element={<VehicleManager />} />}
        />
        <Route
          path="/dashboard/documents"
          element={<Protected element={<DocumentManager />} />}
        />
        <Route
          path="/dashboard/expenses"
          element={<Protected element={<ExpenseTracker />} />}
        />
        <Route
          path="/dashboard/notifications"
          element={<Protected element={<Notifications />} />}
        />
      </Route>

      {/* Fallback */}
      <Route path="*" element={<Navigate to="/landing" replace />} />
    </Routes>
  );
}