import {
  BrowserRouter,
  Routes,
  Route,
  Navigate,
  useLocation,
  useNavigate,
} from "react-router-dom";

import AppLayout from "./layout/AppLayout";
import LandingPage from "./pages/LandingPage";
import SignInPage from "./pages/SignInPage";

// 模块页（先放宽为 any，保证路由跑通；等都用 API 拉数据后可移除 any）
import AnalyticsComp from "./features/Dashboard/Analytics";
import DocumentManagerComp from "./features/Dashboard/DocumentManager";
import ExpenseTrackerComp from "./features/Dashboard/ExpenseTracker";
import NotificationsComp from "./features/Dashboard/Notifications";
import VehicleManagerComp from "./features/Dashboard/VehicleManager";

const Analytics: any = AnalyticsComp as any;
const DocumentManager: any = DocumentManagerComp as any;
const ExpenseTracker: any = ExpenseTrackerComp as any;
const Notifications: any = NotificationsComp as any;
const VehicleManager: any = VehicleManagerComp as any;

function AuthGate({ children }: { children: JSX.Element }) {
  const location = useLocation();
  const navigate = useNavigate();

  // 1) 统一截获 token（不管回到 /signin 还是 /dashboard）
  const params = new URLSearchParams(location.search);
  const token = params.get("token");
  if (token) {
    const current = JSON.parse(localStorage.getItem("user") || "null") || {};
    localStorage.setItem("user", JSON.stringify({ ...current, token }));
    navigate(location.pathname, { replace: true });
    return null; // 下一帧会重新渲染
  }

  // 2) 常规鉴权
  const user = JSON.parse(localStorage.getItem("user") || "null");
  return user?.token ? children : <Navigate to="/signin" replace />;
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* 公开页 */}
        <Route path="/" element={<LandingPage />} />
        <Route path="/signin" element={<SignInPage />} />

        {/* 受保护页：放进布局（Header + Sidebar + Footer） */}
        <Route element={<AuthGate><AppLayout /></AuthGate>}>
          <Route path="/dashboard" element={<Analytics />} />
          <Route path="/dashboard/documents" element={<DocumentManager />} />
          <Route path="/dashboard/expenses" element={<ExpenseTracker />} />
          <Route path="/dashboard/notifications" element={<Notifications />} />
          <Route path="/dashboard/vehicles" element={<VehicleManager />} />
        </Route>

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}