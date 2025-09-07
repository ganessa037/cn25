import * as React from "react";
import { useNavigate } from "react-router-dom";

/**
 * Dashboard (default page after sign in)
 * - Reads user JWT from localStorage.user.token
 * - Calls backend: /api/vehicles, /api/expenses, /api/documents
 * - All endpoints are already scoped by OAuth identity on the server
 * - Renders KPIs + summaries in a glass/frosted UI
 */

/* ------------------------------ API utils ------------------------------ */

function apiBase(): string {
  const raw =
    (import.meta as any).env?.VITE_API_URL ||
    (import.meta as any).env?.VITE_BACKEND_URL ||
    "http://127.0.0.1:3000";
  const t = String(raw).replace(/\/$/, "");
  return t.endsWith("/api") ? t : `${t}/api`;
}
const API = apiBase();

function token(): string | null {
  try {
    return JSON.parse(localStorage.getItem("user") || "{}")?.token ?? null;
  } catch {
    return null;
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    headers: {
      "Content-Type": "application/json",
      ...(token() ? { Authorization: `Bearer ${token()}` } : {}),
    },
  });
  if (!res.ok) throw new Error(await res.text().catch(() => res.statusText));
  return (await res.json()) as T;
}

/* ------------------------------- Types -------------------------------- */

type Vehicle = {
  id: string;
  brand: string;
  model?: string | null;
  year?: number | null;
  plate?: string | null;
  color?: string | null;
  fuelType?: string | null;
  currentMileage?: number | null;
  roadTaxExpiry?: string | null;
  insuranceExpiry?: string | null;
  nextServiceDate?: string | null;
  createdAt?: string;
  updatedAt?: string;
};

type Expense = {
  id: string;
  title?: string | null;
  amount: number;
  category: "Fuel" | "Maintenance" | "Insurance" | "RoadTax" | "Toll_Parking" | "Other";
  date?: string | null;
  vehicleId: string;
  createdAt?: string | null;
};

type Document = {
  id: string;
  type: string;
  name: string;
  vehicleId?: string | null;
  expiryDate?: string | null;
  uploadedAt?: string | null;
};

/* ------------------------------ Helpers ------------------------------- */

const fmtMY = new Intl.NumberFormat("en-MY", { style: "currency", currency: "MYR" });
const fmtMY0 = new Intl.NumberFormat("en-MY", { style: "currency", currency: "MYR", maximumFractionDigits: 0 });

const toISOyyyyMMdd = (d: Date | string) => {
  const base = typeof d === "string" ? new Date(d) : d;
  const local = new Date(base.getTime() - base.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
};
const daysBetween = (a: Date, b: Date) => Math.round((+b - +a) / 86400000);

function vehicleTitle(v: Vehicle) {
  const parts = [v.brand, v.model ? ` ${v.model}` : "", v.year ? ` (${v.year})` : ""];
  return parts.join("").trim();
}

function upcomingOrExpired(dateStr?: string | null, windowDays = 30) {
  if (!dateStr) return { status: "none" as const, days: 0 };
  const today = new Date();
  const d = new Date(dateStr);
  if (isNaN(+d)) return { status: "none" as const, days: 0 };
  const diff = daysBetween(today, d); // positive = in future
  if (diff < 0) return { status: "expired" as const, days: Math.abs(diff) };
  if (diff <= windowDays) return { status: "expiring" as const, days: diff };
  return { status: "ok" as const, days: diff };
}

/* -------------------------------- Page -------------------------------- */

export default function DashboardPage() {
  const nav = useNavigate();

  const [vehicles, setVehicles] = React.useState<Vehicle[]>([]);
  const [expenses, setExpenses] = React.useState<Expense[]>([]);
  const [documents, setDocuments] = React.useState<Document[]>([]);

  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const [v, e, d] = await Promise.all([
          getJson<Vehicle[]>("/vehicles"),
          getJson<Expense[]>("/expenses"),
          getJson<Document[]>("/documents"),
        ]);
        setVehicles(Array.isArray(v) ? v : []);
        setExpenses(Array.isArray(e) ? e : []);
        setDocuments(Array.isArray(d) ? d : []);
      } catch (err: any) {
        setError(err?.message || "Failed to load dashboard data");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  /* ------------------------------ KPIs -------------------------------- */

  const totalVehicles = vehicles.length;

  const now = new Date();
  const thisMonthKey = `${now.getFullYear()}-${now.getMonth() + 1}`;

  const thisMonthSpend = expenses
    .filter((x) => {
      const d = x.date ? new Date(x.date) : x.createdAt ? new Date(x.createdAt) : null;
      if (!d || isNaN(+d)) return false;
      return d.getFullYear() === now.getFullYear() && d.getMonth() === now.getMonth();
    })
    .reduce((s, x) => s + Number(x.amount || 0), 0);

  const last12Totals = (() => {
    const arr = Array.from({ length: 12 }, (_, i) => {
      const d = new Date(now);
      d.setMonth(now.getMonth() - (11 - i));
      return { key: `${d.getFullYear()}-${d.getMonth() + 1}`, total: 0 };
    });
    for (const e of expenses) {
      const d = e.date ? new Date(e.date) : e.createdAt ? new Date(e.createdAt) : null;
      if (!d || isNaN(+d)) continue;
      const key = `${d.getFullYear()}-${d.getMonth() + 1}`;
      const found = arr.find((x) => x.key === key);
      if (found) found.total += Number(e.amount || 0);
    }
    return arr;
  })();
  const avgMonthly = last12Totals.reduce((s, x) => s + x.total, 0) / (last12Totals.length || 1);

  /* ---------------------------- Alerts/renewals ------------------------ */

  type Alert = {
    id: string;
    vehicleId: string;
    vehicleTitle: string;
    plate?: string | null;
    type: "Insurance" | "Road Tax" | "Service";
    status: "expired" | "expiring";
    days: number; // days overdue or days remaining
    date: string; // ISO yyyy-mm-dd for display
  };

  const alerts: Alert[] = [];
  for (const v of vehicles) {
    const ins = upcomingOrExpired(v.insuranceExpiry);
    if (ins.status === "expired" || ins.status === "expiring") {
      alerts.push({
        id: `ins-${v.id}`,
        vehicleId: v.id,
        vehicleTitle: vehicleTitle(v),
        plate: v.plate,
        type: "Insurance",
        status: ins.status,
        days: ins.days,
        date: v.insuranceExpiry ? toISOyyyyMMdd(v.insuranceExpiry) : "",
      });
    }
    const rt = upcomingOrExpired(v.roadTaxExpiry);
    if (rt.status === "expired" || rt.status === "expiring") {
      alerts.push({
        id: `rt-${v.id}`,
        vehicleId: v.id,
        vehicleTitle: vehicleTitle(v),
        plate: v.plate,
        type: "Road Tax",
        status: rt.status,
        days: rt.days,
        date: v.roadTaxExpiry ? toISOyyyyMMdd(v.roadTaxExpiry) : "",
      });
    }
    const svc = upcomingOrExpired(v.nextServiceDate);
    if (svc.status === "expired" || svc.status === "expiring") {
      alerts.push({
        id: `svc-${v.id}`,
        vehicleId: v.id,
        vehicleTitle: vehicleTitle(v),
        plate: v.plate,
        type: "Service",
        status: svc.status,
        days: svc.days,
        date: v.nextServiceDate ? toISOyyyyMMdd(v.nextServiceDate) : "",
      });
    }
  }
  const alertsHigh = alerts.filter((a) => a.status === "expired").length;
  const alertsMedium = alerts.filter((a) => a.status === "expiring").length;
  const totalAlerts = alerts.length;

  /* ------------------------------ Expense breakdown -------------------- */

  const byCat = expenses.reduce<Record<string, number>>((acc, e) => {
    const k =
      e.category === "Toll_Parking"
        ? "Toll/Parking"
        : e.category === "RoadTax"
        ? "Road Tax"
        : e.category;
    acc[k] = (acc[k] || 0) + Number(e.amount || 0);
    return acc;
  }, {});
  const breakdown = [
    { label: "Fuel", total: byCat["Fuel"] || 0 },
    { label: "Insurance", total: byCat["Insurance"] || 0 },
    { label: "Maintenance", total: byCat["Maintenance"] || 0 },
    { label: "Road Tax", total: byCat["Road Tax"] || 0 },
    { label: "Toll/Parking", total: byCat["Toll/Parking"] || 0 },
    { label: "Other", total: byCat["Other"] || 0 },
  ].filter((x) => x.total > 0);

  const recent = expenses
    .slice()
    .sort(
      (a, b) =>
        new Date(b.date || b.createdAt || 0).getTime() -
        new Date(a.date || a.createdAt || 0).getTime()
    )
    .slice(0, 6);

  /* ------------------------------ UI ---------------------------------- */

  return (
    <div className="min-h-[calc(100vh-5rem)] px-3 sm:px-6 text-white">
      <div className="max-w-7xl mx-auto py-6 space-y-6">
        {/* Welcome / Hero */}
        <div className="rounded-2xl p-6 bg-blue-600/80 border border-white/15 text-white shadow-2xl">
          <div className="text-xl sm:text-2xl font-semibold">Welcome back!</div>
          <div className="text-white/90 mt-1">
            Here’s your vehicle management overview.
          </div>
        </div>

        {/* KPI row */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          <KPI
            icon="🚗"
            label="Total Vehicles"
            value={String(totalVehicles)}
            sub="Your registered vehicles"
          />
          <KPI
            icon="💸"
            label="This Month"
            value={fmtMY0.format(thisMonthSpend)}
            sub={toISOyyyyMMdd(new Date()).slice(0, 7)}
          />
          <KPI
            icon="⚠️"
            label="Alerts"
            value={String(totalAlerts)}
            sub={`${alertsHigh} High • ${alertsMedium} Medium`}
          />
          <KPI
            icon="📈"
            label="Avg/Month"
            value={fmtMY0.format(avgMonthly || 0)}
            sub="Last 12 months"
          />
        </div>

        {/* Vehicles & Renewals */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
          <GlassCard title="Your Vehicles" action={{ text: "View All", onClick: () => nav("/dashboard/vehicles") }}>
            {vehicles.length === 0 ? (
              <Empty>No vehicles yet. Add one to get started.</Empty>
            ) : (
              <div className="space-y-3">
                {vehicles.slice(0, 2).map((v) => (
                  <div key={v.id} className="rounded-xl bg-white/5 border border-white/10 p-4 flex items-center justify-between">
                    <div>
                      <div className="font-semibold">{vehicleTitle(v)}</div>
                      <div className="text-white/70 text-sm">{v.plate || "—"} • {v.color || v.fuelType || ""}</div>
                    </div>
                    <div className="text-right text-sm text-white/70">
                      <div>Current Mileage</div>
                      <div className="font-semibold">{(v.currentMileage ?? 0).toLocaleString()} km</div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </GlassCard>

          <GlassCard title="Upcoming Renewals" action={{ text: "Documents", onClick: () => nav("/dashboard/documents") }}>
            {alerts.length === 0 ? (
              <Empty>No upcoming renewals.</Empty>
            ) : (
              <ul className="space-y-2">
                {alerts
                  .sort((a, b) => (a.status === "expired" ? -1 : 1))
                  .slice(0, 4)
                  .map((a) => (
                    <li key={a.id} className="rounded-xl bg-white/5 border border-white/10 p-3 flex items-center justify-between">
                      <div>
                        <div className="font-medium">
                          {a.type} — {a.vehicleTitle}
                          {a.plate ? ` (${a.plate})` : ""}
                        </div>
                        <div className="text-sm text-white/70">
                          Due: {a.date} •{" "}
                          {a.status === "expired" ? `${a.days} days overdue` : `in ${a.days} days`}
                        </div>
                      </div>
                      <span
                        className={
                          "px-2.5 py-1 rounded-lg text-xs " +
                          (a.status === "expired"
                            ? "bg-red-500/25 border border-red-500/40"
                            : "bg-amber-500/25 border border-amber-500/40")
                        }
                      >
                        {a.status === "expired" ? "Expired" : "Expiring"}
                      </span>
                    </li>
                  ))}
              </ul>
            )}
          </GlassCard>
        </div>

        {/* Expense Breakdown */}
        <GlassCard title="Expense Breakdown">
          {breakdown.length === 0 ? (
            <Empty>No expenses yet.</Empty>
          ) : (
            <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-3">
              {breakdown.map((b) => (
                <div
                  key={b.label}
                  className="rounded-xl bg-white/5 border border-white/10 p-4 text-center"
                >
                  <div className="text-sm text-white/70">{b.label}</div>
                  <div className="text-xl font-semibold mt-1">{fmtMY0.format(b.total)}</div>
                </div>
              ))}
            </div>
          )}
        </GlassCard>

        {/* Recent Activity */}
        <GlassCard title="Recent Activity" action={{ text: "Open Expenses", onClick: () => nav("/dashboard/expenses") }}>
          {recent.length === 0 ? (
            <Empty>No recent activity.</Empty>
          ) : (
            <ul className="divide-y divide-white/10">
              {recent.map((e) => (
                <li key={e.id} className="py-3 flex items-center justify-between">
                  <div>
                    <div className="font-medium">{e.title || e.category}</div>
                    <div className="text-white/60 text-sm">
                      {toISOyyyyMMdd(e.date || e.createdAt || new Date())}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="font-semibold">{fmtMY.format(e.amount || 0)}</div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </GlassCard>

        {/* Error / Loading states */}
        {loading && (
          <div className="rounded-2xl p-4 bg-white/10 border border-white/10 backdrop-blur-xl shadow-xl text-white/80">
            Loading dashboard…
          </div>
        )}
        {error && (
          <div className="rounded-2xl p-4 bg-red-900/30 border border-red-500/30 text-red-200">
            {error}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------------------- Small UI bits ---------------------------- */

function KPI({ icon, label, value, sub }: { icon: string; label: string; value: string; sub?: string }) {
  return (
    <div className="rounded-2xl p-4 bg-white/10 border border-white/15 backdrop-blur-xl shadow-xl">
      <div className="flex items-center gap-2 text-sm text-white/80">
        <span>{icon}</span>
        <span>{label}</span>
      </div>
      <div className="text-2xl font-semibold mt-1">{value}</div>
      {sub && <div className="text-xs text-white/55 mt-0.5">{sub}</div>}
    </div>
  );
}

function GlassCard({
  title,
  action,
  children,
}: {
  title: string;
  action?: { text: string; onClick: () => void };
  children: React.ReactNode;
}) {
  return (
    <div className="rounded-2xl p-4 bg-white/10 border border-white/15 backdrop-blur-xl shadow-xl">
      <div className="flex items-center justify-between mb-2">
        <div className="text-white/90 font-medium">{title}</div>
        {action && (
          <button
            onClick={action.onClick}
            className="text-sm px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20"
          >
            {action.text}
          </button>
        )}
      </div>
      {children}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="h-[120px] grid place-items-center rounded-xl bg-white/5 border border-white/10 text-white/60">
      {children}
    </div>
  );
}