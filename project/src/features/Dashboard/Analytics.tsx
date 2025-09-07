import * as React from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

/** ---------------- API helpers ---------------- */
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

/** ---------------- Types & helpers ---------------- */
type Expense = {
  id: string;
  title?: string | null;
  amount: number | string;
  category: "Fuel" | "Maintenance" | "Insurance" | "RoadTax" | "Toll_Parking" | "Other";
  date?: string | null;
  createdAt?: string | null;
};

const fmtMY = new Intl.NumberFormat("en-MY", { style: "currency", currency: "MYR" });
const fmtMY0 = new Intl.NumberFormat("en-MY", { style: "currency", currency: "MYR", maximumFractionDigits: 0 });

const monthLabel = (d: Date) =>
  `${String(d.getMonth() + 1).padStart(2, "0")}/${String(d.getFullYear()).slice(-2)}`;

const getDate = (e: Expense) => new Date(e.date || e.createdAt || "");
const N = (v: unknown) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
};
function catLabel(db: Expense["category"]): string {
  if (db === "Toll_Parking") return "Toll/Parking";
  if (db === "RoadTax") return "Road Tax";
  return db;
}

/** Blue→Purple palette for pie slices */
const BLUPURP = [
  "#60A5FA", // blue-400
  "#6EA8FE",
  "#7B9BFF",
  "#8A8AFD",
  "#977FFB",
  "#A678F8",
  "#B06FF7",
  "#A78BFA", // purple-400
];

/** ---------------- Component ---------------- */
export default function Analytics() {
  const [rows, setRows] = React.useState<Expense[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  React.useEffect(() => {
    (async () => {
      setLoading(true);
      setError(null);
      try {
        const e = await getJson<Expense[]>("/expenses");
        setRows(Array.isArray(e) ? e : []);
      } catch (err: any) {
        setError(err?.message || "Failed to load analytics");
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  /** --------- Totals & series (safe math) ---------- */
  const now = new Date();
  const last12Keys = Array.from({ length: 12 }, (_, i) => {
    const d = new Date(now);
    d.setMonth(now.getMonth() - (11 - i));
    return { key: `${d.getFullYear()}-${d.getMonth() + 1}`, label: monthLabel(d), total: 0 };
  });

  let totalSpend = 0;
  let twelveMonthSum = 0;

  for (const r of rows) {
    const amt = N(r.amount);
    totalSpend += amt;

    const d = getDate(r);
    if (!isNaN(+d)) {
      const start = new Date(now);
      start.setMonth(now.getMonth() - 11);
      start.setHours(0, 0, 0, 0);
      if (d >= start && d <= now) {
        const k = `${d.getFullYear()}-${d.getMonth() + 1}`;
        const bucket = last12Keys.find((x) => x.key === k);
        if (bucket) {
          bucket.total += amt;
          twelveMonthSum += amt;
        }
      }
    }
  }

  const avgMonthly = rows.length ? twelveMonthSum / 12 : 0;

  const monthlyData = last12Keys.map((m) => ({ name: m.label, total: Math.round(m.total) }));

  const byCategory = rows.reduce<Record<string, number>>((acc, r) => {
    const lbl = catLabel(r.category);
    acc[lbl] = (acc[lbl] || 0) + N(r.amount);
    return acc;
  }, {});
  const categoryData = Object.entries(byCategory).map(([name, value]) => ({ name, value }));

  /** ---------------- UI ---------------- */
  return (
    <div className="min-h-[calc(100vh-5rem)] px-3 sm:px-6 text-white">
      <div className="max-w-7xl mx-auto py-6 space-y-6">
        {/* Title */}
        <div className="rounded-2xl p-6 bg-white/10 border border-white/15 backdrop-blur-xl shadow-2xl">
          <div className="text-2xl sm:text-3xl font-semibold">Analytics & Insights</div>
          <div className="text-white/75 mt-1">Understand your vehicle spending patterns</div>
        </div>

        {/* KPIs */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <Card title="Total Spend" sub="All time" value={fmtMY0.format(totalSpend)} />
          <Card title="Average / Month" sub="Last 12 months" value={fmtMY0.format(Math.max(0, Math.round(avgMonthly)))} />
          <Card title="Records" sub="Expense entries" value={String(rows.length)} />
        </div>

        {/* Charts */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <GlassBlock title="Monthly Spend (Last 12 Months)">
            <div className="h-64">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={monthlyData}>
                  {/* Gradient defs for bar fill */}
                  <defs>
                    <linearGradient id="barBluePurple" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#60A5FA" />   {/* blue */}
                      <stop offset="100%" stopColor="#A78BFA" /> {/* purple */}
                    </linearGradient>
                  </defs>
                  <XAxis dataKey="name" stroke="rgba(255,255,255,0.55)" />
                  <YAxis stroke="rgba(255,255,255,0.55)" />
                  <Tooltip
                    contentStyle={{ background: "rgba(20,20,30,.9)", border: "1px solid rgba(255,255,255,.15)", color: "#fff" }}
                  />
                  <Bar dataKey="total" fill="url(#barBluePurple)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </GlassBlock>

          <GlassBlock title="Share by Category">
            {categoryData.length === 0 ? (
              <Empty>No data</Empty>
            ) : (
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie dataKey="value" data={categoryData} innerRadius={60} outerRadius={90} label>
                      {categoryData.map((_, i) => (
                        <Cell key={i} fill={BLUPURP[i % BLUPURP.length]} />
                      ))}
                    </Pie>
                    <Legend
                      wrapperStyle={{ color: "rgba(255,255,255,.85)" }}
                      iconType="square"
                    />
                    <Tooltip
                      contentStyle={{ background: "rgba(20,20,30,.9)", border: "1px solid rgba(255,255,255,.15)", color: "#fff" }}
                    />
                  </PieChart>
                </ResponsiveContainer>
              </div>
            )}
          </GlassBlock>
        </div>

        {/* Loading / Error */}
        {loading && (
          <div className="rounded-2xl p-4 bg-white/10 border border-white/10 backdrop-blur-xl shadow-xl text-white/80">
            Loading analytics…
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

/** --------- UI bits (glass) --------- */
function Card({ title, sub, value }: { title: string; sub?: string; value: string }) {
  return (
    <div className="rounded-2xl p-5 bg-white/10 border border-white/15 backdrop-blur-xl shadow-xl">
      <div className="text-white/80">{title}</div>
      <div className="text-3xl font-semibold mt-1">{value}</div>
      {sub && <div className="text-xs text-white/60 mt-1">{sub}</div>}
    </div>
  );
}

function GlassBlock({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="rounded-2xl p-4 bg-white/10 border border-white/15 backdrop-blur-xl shadow-xl">
      <div className="text-white/90 font-medium mb-2">{title}</div>
      {children}
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="h-[200px] grid place-items-center rounded-xl bg-white/5 border border-white/10 text-white/60">
      {children}
    </div>
  );
}