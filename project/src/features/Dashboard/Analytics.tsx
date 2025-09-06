import React, { useMemo } from "react";
import {
  Bell, Car, CalendarDays, CircleDollarSign, Gauge, PieChart,
  AlertTriangle, Droplet, Wrench, Shield
} from "lucide-react";
import { GlassCard, GlassPanel } from "../../components/ui/Glass";
import type { Vehicle } from "../../pages/Dashboard";
import type { Expense, ExpenseCategory } from "./ExpenseTracker";

/** ===== Helpers ===== */
const rm = (n: number) =>
  `RM${(n || 0).toLocaleString("en-MY", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const fmtDate = (s?: string) => {
  if (!s) return "—";
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString("en-US");
};

const daysDiffFromToday = (dateISO: string) => {
  const today = new Date(); today.setHours(0,0,0,0);
  const d = new Date(dateISO); d.setHours(0,0,0,0);
  const ms = d.getTime() - today.getTime();
  return Math.round(ms / (24 * 3600 * 1000)); // future positive, past negative
};

const vehicleLabel = (v?: any) => {
  if (!v) return undefined;
  const title = [v.brand, v.model].filter(Boolean).join(" ");
  const plate = v.plate ? `(${v.plate})` : "";
  return title ? `${title} ${plate}`.trim() : v.name || v.plate || v.id;
};

const catIcon = (c: ExpenseCategory) => {
  const base = "w-10 h-10 rounded-full flex items-center justify-center";
  switch (c) {
    case "Fuel":
      return <div className={`${base} bg-emerald-400/20`}><Droplet className="w-5 h-5 text-emerald-300" /></div>;
    case "Insurance":
      return <div className={`${base} bg-rose-400/20`}><Shield className="w-5 h-5 text-rose-300" /></div>;
    case "Maintenance":
      return <div className={`${base} bg-indigo-400/20`}><Wrench className="w-5 h-5 text-indigo-300" /></div>;
    case "Road Tax":
      return <div className={`${base} bg-orange-400/20`}><CircleDollarSign className="w-5 h-5 text-orange-300" /></div>;
    case "Toll/Parking":
      return <div className={`${base} bg-teal-400/20`}><CircleDollarSign className="w-5 h-5 text-teal-300" /></div>;
    default:
      return <div className={`${base} bg-slate-400/20`}><PieChart className="w-5 h-5 text-slate-300" /></div>;
  }
};

/** ===== Props ===== */
export interface AnalyticsProps {
  vehicles: Vehicle[];
  expenses: Expense[];
  // 可选：点击“即将到期”或别处引导切页
  onNavigate?: (section: "Vehicle Manager" | "Expense Tracker" | "Notifications" | "Document Manager") => void;
}

export default function Analytics({ vehicles, expenses }: AnalyticsProps) {
  /** 欢迎语（尝试从本地存储取名） */
  let displayName = "back";
  try {
    const stored = localStorage.getItem("user");
    if (stored) {
      const u = JSON.parse(stored);
      if (u?.name) displayName = u.name;
    }
  } catch {}

  /** KPI：总车辆、当月支出、有效提醒数、月均支出 */
  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth();

  const { totalVehicles, thisMonth, avgPerMonth, alertsCount } = useMemo(() => {
    const totalVehicles = vehicles.length;

    const total = expenses.reduce((s, e) => s + (e.amount || 0), 0);
    const thisMonth = expenses.reduce((s, e) => {
      const d = new Date(e.date);
      return d.getFullYear() === year && d.getMonth() === month ? s + (e.amount || 0) : s;
    }, 0);

    // 计算提醒（与 Notifications 逻辑一致的“需要动作”的数量）
    let alertsCount = 0;
    vehicles.forEach((v: any) => {
      const check = (iso?: string) => {
        if (!iso) return;
        const dd = daysDiffFromToday(iso);
        if (dd < 0 || dd <= 30) alertsCount += 1;
      };
      check(v.insuranceExpiry);
      check(v.roadTaxExpiry);
      // 服务逾期才算动作
      if (v.nextServiceDate && daysDiffFromToday(v.nextServiceDate) < 0) alertsCount += 1;
    });

    if (expenses.length === 0) return { totalVehicles, thisMonth, avgPerMonth: 0, alertsCount };

    const ts = expenses.map((e) => new Date(e.date).getTime()).filter((n) => !Number.isNaN(n));
    const minD = new Date(Math.min(...ts));
    const maxD = new Date(Math.max(...ts));
    const months =
      (maxD.getFullYear() - minD.getFullYear()) * 12 + (maxD.getMonth() - minD.getMonth()) + 1;
    const avgPerMonth = months > 0 ? total / months : total;

    return { totalVehicles, thisMonth, avgPerMonth, alertsCount };
  }, [vehicles, expenses, month, year]);

  /** 你的车辆（显示所有） */
  const vehicleCards = useMemo(() => vehicles, [vehicles]);

  /** 即将到期（未来 45 天内） */
  type UpcomingRow = { id: string; title: string; vehicleId: string; date: string; days: number };
  const upcoming: UpcomingRow[] = useMemo(() => {
    const rows: UpcomingRow[] = [];
    vehicles.forEach((v: any) => {
      const pushIfSoon = (label: string, iso?: string) => {
        if (!iso) return;
        const d = daysDiffFromToday(iso);
        if (d >= 0 && d <= 45) {
          rows.push({ id: `${v.id}:${label}`, title: label, vehicleId: v.id, date: iso, days: d });
        }
      };
      pushIfSoon("Insurance Renewal", v.insuranceExpiry);
      pushIfSoon("Road Tax Renewal", v.roadTaxExpiry);
      pushIfSoon("Service Due", v.nextServiceDate);
    });
    return rows.sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }, [vehicles]);

  /** 支出分类（显示四大类） */
  const byCat = useMemo(() => {
    const base: Record<ExpenseCategory | "Other", number> = {
      Fuel: 0,
      Insurance: 0,
      Maintenance: 0,
      "Road Tax": 0,
      "Toll/Parking": 0,
      Other: 0,
    };
    expenses.forEach((e) => {
      const k = (["Fuel","Insurance","Maintenance","Road Tax","Toll/Parking"].includes(e.category)
        ? (e.category as ExpenseCategory)
        : "Other") as ExpenseCategory | "Other";
      base[k] += e.amount || 0;
    });
    return base;
  }, [expenses]);

  /** 最近活动（按时间倒序取前 5 条） */
  const recent = useMemo(() => {
    const list = [...expenses].sort(
      (a, b) => new Date(b.date).getTime() - new Date(a.date).getTime()
    );
    return list.slice(0, 5);
  }, [expenses]);

  return (
    <section className="space-y-6">
      {/* 顶部欢迎条 */}
      <div className="rounded-2xl border border-white/10 bg-gradient-to-r from-indigo-600/70 to-blue-600/70 text-white p-5 shadow-lg">
        <div className="text-lg font-semibold">Welcome back, {displayName}!</div>
        <div className="text-white/90">Here’s your vehicle management overview</div>
      </div>

      {/* KPI */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <GlassCard className="glass-hover">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white/70 text-sm">Total Vehicles</div>
              <div className="text-2xl font-semibold mt-1">{totalVehicles}</div>
            </div>
            <Car className="w-5 h-5 text-white/70" />
          </div>
        </GlassCard>

        <GlassCard className="glass-hover">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white/70 text-sm">This Month</div>
              <div className="text-2xl font-semibold mt-1">{rm(thisMonth)}</div>
            </div>
            <CalendarDays className="w-5 h-5 text-emerald-300" />
          </div>
        </GlassCard>

        <GlassCard className="glass-hover">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white/70 text-sm">Alerts</div>
              <div className="text-2xl font-semibold mt-1">{alertsCount}</div>
            </div>
            <AlertTriangle className="w-5 h-5 text-amber-300" />
          </div>
        </GlassCard>

        <GlassCard className="glass-hover">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-white/70 text-sm">Avg/Month</div>
              <div className="text-2xl font-semibold mt-1">{rm(avgPerMonth)}</div>
            </div>
            <Gauge className="w-5 h-5 text-white/70" />
          </div>
        </GlassCard>
      </div>

      {/* 车辆 & 即将到期 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Your Vehicles */}
        <GlassPanel className="p-0">
          <div className="px-4 pt-4 pb-2 font-semibold">Your Vehicles</div>
          <div className="px-4 pb-4 space-y-3">
            {vehicleCards.length === 0 ? (
              <div className="text-white/60 py-8 text-center">No vehicles added yet</div>
            ) : (
              vehicleCards.map((v: any) => (
                <div key={v.id} className="glass-card p-3 flex items-center justify-between">
                  <div>
                    <div className="font-medium">{vehicleLabel(v)}</div>
                    <div className="text-white/60 text-sm">{v.plate || "—"}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-white/60 text-xs">Current Mileage</div>
                    <div className="font-semibold">
                      {v.currentMileage ? v.currentMileage.toLocaleString() : 0}
                      <span className="text-white/60 text-sm"> km</span>
                    </div>
                  </div>
                </div>
              ))
            )}
          </div>
        </GlassPanel>

        {/* Upcoming Renewals */}
        <GlassPanel className="p-0">
          <div className="px-4 pt-4 pb-2 font-semibold">Upcoming Renewals</div>
          <div className="px-4 pb-4">
            {upcoming.length === 0 ? (
              <div className="glass-card flex items-center justify-center py-10 text-white/60">
                <div className="flex items-center gap-2">
                  <CalendarDays className="w-5 h-5" />
                  <span>No upcoming renewals</span>
                </div>
              </div>
            ) : (
              <div className="space-y-3">
                {upcoming.map((u) => {
                  const v = vehicles.find((x: any) => x.id === u.vehicleId);
                  return (
                    <div key={u.id} className="glass-card p-3 flex items-center justify-between">
                      <div>
                        <div className="font-medium">{u.title}</div>
                        <div className="text-white/60 text-sm">
                          {vehicleLabel(v)} • {fmtDate(u.date)}
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-white/60 text-xs">Due in</div>
                        <div className="font-semibold">{u.days} days</div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </GlassPanel>
      </div>

      {/* Expense Breakdown */}
      <GlassPanel className="p-0">
        <div className="px-4 pt-4 pb-2 font-semibold">Expense Breakdown</div>
        <div className="px-4 pb-4 grid grid-cols-2 sm:grid-cols-4 gap-4">
          {/* Fuel */}
          <div className="glass-card p-4 flex flex-col items-center gap-2">
            <div className="w-14 h-14 rounded-full bg-emerald-400/20 flex items-center justify-center">
              <Droplet className="w-6 h-6 text-emerald-300" />
            </div>
            <div className="font-semibold">{rm(byCat["Fuel"])}</div>
            <div className="text-white/60 text-sm">Fuel</div>
          </div>
          {/* Insurance */}
          <div className="glass-card p-4 flex flex-col items-center gap-2">
            <div className="w-14 h-14 rounded-full bg-rose-400/20 flex items-center justify-center">
              <Shield className="w-6 h-6 text-rose-300" />
            </div>
            <div className="font-semibold">{rm(byCat["Insurance"])}</div>
            <div className="text-white/60 text-sm">Insurance</div>
          </div>
          {/* Maintenance */}
          <div className="glass-card p-4 flex flex-col items-center gap-2">
            <div className="w-14 h-14 rounded-full bg-indigo-400/20 flex items-center justify-center">
              <Wrench className="w-6 h-6 text-indigo-300" />
            </div>
            <div className="font-semibold">{rm(byCat["Maintenance"])}</div>
            <div className="text-white/60 text-sm">Maintenance</div>
          </div>
          {/* Road Tax */}
          <div className="glass-card p-4 flex flex-col items-center gap-2">
            <div className="w-14 h-14 rounded-full bg-orange-400/20 flex items-center justify-center">
              <CircleDollarSign className="w-6 h-6 text-orange-300" />
            </div>
            <div className="font-semibold">{rm(byCat["Road Tax"])}</div>
            <div className="text-white/60 text-sm">Road Tax</div>
          </div>
        </div>
      </GlassPanel>

      {/* Recent Activity */}
      <GlassPanel className="p-0">
        <div className="px-4 pt-4 pb-2 font-semibold">Recent Activity</div>
        <div className="px-4 pb-4">
          {recent.length === 0 ? (
            <div className="text-white/60 py-8 text-center">No activity yet</div>
          ) : (
            <div className="space-y-3">
              {recent.map((e) => (
                <div key={e.id} className="glass-card p-3 flex items-center justify-between">
                  <div className="flex items-center gap-3 min-w-0">
                    {catIcon(e.category)}
                    <div className="min-w-0">
                      <div className="font-medium truncate">{e.title}</div>
                      <div className="text-white/60 text-sm">{e.category}</div>
                    </div>
                  </div>
                  <div className="text-right shrink-0">
                    <div className="font-semibold">{rm(e.amount)}</div>
                    <div className="text-white/60 text-sm">{fmtDate(e.date)}</div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </GlassPanel>
    </section>
  );
}