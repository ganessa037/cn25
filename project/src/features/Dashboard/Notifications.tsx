import React, { useEffect, useMemo, useState } from "react";
import {
  AlertTriangle, Clock, CheckCircle2, BadgeAlert, Wrench, Droplet,
  CalendarDays, ChevronRight
} from "lucide-react";
import { GlassButton, GlassCard, GlassPanel } from "../../components/ui/Glass";
import type { Vehicle } from "../../pages/Dashboard";

// ========== Types ==========
type Priority = "high" | "medium" | "low";

type AlertKind = "insurance_expired" | "roadtax_expired" | "service_overdue" | "fuel_price";

type DashAlert = {
  id: string;              // stable id
  kind: AlertKind;
  priority: Priority;
  title: string;
  message: string;
  vehicleId?: string;
  date?: string;           // for display
  category?: string;       // small label at the right (e.g., Alert/Maintenance/Info)
  actionRequired?: boolean;
};

// Optional: parent可传入导航回调；若未传则 console.log 代替
export interface NotificationsProps {
  vehicles: Vehicle[];
  onNavigate?: (section: "Vehicle Manager" | "Expense Tracker" | "Document Manager" | "Analytics") => void;
}

// ========== Utils ==========
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

const STORAGE_KEY = "keretaku:dismissedAlerts";
const loadDismissed = (): Record<string, true> => {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    return raw ? JSON.parse(raw) : {};
  } catch { return {}; }
};
const saveDismissed = (map: Record<string, true>) => {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(map)); } catch {}
};

// ========== Component ==========
export default function Notifications({ vehicles, onNavigate }: NotificationsProps) {
  const [dismissed, setDismissed] = useState<Record<string, true>>(() => loadDismissed());

  useEffect(() => { saveDismissed(dismissed); }, [dismissed]);

  // 基于车辆生成提醒（稳定 id）
  const generated = useMemo<DashAlert[]>(() => {
    const out: DashAlert[] = [];
    const todayISO = new Date().toISOString().slice(0, 10);

    vehicles.forEach((v: any) => {
      // Insurance
      if (v.insuranceExpiry) {
        const delta = daysDiffFromToday(v.insuranceExpiry);
        if (delta < 0) {
          out.push({
            id: `veh:${v.id}:insurance_expired`,
            kind: "insurance_expired",
            priority: "high",
            title: "Insurance Expired",
            message: `Insurance for ${vehicleLabel(v)} has expired ${Math.abs(delta)} days ago.`,
            vehicleId: v.id,
            date: v.insuranceExpiry,
            category: "Alert",
            actionRequired: true,
          });
        } else if (delta <= 30) {
          out.push({
            id: `veh:${v.id}:insurance_expiring`,
            kind: "insurance_expired",
            priority: "medium",
            title: "Insurance Expiring Soon",
            message: `Insurance for ${vehicleLabel(v)} will expire in ${delta} days.`,
            vehicleId: v.id,
            date: v.insuranceExpiry,
            category: "Alert",
            actionRequired: true,
          });
        }
      }

      // Road Tax
      if (v.roadTaxExpiry) {
        const delta = daysDiffFromToday(v.roadTaxExpiry);
        if (delta < 0) {
          out.push({
            id: `veh:${v.id}:roadtax_expired`,
            kind: "roadtax_expired",
            priority: "high",
            title: "Road Tax Expired",
            message: `Road tax for ${vehicleLabel(v)} has expired ${Math.abs(delta)} days ago.`,
            vehicleId: v.id,
            date: v.roadTaxExpiry,
            category: "Alert",
            actionRequired: true,
          });
        } else if (delta <= 30) {
          out.push({
            id: `veh:${v.id}:roadtax_expiring`,
            kind: "roadtax_expired",
            priority: "medium",
            title: "Road Tax Expiring Soon",
            message: `Road tax for ${vehicleLabel(v)} will expire in ${delta} days.`,
            vehicleId: v.id,
            date: v.roadTaxExpiry,
            category: "Alert",
            actionRequired: true,
          });
        }
      }

      // Next service overdue
      if (v.nextServiceDate) {
        const delta = daysDiffFromToday(v.nextServiceDate);
        if (delta < 0) {
          out.push({
            id: `veh:${v.id}:service_overdue`,
            kind: "service_overdue",
            priority: "medium",
            title: "Service Overdue",
            message: `Service for ${vehicleLabel(v)} is overdue by ${Math.abs(delta)} days.`,
            vehicleId: v.id,
            date: v.nextServiceDate,
            category: "Maintenance",
            actionRequired: true,
          });
        }
      }
    });

    // 示例：Fuel price alert（低优先级信息）
    out.push({
      id: `global:fuel_price_${new Date().toISOString().slice(0,10)}`,
      kind: "fuel_price",
      priority: "low",
      title: "Fuel Price Alert",
      message: "Petrol prices may increase next week. Consider refueling this weekend.",
      date: todayISO,
      category: "Info",
      actionRequired: false,
    });

    return out;
  }, [vehicles]);

  // 过滤已隐藏
  const alerts = useMemo(() => generated.filter(a => !dismissed[a.id]), [generated, dismissed]);

  const kpi = useMemo(() => {
    const res = { high: 0, medium: 0, low: 0, total: alerts.length, actions: 0 };
    alerts.forEach(a => {
      if (a.priority === "high") res.high += 1;
      if (a.priority === "medium") res.medium += 1;
      if (a.priority === "low") res.low += 1;
      if (a.actionRequired) res.actions += 1;
    });
    return res;
  }, [alerts]);

  const dismiss = (id: string) => {
    setDismissed(prev => ({ ...prev, [id]: true }));
  };

  const takeAction = (a: DashAlert) => {
    // 默认跳转到 Vehicle Manager；父层若传 onNavigate，则调用之
    onNavigate?.("Vehicle Manager");
    // 也可以在这里做更多：预选中车辆/打开编辑等（留给父层根据回调处理）
  };

  // 样式工具
  const prBadge = (p: Priority) => {
    const base = "px-2 py-0.5 rounded-md text-xxs font-medium border inline-flex items-center gap-1";
    if (p === "high") return <span className={`${base} bg-red-500/15 text-red-300 border-red-400/40`}><AlertTriangle className="w-3.5 h-3.5"/> HIGH</span>;
    if (p === "medium") return <span className={`${base} bg-amber-500/15 text-amber-300 border-amber-400/40`}><Clock className="w-3.5 h-3.5"/> MEDIUM</span>;
    return <span className={`${base} bg-sky-500/15 text-sky-300 border-sky-400/40`}><BadgeAlert className="w-3.5 h-3.5"/> LOW</span>;
  };

  const kindIcon = (a: DashAlert) => {
    const wrap = "w-9 h-9 rounded-xl flex items-center justify-center mr-2";
    switch (a.kind) {
      case "insurance_expired":
        return <div className={`${wrap} bg-rose-400/15`}><AlertTriangle className="w-5 h-5 text-rose-300"/></div>;
      case "roadtax_expired":
        return <div className={`${wrap} bg-orange-400/15`}><AlertTriangle className="w-5 h-5 text-orange-300"/></div>;
      case "service_overdue":
        return <div className={`${wrap} bg-amber-400/15`}><Wrench className="w-5 h-5 text-amber-300"/></div>;
      default:
        return <div className={`${wrap} bg-emerald-400/15`}><Droplet className="w-5 h-5 text-emerald-300"/></div>;
    }
  };

  return (
    <section className="space-y-6">
      {/* Header + actions required */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Smart Notifications</h3>
          <p className="text-white/60">Stay on top of your vehicle maintenance and renewals</p>
        </div>
        <div className="text-white/60 text-sm">{kpi.actions} actions required</div>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <GlassCard className="glass-hover flex items-center justify-between">
          <div>
            <div className="text-white/70 text-sm">High Priority</div>
            <div className="text-2xl font-semibold mt-1">{kpi.high}</div>
          </div>
          <AlertTriangle className="w-5 h-5 text-red-300" />
        </GlassCard>
        <GlassCard className="glass-hover flex items-center justify-between">
          <div>
            <div className="text-white/70 text-sm">Medium Priority</div>
            <div className="text-2xl font-semibold mt-1">{kpi.medium}</div>
          </div>
          <Clock className="w-5 h-5 text-amber-300" />
        </GlassCard>
        <GlassCard className="glass-hover flex items-center justify-between">
          <div>
            <div className="text-white/70 text-sm">Low Priority</div>
            <div className="text-2xl font-semibold mt-1">{kpi.low}</div>
          </div>
          <BadgeAlert className="w-5 h-5 text-sky-300" />
        </GlassCard>
        <GlassCard className="glass-hover flex items-center justify-between">
          <div>
            <div className="text-white/70 text-sm">Total Alerts</div>
            <div className="text-2xl font-semibold mt-1">{kpi.total}</div>
          </div>
          <CheckCircle2 className="w-5 h-5 text-white/70" />
        </GlassCard>
      </div>

      {/* Alerts list */}
      <div className="space-y-3">
        {alerts.map((a) => {
          const v = vehicles.find((x: any) => x.id === a.vehicleId);
          const subtitle = [
            v ? vehicleLabel(v) : undefined,
            a.date ? fmtDate(a.date) : undefined,
            a.category || undefined,
          ].filter(Boolean).join("  •  ");

          // 背景强调色
          const tone =
            a.priority === "high"
              ? "ring-1 ring-red-400/30 bg-red-400/[0.07]"
              : a.priority === "medium"
              ? "ring-1 ring-amber-400/25 bg-amber-400/[0.06]"
              : "ring-1 ring-sky-400/20 bg-sky-400/[0.05]";

        return (
          <div key={a.id} className={`glass-card p-4 ${tone}`}>
            <div className="flex items-start justify-between gap-4">
              {/* left */}
              <div className="flex items-start gap-3 min-w-0">
                {kindIcon(a)}
                <div className="min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <div className="font-semibold">{a.title}</div>
                    {prBadge(a.priority)}
                    {a.actionRequired && (
                      <span className="px-2 py-0.5 rounded-md text-xxs bg-amber-300/20 text-amber-200 border border-amber-300/30">
                        Action Required
                      </span>
                    )}
                  </div>
                  <div className="text-white/70 mt-1">{a.message}</div>
                  {subtitle && (
                    <div className="text-white/50 text-sm mt-1 flex items-center gap-2">
                      <CalendarDays className="w-3.5 h-3.5" />
                      <span>{subtitle}</span>
                    </div>
                  )}
                </div>
              </div>

              {/* right actions */}
              <div className="flex items-center gap-3 shrink-0">
                {a.actionRequired && (
                  <GlassButton onClick={() => takeAction(a)} className="px-3 py-1.5">
                    Take Action <ChevronRight className="w-4 h-4" />
                  </GlassButton>
                )}
                <button className="text-white/60 hover:text-white text-sm" onClick={() => dismiss(a.id)}>
                  Dismiss
                </button>
              </div>
            </div>
          </div>
        );})}
        {alerts.length === 0 && (
          <GlassPanel className="text-white/70 text-center py-10">
            All caught up — no alerts right now 🎉
          </GlassPanel>
        )}
      </div>

      {/* Smart Predictions */}
      <div className="rounded-2xl border border-sky-300/20 bg-sky-400/10 p-4">
        <div className="flex items-center gap-2 mb-3">
          <BadgeAlert className="w-5 h-5 text-sky-300" />
          <div className="font-semibold">Smart Predictions</div>
        </div>

        <div className="space-y-3">
          <div className="glass-card bg-white/5">
            <div className="flex items-center gap-2 mb-1">
              <Droplet className="w-4 h-4 text-emerald-300" />
              <span className="font-medium">Fuel Efficiency Insight</span>
            </div>
            <div className="text-white/70">
              Based on your driving patterns, you could save RM50/month by optimizing routes and consolidating trips.
            </div>
          </div>

          <div className="glass-card bg-white/5">
            <div className="flex items-center gap-2 mb-1">
              <Wrench className="w-4 h-4 text-amber-300" />
              <span className="font-medium">Maintenance Prediction</span>
            </div>
            <div className="text-white/70">
              Your vehicle will likely need brake pad replacement in 3–4 months based on current mileage trends.
            </div>
          </div>

          <div className="glass-card bg-white/5">
            <div className="flex items-center gap-2 mb-1">
              <CheckCircle2 className="w-4 h-4 text-white/80" />
              <span className="font-medium">Budget Forecast</span>
            </div>
            <div className="text-white/70">
              Expected vehicle expenses for next month: RM400–500 (includes fuel, maintenance, and insurance).
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}