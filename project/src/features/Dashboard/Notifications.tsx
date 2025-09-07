import * as React from "react";

/**
 * Notifications
 * - Data source: Vehicle table via /api/vehicles (OAuth Bearer token).
 * - Computes: Insurance & Road Tax expiry, Service overdue/due soon, plus a Low-priority fuel tip.
 * - "Take Action" opens modals that PUT updates to /api/vehicles/:id.
 * - "Dismiss" hides an alert for 7 days (localStorage only, no DB row required).
 */

/* ----------------------------- Types ----------------------------- */

type Vehicle = {
  id: string;
  brand: string;
  model?: string | null;
  year?: number | null;
  plate?: string | null;
  color?: string | null;
  fuelType?: string | null;

  roadTaxExpiry?: string | null;
  insuranceExpiry?: string | null;
  lastServiceDate?: string | null;
  nextServiceDate?: string | null;
  currentMileage?: number | null;
};

type Priority = "high" | "medium" | "low";
type AlertKind =
  | "insurance_expired"
  | "insurance_expiring"
  | "roadtax_expired"
  | "roadtax_expiring"
  | "service_overdue"
  | "service_due_soon"
  | "fuel_price";

type Alert = {
  id: string;
  kind: AlertKind;
  priority: Priority;
  title: string;
  message: string;
  context: string;
  dateLabel?: string;
  vehicleId?: string;
  action?: { label: string; open: () => void };
};

/* -------------------------- API Utilities ------------------------ */

function resolveApiBase(): string {
  const raw =
    (import.meta as any).env?.VITE_API_URL ||
    (import.meta as any).env?.VITE_BACKEND_URL ||
    "http://127.0.0.1:3000";
  const trimmed = String(raw).replace(/\/$/, "");
  return trimmed.endsWith("/api") ? trimmed : `${trimmed}/api`;
}
const API_BASE = resolveApiBase();

function getToken(): string | null {
  try {
    const raw = localStorage.getItem("user");
    if (!raw) return null;
    return JSON.parse(raw)?.token ?? null;
  } catch {
    return null;
  }
}

async function api<T>(
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
  body?: unknown
): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = { "Content-Type": "application/json" };
  if (token) headers["Authorization"] = `Bearer ${token}`;
  const res = await fetch(`${API_BASE}${path}`, {
    method,
    headers,
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${text || "Request failed"}`);
  }
  return (res.status === 204 ? (null as unknown as T) : ((await res.json()) as T));
}

/* ----------------------------- Helpers --------------------------- */

const toISODate = (s?: string | null) => {
  if (!s) return "";
  const d = new Date(s);
  if (isNaN(+d)) return "";
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
};
const fmtDate = (s?: string | null) => (s ? toISODate(s) : "—");

function toDbTimestamp(localDate: string | ""): string | null {
  if (!localDate) return null;
  const [y, m, d] = localDate.split("-").map(Number);
  if (!y || !m || !d) return null;
  const yyyy = y;
  const mm = String(m).padStart(2, "0");
  const dd = String(d).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}T00:00:00.000`;
}

function daysFromToday(dateIso?: string | null): number | null {
  if (!dateIso) return null;
  const d = new Date(dateIso);
  if (isNaN(+d)) return null;
  const today = new Date();
  const a = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const b = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  return Math.round((a - b) / (24 * 60 * 60 * 1000));
}

function labelForVehicle(v: Vehicle): string {
  const base = `${v.brand ?? "Vehicle"}${v.model ? " " + v.model : ""}`.trim();
  return v.plate ? `${base} (${v.plate})` : base;
}

/* -------------------------- Local Dismiss ------------------------- */

type Dismissed = { id: string; until: number }; // epoch ms

const DISMISS_KEY = "notifications_dismissed";

function loadDismissed(): Dismissed[] {
  try {
    const raw = localStorage.getItem(DISMISS_KEY);
    const arr = raw ? (JSON.parse(raw) as Dismissed[]) : [];
    const now = Date.now();
    return arr.filter((x) => x.until > now);
  } catch {
    return [];
  }
}
function saveDismissed(list: Dismissed[]) {
  localStorage.setItem(DISMISS_KEY, JSON.stringify(list));
}
function isDismissed(id: string): boolean {
  return loadDismissed().some((x) => x.id === id && x.until > Date.now());
}
function dismissFor7Days(id: string) {
  const list = loadDismissed().filter((x) => x.id !== id);
  list.push({ id, until: Date.now() + 7 * 24 * 60 * 60 * 1000 });
  saveDismissed(list);
}

/* ------------------------- Main Component ------------------------- */

export default function Notifications() {
  const [vehicles, setVehicles] = React.useState<Vehicle[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Modals
  const [editVehicle, setEditVehicle] = React.useState<Vehicle | null>(null);
  const [editMode, setEditMode] = React.useState<
    | null
    | { type: "insurance"; date: string }
    | { type: "roadtax"; date: string }
    | { type: "service"; lastDate: string; nextDate: string; mileage: string }
  >(null);

  const refetch = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const v = await api<Vehicle[]>("GET", "/vehicles");
      setVehicles(v || []);
    } catch (err: any) {
      setError(err?.message || "Failed to load notifications");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    refetch();
  }, [refetch]);

  /* ------------------------- Build Alerts -------------------------- */

  const alerts = React.useMemo<Alert[]>(() => {
    const list: Alert[] = [];

    vehicles.forEach((v) => {
      // Insurance
      const insDays = daysFromToday(v.insuranceExpiry);
      if (insDays !== null) {
        if (insDays < 0) {
          const id = `ins-exp-${v.id}-${toISODate(v.insuranceExpiry)}`;
          if (!isDismissed(id))
            list.push({
              id,
              kind: "insurance_expired",
              priority: "high",
              title: "Insurance Expired",
              message: `Insurance for ${labelForVehicle(v)} has expired ${Math.abs(insDays)} days ago.`,
              context: `${labelForVehicle(v)} • ${fmtDate(v.insuranceExpiry)}`,
              dateLabel: fmtDate(v.insuranceExpiry),
              vehicleId: v.id,
              action: {
                label: "Take Action",
                open: () => {
                  setEditVehicle(v);
                  setEditMode({ type: "insurance", date: toISODate(v.insuranceExpiry) });
                },
              },
            });
        } else if (insDays <= 30) {
          const id = `ins-soon-${v.id}-${toISODate(v.insuranceExpiry)}`;
          if (!isDismissed(id))
            list.push({
              id,
              kind: "insurance_expiring",
              priority: "medium",
              title: "Insurance Expiring Soon",
              message: `Insurance for ${labelForVehicle(v)} expires in ${insDays} days.`,
              context: `${labelForVehicle(v)} • ${fmtDate(v.insuranceExpiry)}`,
              dateLabel: fmtDate(v.insuranceExpiry),
              vehicleId: v.id,
              action: {
                label: "Take Action",
                open: () => {
                  setEditVehicle(v);
                  setEditMode({ type: "insurance", date: toISODate(v.insuranceExpiry) });
                },
              },
            });
        }
      }

      // Road Tax
      const taxDays = daysFromToday(v.roadTaxExpiry);
      if (taxDays !== null) {
        if (taxDays < 0) {
          const id = `tax-exp-${v.id}-${toISODate(v.roadTaxExpiry)}`;
          if (!isDismissed(id))
            list.push({
              id,
              kind: "roadtax_expired",
              priority: "high",
              title: "Road Tax Expired",
              message: `Road tax for ${labelForVehicle(v)} has expired ${Math.abs(taxDays)} days ago.`,
              context: `${labelForVehicle(v)} • ${fmtDate(v.roadTaxExpiry)}`,
              dateLabel: fmtDate(v.roadTaxExpiry),
              vehicleId: v.id,
              action: {
                label: "Take Action",
                open: () => {
                  setEditVehicle(v);
                  setEditMode({ type: "roadtax", date: toISODate(v.roadTaxExpiry) });
                },
              },
            });
        } else if (taxDays <= 30) {
          const id = `tax-soon-${v.id}-${toISODate(v.roadTaxExpiry)}`;
          if (!isDismissed(id))
            list.push({
              id,
              kind: "roadtax_expiring",
              priority: "medium",
              title: "Road Tax Expiring Soon",
              message: `Road tax for ${labelForVehicle(v)} expires in ${taxDays} days.`,
              context: `${labelForVehicle(v)} • ${fmtDate(v.roadTaxExpiry)}`,
              dateLabel: fmtDate(v.roadTaxExpiry),
              vehicleId: v.id,
              action: {
                label: "Take Action",
                open: () => {
                  setEditVehicle(v);
                  setEditMode({ type: "roadtax", date: toISODate(v.roadTaxExpiry) });
                },
              },
            });
        }
      }

      // Service overdue / due soon
      const svcDays = daysFromToday(v.nextServiceDate);
      if (svcDays !== null) {
        if (svcDays < 0) {
          const id = `svc-ovd-${v.id}-${toISODate(v.nextServiceDate)}`;
          if (!isDismissed(id))
            list.push({
              id,
              kind: "service_overdue",
              priority: "medium",
              title: "Service Overdue",
              message: `Service for ${labelForVehicle(v)} is overdue by ${Math.abs(svcDays)} days.`,
              context: `${labelForVehicle(v)} • ${fmtDate(v.nextServiceDate)} • Maintenance`,
              dateLabel: fmtDate(v.nextServiceDate),
              vehicleId: v.id,
              action: {
                label: "Take Action",
                open: () => {
                  setEditVehicle(v);
                  setEditMode({
                    type: "service",
                    lastDate: toISODate(v.lastServiceDate),
                    nextDate: toISODate(v.nextServiceDate),
                    mileage: String(v.currentMileage ?? ""),
                  });
                },
              },
            });
        } else if (svcDays <= 30) {
          const id = `svc-soon-${v.id}-${toISODate(v.nextServiceDate)}`;
          if (!isDismissed(id))
            list.push({
              id,
              kind: "service_due_soon",
              priority: "low",
              title: "Service Due Soon",
              message: `Service for ${labelForVehicle(v)} is due in ${svcDays} days.`,
              context: `${labelForVehicle(v)} • ${fmtDate(v.nextServiceDate)} • Maintenance`,
              dateLabel: fmtDate(v.nextServiceDate),
              vehicleId: v.id,
              action: {
                label: "Take Action",
                open: () => {
                  setEditVehicle(v);
                  setEditMode({
                    type: "service",
                    lastDate: toISODate(v.lastServiceDate),
                    nextDate: toISODate(v.nextServiceDate),
                    mileage: String(v.currentMileage ?? ""),
                  });
                },
              },
            });
        }
      }
    });

    // Optional low-priority info
    const fuelId = `fuel-tip-${new Date().toISOString().slice(0, 10)}`;
    if (!isDismissed(fuelId)) {
      list.push({
        id: fuelId,
        kind: "fuel_price",
        priority: "low",
        title: "Fuel Price Alert",
        message: "Petrol prices may increase next week. Consider refueling this weekend.",
        context: `${new Date().toLocaleDateString()} • Info`,
      });
    }

    // Sort by priority then by title
    const weight: Record<Priority, number> = { high: 0, medium: 1, low: 2 };
    return list.sort((a, b) => weight[a.priority] - weight[b.priority] || a.title.localeCompare(b.title));
  }, [vehicles]);

  const counts = React.useMemo(() => {
    let high = 0,
      medium = 0,
      low = 0;
    alerts.forEach((a) => {
      if (a.priority === "high") high++;
      else if (a.priority === "medium") medium++;
      else low++;
    });
    return { high, medium, low, total: alerts.length };
  }, [alerts]);

  /* --------------------------- Actions ---------------------------- */

  async function saveVehicle(id: string, patch: Partial<Vehicle>) {
    await api<Vehicle>("PUT", `/vehicles/${encodeURIComponent(id)}`, patch);
    await refetch();
  }

  function onDismiss(a: Alert) {
    dismissFor7Days(a.id);
    // Trigger a re-render by touching state (no need to refetch)
    setVehicles((v) => [...v]);
  }

  /* ------------------------------ UI -------------------------------- */

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-5rem)] grid place-items-center px-4 text-white bg-[radial-gradient(1200px_600px_at_50%_-200px,rgba(88,101,242,.35),rgba(2,8,23,1)_60%)]">
        <div className="max-w-md w-full bg-white/10 border border-white/10 rounded-2xl p-6 backdrop-blur-xl shadow-xl">
          Loading notifications…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-5rem)] px-3 sm:px-6 text-white bg-[radial-gradient(1200px_600px_at_50%_-200px,rgba(88,101,242,.35),rgba(2,8,23,1)_60%)]">
      <div className="max-w-6xl mx-auto py-6 space-y-6">
        {/* Header + summary */}
        <div className="flex items-center justify-between">
          <div>
            <div className="text-2xl sm:text-3xl font-semibold tracking-tight">Smart Notifications</div>
            <div className="text-white/70 text-sm">Stay on top of your vehicle maintenance and renewals</div>
          </div>
          <div className="text-white/70 text-sm">{counts.high + counts.medium} actions required</div>
        </div>

        {/* Stat tiles */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          <Tile label="High Priority" value={counts.high} icon="🚨" />
          <Tile label="Medium Priority" value={counts.medium} icon="🟧" />
          <Tile label="Low Priority" value={counts.low} icon="🔔" />
          <Tile label="Total Alerts" value={counts.total} icon="✅" />
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-500/30 rounded-2xl p-4 text-sm text-red-200">
            {error}
          </div>
        )}

        {/* Alerts list */}
        <div className="space-y-3">
          {alerts.length === 0 ? (
            <div className="bg-white/10 border border-white/10 rounded-2xl p-5 backdrop-blur-xl shadow-xl text-white/70">
              All clear. No alerts at the moment.
            </div>
          ) : (
            alerts.map((a) => (
              <AlertRow key={a.id} alert={a} onDismiss={onDismiss} />
            ))
          )}
        </div>

        {/* Smart predictions section (informational) */}
        <div className="bg-indigo-400/10 border border-indigo-300/20 rounded-2xl p-5 backdrop-blur-xl shadow-xl space-y-3">
          <div className="text-white/80 font-medium">Smart Predictions</div>
          <PredictionCard
            icon="🛣️"
            title="Fuel Efficiency Insight"
            text="Based on your driving patterns, you could save RM50/month by optimizing routes and consolidating trips."
          />
          <PredictionCard
            icon="🧰"
            title="Maintenance Prediction"
            text="Your vehicles will likely need brake pad replacement in 3–4 months based on current mileage trends."
          />
          <PredictionCard
            icon="💸"
            title="Budget Forecast"
            text="Expected vehicle expenses for next month: RM400–500 (fuel, maintenance, insurance)."
          />
        </div>
      </div>

      {/* Action modals */}
      {editVehicle && editMode?.type === "insurance" && (
        <Modal title="Update Insurance Expiry" onClose={() => setEditVehicle(null)}>
          <InsuranceForm
            dateDefault={(editMode as any).date || ""}
            onCancel={() => setEditVehicle(null)}
            onSave={async (d) => {
              await saveVehicle(editVehicle.id, { insuranceExpiry: toDbTimestamp(d) });
              setEditVehicle(null);
            }}
            vehicleLabel={labelForVehicle(editVehicle)}
          />
        </Modal>
      )}

      {editVehicle && editMode?.type === "roadtax" && (
        <Modal title="Update Road Tax Expiry" onClose={() => setEditVehicle(null)}>
          <RoadTaxForm
            dateDefault={(editMode as any).date || ""}
            onCancel={() => setEditVehicle(null)}
            onSave={async (d) => {
              await saveVehicle(editVehicle.id, { roadTaxExpiry: toDbTimestamp(d) });
              setEditVehicle(null);
            }}
            vehicleLabel={labelForVehicle(editVehicle)}
          />
        </Modal>
      )}

      {editVehicle && editMode?.type === "service" && (
        <Modal title="Record Service" onClose={() => setEditVehicle(null)}>
          <ServiceForm
            lastDefault={(editMode as any).lastDate || ""}
            nextDefault={(editMode as any).nextDate || ""}
            mileageDefault={(editMode as any).mileage || ""}
            onCancel={() => setEditVehicle(null)}
            onSave={async (payload) => {
              const patch: Partial<Vehicle> = {
                lastServiceDate: toDbTimestamp(payload.lastDate),
                nextServiceDate: toDbTimestamp(payload.nextDate),
                currentMileage:
                  payload.mileage === "" ? undefined : Number(payload.mileage),
              };
              await saveVehicle(editVehicle.id, patch);
              setEditVehicle(null);
            }}
            vehicleLabel={labelForVehicle(editVehicle)}
          />
        </Modal>
      )}
    </div>
  );
}

/* ----------------------------- Pieces ----------------------------- */

function Tile({ label, value, icon }: { label: string; value: number; icon: string }) {
  return (
    <div className="bg-white/10 border border-white/10 rounded-2xl p-4 backdrop-blur-xl shadow-xl">
      <div className="text-sm text-white/70">{label}</div>
      <div className="mt-1 text-2xl font-semibold flex items-center gap-2">
        <span>{icon}</span> {value}
      </div>
    </div>
  );
}

function AlertRow({
  alert,
  onDismiss,
}: {
  alert: Alert;
  onDismiss: (a: Alert) => void;
}) {
  const tone =
    alert.priority === "high"
      ? "border-red-400/60 bg-red-400/10"
      : alert.priority === "medium"
      ? "border-amber-400/60 bg-amber-400/10"
      : "border-sky-400/60 bg-sky-400/10";

  return (
    <div
      className={`rounded-2xl border ${tone} p-4 backdrop-blur-xl shadow-xl flex items-start justify-between gap-4`}
    >
      <div className="space-y-1">
        <div className="flex items-center gap-2">
          <span className="text-white/80 font-medium">{alert.title}</span>
          <PriorityPill level={alert.priority} />
          {(alert.kind === "insurance_expired" ||
            alert.kind === "roadtax_expired" ||
            alert.kind === "service_overdue") && <Badge text="Action Required" />}
        </div>
        <div className="text-white/80">{alert.message}</div>
        <div className="text-white/60 text-sm">{alert.context}</div>
      </div>

      <div className="flex items-center gap-3 shrink-0">
        {alert.action && (
          <button
            className="px-3 py-1.5 rounded-lg bg-white/20 hover:bg-white/25 border border-white/30"
            onClick={alert.action.open}
          >
            {alert.action.label}
          </button>
        )}
        <button
          className="text-white/60 hover:text-white/90"
          onClick={() => onDismiss(alert)}
          aria-label="Dismiss"
        >
          Dismiss
        </button>
      </div>
    </div>
  );
}

function PriorityPill({ level }: { level: Priority }) {
  const map: Record<Priority, string> = {
    high: "bg-red-500/20 border-red-500/40 text-red-100",
    medium: "bg-amber-500/20 border-amber-500/40 text-amber-100",
    low: "bg-sky-500/20 border-sky-500/40 text-sky-100",
  };
  const label = level === "high" ? "HIGH" : level === "medium" ? "MEDIUM" : "LOW";
  return <span className={`text-xs px-2 py-0.5 rounded-full border ${map[level]}`}>{label}</span>;
}

function Badge({ text }: { text: string }) {
  return (
    <span className="text-xs px-2 py-0.5 rounded-full border bg-white/10 border-white/20 text-white/80">
      {text}
    </span>
  );
}

/* ---------------------------- Modal + Forms ---------------------------- */

function Modal({
  title,
  children,
  onClose,
}: {
  title: string;
  children: React.ReactNode;
  onClose: () => void;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center">
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-10 w-full max-w-xl bg-white/10 border border-white/15 rounded-2xl shadow-2xl backdrop-blur-xl p-6 max-h-[80vh] overflow-y-auto">
        <div className="text-xl font-semibold">{title}</div>
        <div className="text-white/70 text-sm mb-4">Changes are saved to your account</div>
        {children}
      </div>
    </div>
  );
}

function Input({
  label,
  value,
  onChange,
  type = "text",
  placeholder,
  min,
  max,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
  min?: string | number;
  max?: string | number;
}) {
  return (
    <div>
      <label className="block text-sm text-white/80 mb-1">{label}</label>
      <input
        className="w-full rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        type={type}
        placeholder={placeholder}
        min={min}
        max={max}
      />
    </div>
  );
}

function ModalActions({
  onCancel,
  submitText,
}: {
  onCancel: () => void;
  submitText: string;
}) {
  return (
    <div className="flex items-center justify-end gap-3 pt-2">
      <button
        type="button"
        onClick={onCancel}
        className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/20 transition"
      >
        Cancel
      </button>
      <button
        type="submit"
        className="px-4 py-2 rounded-xl bg-white/20 hover:bg-white/25 border border-white/30 transition"
      >
        {submitText}
      </button>
    </div>
  );
}

function InsuranceForm({
  dateDefault,
  onSave,
  onCancel,
  vehicleLabel,
}: {
  dateDefault: string;
  onSave: (date: string) => void | Promise<void>;
  onCancel: () => void;
  vehicleLabel: string;
}) {
  const [date, setDate] = React.useState(dateDefault || "");
  const [saving, setSaving] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);
  return (
    <form
      className="space-y-4"
      onSubmit={async (e) => {
        e.preventDefault();
        setSaving(true);
        setErr(null);
        try {
          await onSave(date);
        } catch (ex: any) {
          setErr(ex?.message || "Failed to save");
        } finally {
          setSaving(false);
        }
      }}
    >
      <div className="text-white/70 text-sm">Vehicle: {vehicleLabel}</div>
      <Input label="New Insurance Expiry" type="date" value={date} onChange={setDate} />
      {err && <div className="text-sm text-red-300 bg-red-900/30 border border-red-500/30 rounded-lg p-2">{err}</div>}
      <ModalActions onCancel={onCancel} submitText={saving ? "Saving…" : "Save"} />
    </form>
  );
}

function RoadTaxForm({
  dateDefault,
  onSave,
  onCancel,
  vehicleLabel,
}: {
  dateDefault: string;
  onSave: (date: string) => void | Promise<void>;
  onCancel: () => void;
  vehicleLabel: string;
}) {
  const [date, setDate] = React.useState(dateDefault || "");
  const [saving, setSaving] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);
  return (
    <form
      className="space-y-4"
      onSubmit={async (e) => {
        e.preventDefault();
        setSaving(true);
        setErr(null);
        try {
          await onSave(date);
        } catch (ex: any) {
          setErr(ex?.message || "Failed to save");
        } finally {
          setSaving(false);
        }
      }}
    >
      <div className="text-white/70 text-sm">Vehicle: {vehicleLabel}</div>
      <Input label="New Road Tax Expiry" type="date" value={date} onChange={setDate} />
      {err && <div className="text-sm text-red-300 bg-red-900/30 border border-red-500/30 rounded-lg p-2">{err}</div>}
      <ModalActions onCancel={onCancel} submitText={saving ? "Saving…" : "Save"} />
    </form>
  );
}

function ServiceForm({
  lastDefault,
  nextDefault,
  mileageDefault,
  onSave,
  onCancel,
  vehicleLabel,
}: {
  lastDefault: string;
  nextDefault: string;
  mileageDefault: string;
  onSave: (payload: { lastDate: string; nextDate: string; mileage: string }) => void | Promise<void>;
  onCancel: () => void;
  vehicleLabel: string;
}) {
  const [lastDate, setLastDate] = React.useState(lastDefault || "");
  const [nextDate, setNextDate] = React.useState(nextDefault || "");
  const [mileage, setMileage] = React.useState(mileageDefault || "");
  const [saving, setSaving] = React.useState(false);
  const [err, setErr] = React.useState<string | null>(null);
  return (
    <form
      className="space-y-4"
      onSubmit={async (e) => {
        e.preventDefault();
        setSaving(true);
        setErr(null);
        try {
          await onSave({ lastDate, nextDate, mileage });
        } catch (ex: any) {
          setErr(ex?.message || "Failed to save");
        } finally {
          setSaving(false);
        }
      }}
    >
      <div className="text-white/70 text-sm">Vehicle: {vehicleLabel}</div>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <Input label="Last Service Date" type="date" value={lastDate} onChange={setLastDate} />
        <Input label="Next Service Date" type="date" value={nextDate} onChange={setNextDate} />
      </div>
      <Input
        label="Current Mileage (km)"
        type="number"
        value={String(mileage)}
        onChange={setMileage}
        min="0"
      />
      {err && <div className="text-sm text-red-300 bg-red-900/30 border border-red-500/30 rounded-lg p-2">{err}</div>}
      <ModalActions onCancel={onCancel} submitText={saving ? "Saving…" : "Save"} />
    </form>
  );
}

function PredictionCard({ icon, title, text }: { icon: string; title: string; text: string }) {
  return (
    <div className="bg-white/10 border border-white/10 rounded-xl p-4">
      <div className="flex items-center gap-2 font-medium">
        <span>{icon}</span>
        <span>{title}</span>
      </div>
      <div className="text-white/70 mt-1 text-sm">{text}</div>
    </div>
  );
}