import * as React from "react";

/**
 * - Aligned to your Prisma table "Vehicle":
 *   brand (required), model, year, plate, color, fuelType, chassisNumber, engineNumber,
 *   roadTaxExpiry, insuranceExpiry, lastServiceDate, nextServiceDate, currentMileage.
 * - All API calls include Authorization: Bearer <token> (OAuth identity).
 * - UI remains the frosted-glass look; only content/fields were aligned.
 */

/* ----------------------------- Types ----------------------------- */

type Vehicle = {
  id: string;
  brand: string;                // NOT NULL in DB
  model?: string | null;
  year?: number | null;
  plate?: string | null;
  color?: string | null;
  fuelType?: "Petrol" | "Diesel" | "Hybrid" | "Electric" | string | null;
  chassisNumber?: string | null;
  engineNumber?: string | null;

  roadTaxExpiry?: string | null;     // timestamp (no tz)
  insuranceExpiry?: string | null;
  lastServiceDate?: string | null;
  nextServiceDate?: string | null;

  currentMileage?: number | null;

  createdAt?: string;
  updatedAt?: string;
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

// UI date input (YYYY-MM-DD) -> DB "timestamp without time zone" string (no Z)
function toDbTimestamp(localDate: string | ""): string | null {
  if (!localDate) return null;
  const [y, m, d] = localDate.split("-").map(Number);
  if (!y || !m || !d) return null;
  const dt = new Date(y, m - 1, d, 0, 0, 0);
  // Render without trailing "Z" so Postgres stores literal local date-time
  const yyyy = dt.getFullYear();
  const mm = String(dt.getMonth() + 1).padStart(2, "0");
  const dd = String(dt.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}T00:00:00.000`;
}

const toISODate = (s?: string | null) => {
  if (!s) return "";
  const d = new Date(s);
  if (isNaN(+d)) return "";
  // normalize to local date for input value
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
};

const fmtDate = (s?: string | null) => (s ? toISODate(s) : "—");
const fmtKm = (n?: number | null) =>
  typeof n === "number" && !isNaN(n) ? `${n.toLocaleString()} km` : "—";
const yearNow = new Date().getFullYear();

function statusPill(dateIso?: string | null): { label: string; tone: "ok" | "expired" } {
  if (!dateIso) return { label: "—", tone: "ok" };
  const d = new Date(dateIso);
  if (isNaN(+d)) return { label: "—", tone: "ok" };
  const today = new Date();
  const a = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const b = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  return a < b ? { label: "Expired", tone: "expired" } : { label: "OK", tone: "ok" };
}

/* -------------------------- Main Component ----------------------- */

export default function VehicleManager() {
  const [vehicles, setVehicles] = React.useState<Vehicle[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Filters
  const [q, setQ] = React.useState("");

  // Modal state
  const [showAdd, setShowAdd] = React.useState(false);
  const [showEdit, setShowEdit] = React.useState<Vehicle | null>(null);
  const [confirmDelete, setConfirmDelete] = React.useState<Vehicle | null>(null);

  // Form state (aligned to DB)
  const [form, setForm] = React.useState<{
    brand: string; // required
    model: string;
    year: number | "";
    plate: string;
    color: string;
    fuelType: Vehicle["fuelType"];
    chassisNumber: string;
    engineNumber: string;

    roadTaxExpiry: string;
    insuranceExpiry: string;
    lastServiceDate: string;
    nextServiceDate: string;

    currentMileage: number | "";
  }>({
    brand: "",
    model: "",
    year: "",
    plate: "",
    color: "",
    fuelType: "Petrol",
    chassisNumber: "",
    engineNumber: "",

    roadTaxExpiry: "",
    insuranceExpiry: "",
    lastServiceDate: "",
    nextServiceDate: "",

    currentMileage: "",
  });

  const [saving, setSaving] = React.useState(false);
  const [formError, setFormError] = React.useState<string | null>(null);

  const refetch = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const v = await api<Vehicle[]>("GET", "/vehicles");
      setVehicles(v || []);
    } catch (err: any) {
      setError(err?.message || "Failed to load vehicles");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    refetch();
  }, [refetch]);

  /* --------------------------- Derived data --------------------------- */

  const filtered = React.useMemo(() => {
    const qlc = q.trim().toLowerCase();
    return vehicles
      .filter((x) => {
        if (!qlc) return true;
        const s = `${x.brand ?? ""} ${x.model ?? ""} ${x.plate ?? ""} ${x.year ?? ""}`.toLowerCase();
        return s.includes(qlc);
      })
      .sort((a, b) => {
        const ad = new Date(a.createdAt || 0).getTime();
        const bd = new Date(b.createdAt || 0).getTime();
        return bd - ad;
      });
  }, [vehicles, q]);

  /* ------------------------------ Actions ----------------------------- */

  function openAdd() {
    setForm({
      brand: "",
      model: "",
      year: "",
      plate: "",
      color: "",
      fuelType: "Petrol",
      chassisNumber: "",
      engineNumber: "",

      roadTaxExpiry: "",
      insuranceExpiry: "",
      lastServiceDate: "",
      nextServiceDate: "",

      currentMileage: "",
    });
    setFormError(null);
    setShowAdd(true);
  }

  function openEdit(row: Vehicle) {
    setForm({
      brand: row.brand || "",
      model: row.model || "",
      year: typeof row.year === "number" ? row.year : "",
      plate: row.plate || "",
      color: row.color || "",
      fuelType: (row.fuelType as any) || "Petrol",
      chassisNumber: row.chassisNumber || "",
      engineNumber: row.engineNumber || "",

      roadTaxExpiry: toISODate(row.roadTaxExpiry),
      insuranceExpiry: toISODate(row.insuranceExpiry),
      lastServiceDate: toISODate(row.lastServiceDate),
      nextServiceDate: toISODate(row.nextServiceDate),

      currentMileage: typeof row.currentMileage === "number" ? row.currentMileage : "",
    });
    setFormError(null);
    setShowEdit(row);
  }

  async function submitAdd(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setFormError(null);
    try {
      const payload: Partial<Vehicle> = {
        brand: form.brand.trim() || undefined, // DB requires brand; keep required in UI
        model: form.model.trim() || null,
        year: form.year === "" ? null : Number(form.year),
        plate: form.plate.trim() || null,
        color: form.color.trim() || null,
        fuelType: form.fuelType || null,
        chassisNumber: form.chassisNumber.trim() || null,
        engineNumber: form.engineNumber.trim() || null,

        roadTaxExpiry: toDbTimestamp(form.roadTaxExpiry),
        insuranceExpiry: toDbTimestamp(form.insuranceExpiry),
        lastServiceDate: toDbTimestamp(form.lastServiceDate),
        nextServiceDate: toDbTimestamp(form.nextServiceDate),

        currentMileage: form.currentMileage === "" ? 0 : Number(form.currentMileage),
      };
      await api<Vehicle>("POST", "/vehicles", payload);
      setShowAdd(false);
      await refetch();
    } catch (err: any) {
      setFormError(err?.message || "Failed to add vehicle");
    } finally {
      setSaving(false);
    }
  }

  async function submitEdit(e: React.FormEvent) {
    e.preventDefault();
    if (!showEdit) return;
    setSaving(true);
    setFormError(null);
    try {
      const payload: Partial<Vehicle> = {
        brand: form.brand.trim() || undefined,
        model: form.model.trim() || null,
        year: form.year === "" ? null : Number(form.year),
        plate: form.plate.trim() || null,
        color: form.color.trim() || null,
        fuelType: form.fuelType || null,
        chassisNumber: form.chassisNumber.trim() || null,
        engineNumber: form.engineNumber.trim() || null,

        roadTaxExpiry: toDbTimestamp(form.roadTaxExpiry),
        insuranceExpiry: toDbTimestamp(form.insuranceExpiry),
        lastServiceDate: toDbTimestamp(form.lastServiceDate),
        nextServiceDate: toDbTimestamp(form.nextServiceDate),

        currentMileage: form.currentMileage === "" ? 0 : Number(form.currentMileage),
      };
      await api<Vehicle>("PUT", `/vehicles/${encodeURIComponent(showEdit.id)}`, payload);
      setShowEdit(null);
      await refetch();
    } catch (err: any) {
      setFormError(err?.message || "Failed to update vehicle");
    } finally {
      setSaving(false);
    }
  }

  async function submitDelete() {
    if (!confirmDelete) return;
    try {
      await api<void>("DELETE", `/vehicles/${encodeURIComponent(confirmDelete.id)}`);
      setConfirmDelete(null);
      await refetch();
    } catch (err: any) {
      alert(err?.message || "Failed to delete vehicle");
    }
  }

  /* ------------------------------- UI -------------------------------- */

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-5rem)] grid place-items-center px-4 text-white bg-[radial-gradient(1200px_600px_at_50%_-200px,rgba(88,101,242,.35),rgba(2,8,23,1)_60%)]">
        <div className="max-w-md w-full bg-white/10 border border-white/10 rounded-2xl p-6 backdrop-blur-xl shadow-xl">
          Loading your vehicles…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-5rem)] px-3 sm:px-6 text-white bg-[radial-gradient(1200px_600px_at_50%_-200px,rgba(88,101,242,.35),rgba(2,8,23,1)_60%)]">
      <div className="max-w-7xl mx-auto py-6 space-y-6">
        {/* Header */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-2xl sm:text-3xl font-semibold tracking-tight">Vehicle Management</div>
            <div className="text-white/70 text-sm">Manage your fleet with Malaysian regulatory compliance</div>
          </div>
          <div className="flex items-center gap-3">
            <button
              className="glass-btn px-4 py-2 rounded-xl bg-white/15 hover:bg-white/20 border border-white/20 backdrop-blur-xl shadow-lg transition"
              onClick={openAdd}
            >
              + Add Vehicle
            </button>
          </div>
        </div>

        {/* Search */}
        <div className="bg-white/10 border border-white/10 rounded-2xl p-4 backdrop-blur-xl shadow-xl">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            <input
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              placeholder="Search brand, model, or plate"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
            <div className="rounded-xl bg-white/5 border border-white/10 px-3 py-2 text-white/70">
              Total vehicles: <span className="text-white">{filtered.length}</span>
            </div>
            <div className="rounded-xl bg-white/5 border border-white/10 px-3 py-2 text-white/60">
              Tip: edit to keep compliance dates up to date.
            </div>
          </div>
        </div>

        {/* Cards (sample-alike) */}
        <div className="space-y-4">
          {filtered.length === 0 ? (
            <div className="bg-white/10 border border-white/10 rounded-2xl p-6 backdrop-blur-xl shadow-xl text-white/70">
              No vehicles yet. Add one to get started.
            </div>
          ) : (
            filtered.map((v) => {
              const road = statusPill(v.roadTaxExpiry);
              const ins = statusPill(v.insuranceExpiry);
              const svc = statusPill(v.nextServiceDate);
              return (
                <div key={v.id} className="bg-white/10 border border-white/10 rounded-2xl p-5 backdrop-blur-xl shadow-xl">
                  {/* Title row */}
                  <div className="flex items-start justify-between gap-3">
                    <div>
                      <div className="text-lg font-semibold">
                        {`${v.brand ?? "Vehicle"} ${v.model ?? ""}`.trim() || "Vehicle"}{" "}
                        {v.year ? `(${v.year})` : ""}
                      </div>
                      <div className="text-white/70 text-sm">
                        {v.plate ? `${v.plate} • ` : ""}
                        {v.color ? `${v.color} • ` : ""}
                        {v.fuelType ?? "—"}
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20"
                        onClick={() => openEdit(v)}
                        title="Edit"
                        aria-label="Edit vehicle"
                      >
                        ✏️
                      </button>
                      <button
                        className="px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40"
                        onClick={() => setConfirmDelete(v)}
                        title="Delete"
                        aria-label="Delete vehicle"
                      >
                        🗑️
                      </button>
                    </div>
                  </div>

                  {/* Compliance row */}
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-4">
                    <ComplianceCard label="Road Tax" status={road} date={fmtDate(v.roadTaxExpiry)} icon="📅" />
                    <ComplianceCard label="Insurance" status={ins} date={fmtDate(v.insuranceExpiry)} icon="🛡️" />
                    <ComplianceCard label="Next Service" status={svc} date={fmtDate(v.nextServiceDate)} icon="🔧" />
                  </div>

                  {/* Mileage bar */}
                  <div className="mt-4">
                    <div className="text-white/70 text-sm">
                      Current Mileage:{" "}
                      <span className="text-white font-medium">{fmtKm(v.currentMileage)}</span>
                    </div>
                    <div className="h-2 mt-2 rounded-full bg-white/10 overflow-hidden">
                      <div
                        className="h-full bg-white/40"
                        style={{
                          width: `${Math.min(
                            100,
                            Math.max(0, (Number(v.currentMileage || 0) % 100000) / 1000)
                          )}%`,
                        }}
                      />
                    </div>
                  </div>
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Add Modal (scrollable, 2-column like your sample) */}
      {showAdd && (
        <Modal onClose={() => setShowAdd(false)} title="Add New Vehicle">
          {formError && <ErrorBanner text={formError} />}

          <form className="space-y-6" onSubmit={submitAdd}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Basic Information */}
              <section>
                <h4 className="font-medium mb-3">Basic Information</h4>
                <div className="space-y-3">
                  <Select
                    label="Brand"
                    value={form.brand}
                    onChange={(v) => setForm((f) => ({ ...f, brand: v }))}
                    options={BRANDS}
                    placeholder="Select Brand"
                    required
                  />
                  <Input label="Model" placeholder="e.g., Vios, Accord" value={form.model} onChange={(v) => setForm((f) => ({ ...f, model: v }))} />
                  <Input label="Year" type="number" min="1900" max={String(yearNow + 1)} value={String(form.year ?? "")} onChange={(v) => setForm((f) => ({ ...f, year: v === "" ? "" : Number(v) }))} />
                  <Input label="Plate Number" placeholder="e.g., WA 1234 A" value={form.plate} onChange={(v) => setForm((f) => ({ ...f, plate: v }))} />
                </div>
              </section>

              {/* Technical Details */}
              <section>
                <h4 className="font-medium mb-3">Technical Details</h4>
                <div className="space-y-3">
                  <Input label="Chassis Number" placeholder="17-digit chassis number" value={form.chassisNumber} onChange={(v) => setForm((f) => ({ ...f, chassisNumber: v }))} />
                  <Input label="Engine Number" placeholder="Engine identification number" value={form.engineNumber} onChange={(v) => setForm((f) => ({ ...f, engineNumber: v }))} />
                  <Input label="Color" placeholder="e.g., Silver, White, Black" value={form.color} onChange={(v) => setForm((f) => ({ ...f, color: v }))} />
                  <Select label="Fuel Type" value={String(form.fuelType || "")} onChange={(v) => setForm((f) => ({ ...f, fuelType: v as Vehicle["fuelType"] }))} options={["Petrol", "Diesel", "Hybrid", "Electric"]} />
                </div>
              </section>
            </div>

            {/* Compliance & Maintenance */}
            <section>
              <h4 className="font-medium mb-3">Compliance & Maintenance</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Input label="Road Tax Expiry" type="date" value={form.roadTaxExpiry} onChange={(v) => setForm((f) => ({ ...f, roadTaxExpiry: v }))} />
                <Input label="Insurance Expiry" type="date" value={form.insuranceExpiry} onChange={(v) => setForm((f) => ({ ...f, insuranceExpiry: v }))} />
                <Input label="Current Mileage (km)" type="number" min="0" value={String(form.currentMileage)} onChange={(v) => setForm((f) => ({ ...f, currentMileage: v === "" ? "" : Number(v) }))} />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <Input label="Last Service Date" type="date" value={form.lastServiceDate} onChange={(v) => setForm((f) => ({ ...f, lastServiceDate: v }))} />
                <Input label="Next Service Date" type="date" value={form.nextServiceDate} onChange={(v) => setForm((f) => ({ ...f, nextServiceDate: v }))} />
              </div>
            </section>

            <ModalActions onCancel={() => setShowAdd(false)} submitText={saving ? "Adding…" : "Add Vehicle"} />
          </form>
        </Modal>
      )}

      {/* Edit Modal */}
      {showEdit && (
        <Modal onClose={() => setShowEdit(null)} title="Edit Vehicle">
          {formError && <ErrorBanner text={formError} />}

          <form className="space-y-6" onSubmit={submitEdit}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <section>
                <h4 className="font-medium mb-3">Basic Information</h4>
                <div className="space-y-3">
                  <Select label="Brand" value={form.brand} onChange={(v) => setForm((f) => ({ ...f, brand: v }))} options={BRANDS} placeholder="Select Brand" required />
                  <Input label="Model" placeholder="e.g., Vios, Accord" value={form.model} onChange={(v) => setForm((f) => ({ ...f, model: v }))} />
                  <Input label="Year" type="number" min="1900" max={String(yearNow + 1)} value={String(form.year ?? "")} onChange={(v) => setForm((f) => ({ ...f, year: v === "" ? "" : Number(v) }))} />
                  <Input label="Plate Number" placeholder="e.g., WA 1234 A" value={form.plate} onChange={(v) => setForm((f) => ({ ...f, plate: v }))} />
                </div>
              </section>

              <section>
                <h4 className="font-medium mb-3">Technical Details</h4>
                <div className="space-y-3">
                  <Input label="Chassis Number" placeholder="17-digit chassis number" value={form.chassisNumber} onChange={(v) => setForm((f) => ({ ...f, chassisNumber: v }))} />
                  <Input label="Engine Number" placeholder="Engine identification number" value={form.engineNumber} onChange={(v) => setForm((f) => ({ ...f, engineNumber: v }))} />
                  <Input label="Color" placeholder="e.g., Silver, White, Black" value={form.color} onChange={(v) => setForm((f) => ({ ...f, color: v }))} />
                  <Select label="Fuel Type" value={String(form.fuelType || "")} onChange={(v) => setForm((f) => ({ ...f, fuelType: v as Vehicle["fuelType"] }))} options={["Petrol", "Diesel", "Hybrid", "Electric"]} />
                </div>
              </section>
            </div>

            <section>
              <h4 className="font-medium mb-3">Compliance & Maintenance</h4>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <Input label="Road Tax Expiry" type="date" value={form.roadTaxExpiry} onChange={(v) => setForm((f) => ({ ...f, roadTaxExpiry: v }))} />
                <Input label="Insurance Expiry" type="date" value={form.insuranceExpiry} onChange={(v) => setForm((f) => ({ ...f, insuranceExpiry: v }))} />
                <Input label="Current Mileage (km)" type="number" min="0" value={String(form.currentMileage)} onChange={(v) => setForm((f) => ({ ...f, currentMileage: v === "" ? "" : Number(v) }))} />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                <Input label="Last Service Date" type="date" value={form.lastServiceDate} onChange={(v) => setForm((f) => ({ ...f, lastServiceDate: v }))} />
                <Input label="Next Service Date" type="date" value={form.nextServiceDate} onChange={(v) => setForm((f) => ({ ...f, nextServiceDate: v }))} />
              </div>
            </section>

            <ModalActions onCancel={() => setShowEdit(null)} submitText={saving ? "Saving…" : "Save Changes"} />
          </form>
        </Modal>
      )}

      {/* Delete Confirmation */}
      {confirmDelete && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setConfirmDelete(null)} />
          <div className="relative z-10 w-full max-w-md bg-white/10 border border-white/15 rounded-2xl shadow-2xl backdrop-blur-xl p-6">
            <div className="text-xl font-semibold">Delete Vehicle</div>
            <div className="text-white/70 text-sm mt-1">
              Are you sure you want to delete{" "}
              <span className="font-medium">
                {confirmDelete.plate || `${confirmDelete.brand ?? ""} ${confirmDelete.model ?? ""}`.trim() || "this vehicle"}
              </span>
              ?
            </div>
            <div className="flex items-center justify-end gap-3 mt-6">
              <button className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/20" onClick={() => setConfirmDelete(null)}>
                Cancel
              </button>
              <button className="px-4 py-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40" onClick={submitDelete}>
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

/* ---------------------------- UI Primitives --------------------------- */

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
      {/* overlay */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      {/* modal */}
      <div className="relative z-10 w-full max-w-4xl bg-white/10 border border-white/15 rounded-2xl shadow-2xl backdrop-blur-xl p-6 max-h-[80vh] overflow-y-auto">
        <div className="text-xl font-semibold">{title}</div>
        <div className="text-white/70 text-sm mb-4">Saved to your account via OAuth identity</div>
        {children}
      </div>
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
      <button type="button" onClick={onCancel} className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/20 transition">
        Cancel
      </button>
      <button type="submit" className="px-4 py-2 rounded-xl bg-white/20 hover:bg-white/25 border border-white/30 backdrop-blur-xl shadow-lg transition">
        {submitText}
      </button>
    </div>
  );
}

function Input({
  label,
  value,
  onChange,
  placeholder,
  type = "text",
  required,
  step,
  min,
  max,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  required?: boolean;
  step?: string;
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
        placeholder={placeholder}
        type={type}
        required={required}
        step={step}
        min={min}
        max={max}
      />
    </div>
  );
}

function Select({
  label,
  value,
  onChange,
  options,
  placeholder,
  required,
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: string[];
  placeholder?: string;
  required?: boolean;
}) {
  return (
    <div>
      <label className="block text-sm text-white/80 mb-1">{label}</label>
      <select
        className="w-full rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
      >
        {placeholder && <option value="">{placeholder}</option>}
        {options.map((opt) => (
          <option key={opt} value={opt}>
            {opt}
          </option>
        ))}
      </select>
    </div>
  );
}

function ErrorBanner({ text }: { text: string }) {
  return (
    <div className="mb-3 text-sm text-red-300 bg-red-900/30 border border-red-500/30 rounded-lg p-2">
      {text}
    </div>
  );
}

function ComplianceCard({
  label,
  status,
  date,
  icon,
}: {
  label: string;
  status: { label: string; tone: "ok" | "expired" };
  date: string;
  icon: string;
}) {
  return (
    <div className="bg-white/10 border border-white/10 rounded-xl p-4">
      <div className="flex items-center gap-2">
        <span className="text-lg">{icon}</span>
        <div className="font-medium">{label}</div>
      </div>
      <div className="mt-2 flex items-center gap-2">
        <span
          className={
            "text-xs px-2 py-0.5 rounded-full border " +
            (status.tone === "expired"
              ? "bg-red-500/20 border-red-500/40 text-red-200"
              : "bg-emerald-500/20 border-emerald-500/40 text-emerald-100")
          }
        >
          {status.label}
        </span>
        <span className="text-sm text-white/70">{date}</span>
      </div>
    </div>
  );
}

/* ------------------------------ Constants ---------------------------- */

const BRANDS = [
  "Toyota","Honda","Perodua","Proton","Nissan","Mazda","BMW","Mercedes-Benz","Audi","Volkswagen",
  "Hyundai","Kia","Volvo","Mitsubishi","Subaru",
];