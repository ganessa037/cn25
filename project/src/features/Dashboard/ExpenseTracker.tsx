import * as React from "react";

/**
 * Expense Tracker
 * - CRUD via /api/expenses
 * - Reads Authorization token from localStorage.user.token
 * - Glass UI, modal forms (scrollable), responsive table
 * - Aligns with Prisma schema:
 *   enum ExpenseCategory { Fuel, Maintenance, Insurance, RoadTax, Toll_Parking, Other }
 *   model Expense { id, userId, vehicleId (required), title, category, amount, date, ... }
 */

/* ----------------------- Category mapping (UI <-> DB) ---------------------- */

const CATEGORY_OPTIONS_UI = [
  "Fuel",
  "Maintenance",
  "Insurance",
  "Road Tax",
  "Toll/Parking",
  "Other",
] as const;

type CategoryUI = typeof CATEGORY_OPTIONS_UI[number];
type CategoryDB =
  | "Fuel"
  | "Maintenance"
  | "Insurance"
  | "RoadTax"
  | "Toll_Parking"
  | "Other";

const CATEGORY_UI_TO_DB: Record<CategoryUI | string, CategoryDB> = {
  Fuel: "Fuel",
  Maintenance: "Maintenance",
  Insurance: "Insurance",
  "Road Tax": "RoadTax",
  "Toll/Parking": "Toll_Parking",
  Other: "Other",

  // tolerate old labels
  Service: "Maintenance",
  Tax: "RoadTax",
  Toll: "Toll_Parking",
  Parts: "Maintenance",
  General: "Other",
};

const CATEGORY_DB_TO_UI: Record<CategoryDB, CategoryUI> = {
  Fuel: "Fuel",
  Maintenance: "Maintenance",
  Insurance: "Insurance",
  RoadTax: "Road Tax",
  Toll_Parking: "Toll/Parking",
  Other: "Other",
};

/* --------------------------------- Types ---------------------------------- */

type Expense = {
  id: string;
  title: string;
  amount: number;
  category: CategoryDB; // stored as DB token
  date: string; // ISO
  vehicleId: string;
  createdAt?: string;
  updatedAt?: string;
};

type Vehicle = {
  id: string;
  brand?: string | null; // Prisma uses 'brand'
  model?: string | null;
  plate?: string | null; // Prisma uses 'plate'
  year?: number | null;
};

/* --------------------------- API utilities -------------------------------- */

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

async function api<T>(
  method: "GET" | "POST" | "PUT" | "DELETE",
  path: string,
  body?: unknown
): Promise<T> {
  const res = await fetch(`${API}${path}`, {
    method,
    headers: {
      "Content-Type": "application/json",
      ...(token() ? { Authorization: `Bearer ${token()}` } : {}),
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}${text ? ` — ${text}` : ""}`);
  }
  return (await res.json()) as T;
}

/* ------------------------------ Helpers ----------------------------------- */

const fmtMY = new Intl.NumberFormat("en-MY", {
  style: "currency",
  currency: "MYR",
});

const toISOyyyyMMdd = (d: Date | string) => {
  const base = typeof d === "string" ? new Date(d) : d;
  // keep the date as local date (no TZ shift)
  const local = new Date(base.getTime() - base.getTimezoneOffset() * 60000);
  return local.toISOString().slice(0, 10);
};
Select
/** Accepts 'yyyy-mm-dd' or 'dd.mm.yyyy' and returns a full ISO string */
function toApiDate(input: string | undefined | null) {
  if (!input) return new Date().toISOString();
  if (/^\d{4}-\d{2}-\d{2}$/.test(input)) return new Date(`${input}T00:00:00`).toISOString();
  const m = input.match(/^(\d{2})\.(\d{2})\.(\d{4})$/);
  if (m) {
    const [, dd, mm, yyyy] = m;
    return new Date(`${yyyy}-${mm}-${dd}T00:00:00`).toISOString();
  }
  // try native parsing as last resort
  return new Date(input).toISOString();
}

function vehicleLabel(v?: Vehicle) {
  if (!v) return "—";
  return v.plate?.trim() || `${v.brand ?? ""} ${v.model ?? ""}`.trim() || v.id.slice(0, 6);
}

/* --------------------------- Component ------------------------------------ */

export default function ExpenseTracker() {
  const [expenses, setExpenses] = React.useState<Expense[]>([]);
  const [vehicles, setVehicles] = React.useState<Vehicle[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // filters
  const [q, setQ] = React.useState("");
  const [category, setCategory] = React.useState<"All" | CategoryUI>("All");
  const [vehicleId, setVehicleId] = React.useState<string>("");
  const [from, setFrom] = React.useState<string>("");
  const [to, setTo] = React.useState<string>("");

  // modals
  const [showAdd, setShowAdd] = React.useState(false);
  const [showEdit, setShowEdit] = React.useState<Expense | null>(null);
  const [confirmDelete, setConfirmDelete] = React.useState<Expense | null>(null);

  // form
  const [form, setForm] = React.useState<{
    title: string;
    amount: number | "";
    category: CategoryUI;
    date: string; // yyyy-mm-dd
    vehicleId: string;
    description?: string;
  }>({
    title: "",
    amount: "",
    category: "Other",
    date: toISOyyyyMMdd(new Date()),
    vehicleId: "",
  });

  const [saving, setSaving] = React.useState(false);
  const [formError, setFormError] = React.useState<string | null>(null);

  const refetch = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [e, v] = await Promise.all([
        api<Expense[]>("GET", "/expenses"),
        api<Vehicle[]>("GET", "/vehicles"),
      ]);
      setExpenses(e || []);
      setVehicles(v || []);
    } catch (err: any) {
      setError(err?.message || "Failed to load expenses");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    refetch();
  }, [refetch]);

  /* ------------------------ Derived / filtered list ----------------------- */

  const filtered = React.useMemo(() => {
    const qLower = q.trim().toLowerCase();
    const fromD = from ? new Date(from) : null;
    const toD = to ? new Date(to) : null;
    const wantDbCat = category === "All" ? null : CATEGORY_UI_TO_DB[category];

    return expenses
      .filter((x) => {
        const title = (x.title || "").toLowerCase();
        const uiCat = CATEGORY_DB_TO_UI[x.category] || x.category;
        const hay = `${title} ${uiCat}`.toLowerCase();
        if (qLower && !hay.includes(qLower)) return false;

        if (wantDbCat && x.category !== wantDbCat) return false;
        if (vehicleId && x.vehicleId !== vehicleId) return false;

        const d = new Date(x.date);
        if (fromD && d < fromD) return false;
        if (toD && d > toD) return false;
        return true;
      })
      .sort((a, b) => new Date(b.date).getTime() - new Date(a.date).getTime());
  }, [expenses, q, category, vehicleId, from, to]);

  const total = React.useMemo(
    () => filtered.reduce((s, x) => s + Number(x.amount || 0), 0),
    [filtered]
  );

  /* ------------------------------- Actions -------------------------------- */

  function openAdd() {
    setForm({
      title: "",
      amount: "",
      category: "Other",
      date: toISOyyyyMMdd(new Date()),
      vehicleId: "", // force user to pick
    });
    setFormError(null);
    setShowAdd(true);
  }

  function openEdit(row: Expense) {
    setForm({
      title: row.title || "",
      amount: Number(row.amount ?? 0),
      category: CATEGORY_DB_TO_UI[row.category] || "Other",
      date: row.date ? toISOyyyyMMdd(row.date) : toISOyyyyMMdd(new Date()),
      vehicleId: row.vehicleId || "",
    });
    setFormError(null);
    setShowEdit(row);
  }

  async function submitAdd(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setFormError(null);
    try {
      if (!form.vehicleId) throw new Error("Please select a vehicle (required).");

      const payload = {
        title: form.title.trim() || "Expense",
        amount: Number(form.amount || 0),
        category: CATEGORY_UI_TO_DB[form.category] ?? "Other",
        date: toApiDate(form.date),
        vehicleId: String(form.vehicleId),
        description: form.description?.trim() || undefined,
      };

      if (!Number.isFinite(payload.amount) || payload.amount <= 0) {
        throw new Error("Please enter a valid amount.");
      }

      await api<Expense>("POST", "/expenses", payload);
      setShowAdd(false);
      await refetch();
    } catch (err: any) {
      setFormError(err?.message || "Failed to create expense");
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
      if (!form.vehicleId) throw new Error("Please select a vehicle (required).");

      const payload = {
        title: form.title.trim() || "Expense",
        amount: Number(form.amount || 0),
        category: CATEGORY_UI_TO_DB[form.category] ?? "Other",
        date: toApiDate(form.date),
        vehicleId: String(form.vehicleId),
        description: form.description?.trim() || undefined,
      };

      await api<Expense>("PUT", `/expenses/${encodeURIComponent(showEdit.id)}`, payload);
      setShowEdit(null);
      await refetch();
    } catch (err: any) {
      setFormError(err?.message || "Failed to update expense");
    } finally {
      setSaving(false);
    }
  }

  async function submitDelete() {
    if (!confirmDelete) return;
    try {
      await api<void>("DELETE", `/expenses/${encodeURIComponent(confirmDelete.id)}`);
      setConfirmDelete(null);
      await refetch();
    } catch (err: any) {
      alert(err?.message || "Failed to delete expense");
    }
  }

  /* --------------------------------- UI ----------------------------------- */

  return (
    <div className="min-h-[calc(100vh-5rem)] px-3 sm:px-6 text-white">
      <div className="max-w-7xl mx-auto py-6 space-y-6">
        {/* Heading */}
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <div className="text-2xl sm:text-3xl font-semibold tracking-tight">Expense Tracker</div>
            <div className="text-white/70 text-sm">Record and review your spending</div>
          </div>
          <button
            className="px-4 py-2 rounded-xl bg-white/15 hover:bg-white/20 border border-white/20 backdrop-blur-xl shadow-lg transition"
            onClick={openAdd}
          >
            + Add Expense
          </button>
        </div>

        {/* Filters */}
        <div className="bg-white/10 border border-white/10 rounded-2xl p-4 backdrop-blur-xl shadow-xl">
          <div className="grid grid-cols-1 md:grid-cols-5 gap-3">
            <input
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              placeholder="Search title or category"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />

            <select
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              value={category}
              onChange={(e) => setCategory(e.target.value as any)}
            >
              <option value="All">All</option>
              {CATEGORY_OPTIONS_UI.map((c) => (
                <option key={c} value={c}>
                  {c}
                </option>
              ))}
            </select>

            <select
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              value={vehicleId}
              onChange={(e) => setVehicleId(e.target.value)}
            >
              <option value="">All Vehicles</option>
              {vehicles.map((v) => (
                <option key={v.id} value={v.id}>
                  {vehicleLabel(v)}
                </option>
              ))}
            </select>

            <input
              type="date"
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              value={from}
              onChange={(e) => setFrom(e.target.value)}
              placeholder="From"
            />
            <input
              type="date"
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              value={to}
              onChange={(e) => setTo(e.target.value)}
              placeholder="To"
            />
          </div>
          <div className="mt-3 text-sm text-white/70">
            Showing {filtered.length} of {expenses.length} • Total {fmtMY.format(total)}
          </div>
        </div>

        {/* Loading */}
        {loading && (
          <div className="bg-white/10 border border-white/10 rounded-2xl p-4 backdrop-blur-xl shadow-xl text-white/80">
            Loading expenses…
          </div>
        )}

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-500/30 rounded-2xl p-4 text-sm text-red-200">
            {error}
          </div>
        )}

        {/* Table */}
        <div className="bg-white/10 border border-white/10 rounded-2xl backdrop-blur-xl shadow-xl overflow-hidden">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead className="bg-white/10 text-white/80">
                <tr>
                  <th className="text-left px-4 py-3">Date</th>
                  <th className="text-left px-4 py-3">Title</th>
                  <th className="text-left px-4 py-3">Category</th>
                  <th className="text-right px-4 py-3">Amount</th>
                  <th className="text-left px-4 py-3">Vehicle</th>
                  <th className="text-right px-4 py-3">Actions</th>
                </tr>
              </thead>
              <tbody>
                {filtered.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="px-4 py-6 text-center text-white/60">
                      No expenses found. Try adjusting filters or add a new expense.
                    </td>
                  </tr>
                ) : (
                  filtered.map((x) => (
                    <tr key={x.id} className="border-t border-white/10 hover:bg-white/5">
                      <td className="px-4 py-3">{toISOyyyyMMdd(x.date)}</td>
                      <td className="px-4 py-3">{x.title}</td>
                      <td className="px-4 py-3">{CATEGORY_DB_TO_UI[x.category] || x.category}</td>
                      <td className="px-4 py-3 text-right">{fmtMY.format(Number(x.amount || 0))}</td>
                      <td className="px-4 py-3">{vehicleLabel(vehicles.find((v) => v.id === x.vehicleId))}</td>
                      <td className="px-4 py-3 text-right">
                        <button
                          className="px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20 mr-2"
                          onClick={() => openEdit(x)}
                        >
                          Edit
                        </button>
                        <button
                          className="px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40"
                          onClick={() => setConfirmDelete(x)}
                        >
                          Delete
                        </button>
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Add Modal */}
      {showAdd && (
        <Modal onClose={() => setShowAdd(false)} title="Add Expense">
          {formError && <ErrorBanner text={formError} />}
          <form className="space-y-4" onSubmit={submitAdd}>
            <Input
              label="Title"
              placeholder="e.g., Fuel, Road Tax, Maintenance"
              value={form.title}
              onChange={(v) => setForm((f) => ({ ...f, title: v }))}
              required
            />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Input
                label="Amount (RM)"
                type="number"
                step="0.01"
                min="0"
                value={String(form.amount)}
                onChange={(v) => setForm((f) => ({ ...f, amount: v === "" ? "" : Number(v) }))}
                required
              />
              <Input
                label="Date"
                type="date"
                value={form.date}
                onChange={(v) => setForm((f) => ({ ...f, date: v }))}
                required
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Select
                label="Category"
                value={form.category}
                onChange={(v) => setForm((f) => ({ ...f, category: v as CategoryUI }))}
                options={CATEGORY_OPTIONS_UI}
              />
              <Select
                label="Vehicle (required)"
                value={form.vehicleId}
                onChange={(v) => setForm((f) => ({ ...f, vehicleId: v }))}
                options={["", ...vehicles.map((v) => v.id)]}
                renderOption={(val) => (val === "" ? "— Select Vehicle —" : vehicleLabel(vehicles.find((x) => x.id === val)))}
              />
            </div>

            <ModalActions onCancel={() => setShowAdd(false)} submitText={saving ? "Saving…" : "Save Expense"} />
          </form>
        </Modal>
      )}

      {/* Edit Modal */}
      {showEdit && (
        <Modal onClose={() => setShowEdit(null)} title="Edit Expense">
          {formError && <ErrorBanner text={formError} />}
          <form className="space-y-4" onSubmit={submitEdit}>
            <Input
              label="Title"
              placeholder="e.g., Fuel, Road Tax, Maintenance"
              value={form.title}
              onChange={(v) => setForm((f) => ({ ...f, title: v }))}
              required
            />
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Input
                label="Amount (RM)"
                type="number"
                step="0.01"
                min="0"
                value={String(form.amount)}
                onChange={(v) => setForm((f) => ({ ...f, amount: v === "" ? "" : Number(v) }))}
                required
              />
              <Input
                label="Date"
                type="date"
                value={form.date}
                onChange={(v) => setForm((f) => ({ ...f, date: v }))}
                required
              />
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <Select
                label="Category"
                value={form.category}
                onChange={(v) => setForm((f) => ({ ...f, category: v as CategoryUI }))}
                options={CATEGORY_OPTIONS_UI}
              />
              <Select
                label="Vehicle (required)"
                value={form.vehicleId}
                onChange={(v) => setForm((f) => ({ ...f, vehicleId: v }))}
                options={["", ...vehicles.map((v) => v.id)]}
                renderOption={(val) => (val === "" ? "— Select Vehicle —" : vehicleLabel(vehicles.find((x) => x.id === val)))}
              />
            </div>

            <ModalActions onCancel={() => setShowEdit(null)} submitText={saving ? "Saving…" : "Update Expense"} />
          </form>
        </Modal>
      )}

      {/* Delete Modal */}
      {confirmDelete && (
        <Modal onClose={() => setConfirmDelete(null)} title="Delete Expense">
          <div className="text-white/80">
            Are you sure you want to delete <span className="font-semibold">{confirmDelete.title}</span>?
          </div>
          <div className="mt-4 flex justify-end gap-3">
            <button
              className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/20"
              onClick={() => setConfirmDelete(null)}
            >
              Cancel
            </button>
            <button
              className="px-4 py-2 rounded-xl bg-red-500/20 hover:bg-red-500/30 border border-red-500/40"
              onClick={submitDelete}
            >
              Delete
            </button>
          </div>
        </Modal>
      )}
    </div>
  );
}

/* ----------------------------- UI bits ------------------------------------ */

function Modal({
  children,
  title,
  onClose,
}: {
  children: React.ReactNode;
  title: string;
  onClose: () => void;
}) {
  React.useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-[60]">
      <div className="absolute inset-0 bg-black/50" onClick={onClose} />
      <div className="absolute left-1/2 top-16 -translate-x-1/2 w-[min(680px,92vw)]">
        <div className="rounded-2xl bg-white/10 border border-white/15 backdrop-blur-xl shadow-2xl">
          <div className="px-5 py-4 border-b border-white/10 text-lg font-semibold">{title}</div>
          <div className="max-h-[70vh] overflow-y-auto p-5">{children}</div>
        </div>
      </div>
    </div>
  );
}

function ModalActions({
  submitText = "Save",
  onCancel,
}: {
  submitText?: string;
  onCancel: () => void;
}) {
  return (
    <div className="pt-2 flex justify-end gap-3">
      <button
        type="button"
        className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/20"
        onClick={onCancel}
      >
        Cancel
      </button>
      <button type="submit" className="px-4 py-2 rounded-xl bg-indigo-500/80 hover:bg-indigo-500 text-white">
        {submitText}
      </button>
    </div>
  );
}

function Input({
  label,
  type = "text",
  value,
  onChange,
  placeholder,
  required,
  step,
  min,
}: {
  label: string;
  type?: "text" | "number" | "date";
  value: string;
  onChange: (val: string) => void;
  placeholder?: string;
  required?: boolean;
  step?: string;
  min?: string | number;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm text-white/80">{label}</label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        required={required}
        step={step}
        min={min}
        className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
      />
    </div>
  );
}

function Select<T extends string>({
  label,
  value,
  onChange,
  options,
  renderOption,
}: {
  label: string;
  value: T;
  onChange: (v: T) => void;
  options: ReadonlyArray<T>;          // ← accept readonly arrays too
  renderOption?: (v: T) => React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-1.5">
      <label className="text-sm text-white/80">{label}</label>
      <select
        className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
        value={value}
        onChange={(e) => onChange(e.target.value as T)}
      >
        {options.map((opt) => (
          <option key={String(opt)} value={opt}>
            {renderOption ? renderOption(opt) : String(opt)}
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