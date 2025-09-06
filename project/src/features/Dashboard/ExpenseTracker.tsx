import React, { useMemo, useRef, useState } from "react";
import {
  Plus, Pencil, Trash2, Search, Car, Wrench, Droplet, Shield, ReceiptText,
  BadgeDollarSign, Store, X, Upload
} from "lucide-react";
import { GlassButton, GlassCard, GlassPanel } from "../../components/ui/Glass";
import type { Vehicle } from "../../pages/Dashboard";

/** ===== Types ===== */
export type ExpenseCategory =
  | "Fuel"
  | "Maintenance"
  | "Insurance"
  | "Road Tax"
  | "Toll/Parking"
  | "Other";

export type Expense = {
  id: string;
  vehicleId: string;
  title: string;            // e.g. "Shell Station, Damansara"
  category: ExpenseCategory;
  amount: number;           // RM
  date: string;             // YYYY-MM-DD
  description?: string;
  receiptBase64?: string;   // optional uploaded receipt
};

export interface ExpenseTrackerProps {
  expenses: Expense[];
  setExpenses: React.Dispatch<React.SetStateAction<Expense[]>>;
  vehicles: Vehicle[];
}

/** ===== Helpers ===== */
const CATEGORIES: ExpenseCategory[] = [
  "Fuel",
  "Maintenance",
  "Insurance",
  "Road Tax",
  "Toll/Parking",
  "Other",
];

const rm = (n: number) =>
  `RM${n.toLocaleString("en-MY", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

const fmtDate = (s?: string) => {
  if (!s) return "—";
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString("en-US");
};

const IconByCategory: React.FC<{ cat: ExpenseCategory }> = ({ cat }) => {
  const base = "w-10 h-10 rounded-xl flex items-center justify-center";
  switch (cat) {
    case "Fuel":
      return <div className={`${base} bg-emerald-400/15`}><Droplet className="w-5 h-5 text-emerald-300" /></div>;
    case "Maintenance":
      return <div className={`${base} bg-indigo-400/15`}><Wrench className="w-5 h-5 text-indigo-300" /></div>;
    case "Insurance":
      return <div className={`${base} bg-rose-400/15`}><Shield className="w-5 h-5 text-rose-300" /></div>;
    case "Road Tax":
      return <div className={`${base} bg-amber-400/15`}><ReceiptText className="w-5 h-5 text-amber-300" /></div>;
    case "Toll/Parking":
      return <div className={`${base} bg-teal-400/15`}><BadgeDollarSign className="w-5 h-5 text-teal-300" /></div>;
    default:
      return <div className={`${base} bg-slate-400/15`}><Store className="w-5 h-5 text-slate-300" /></div>;
  }
};

const vehicleLabel = (vehicles: Vehicle[], id?: string) => {
  if (!id) return undefined;
  const v = vehicles.find((x: any) => x.id === id) as any;
  if (!v) return undefined;
  const title = [v.brand, v.model].filter(Boolean).join(" ");
  const plate = v.plate ? `(${v.plate})` : "";
  return title ? `${title} ${plate}`.trim() : v.name || v.plate || id;
};

const fileToBase64 = (file: File) =>
  new Promise<string>((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result));
    r.onerror = () => rej(r.error);
    r.readAsDataURL(file);
  });

/** ===== Component ===== */
export default function ExpenseTracker({ expenses, setExpenses, vehicles }: ExpenseTrackerProps) {
  /** KPIs */
  const now = new Date();
  const year = now.getFullYear();
  const month = now.getMonth(); // 0-11

  const { total, thisMonth, count, avgPerMonth } = useMemo(() => {
    const total = expenses.reduce((s, e) => s + (e.amount || 0), 0);
    const thisMonth = expenses.reduce((s, e) => {
      const d = new Date(e.date);
      return d.getFullYear() === year && d.getMonth() === month ? s + (e.amount || 0) : s;
    }, 0);
    const count = expenses.length;

    // Avg per month over active months between min and max date (inclusive)
    if (count === 0) return { total, thisMonth, count, avgPerMonth: 0 };
    const timestamps = expenses.map((e) => new Date(e.date).getTime()).filter((n) => !Number.isNaN(n));
    const minT = Math.min(...timestamps);
    const maxT = Math.max(...timestamps);
    const minD = new Date(minT);
    const maxD = new Date(maxT);
    const months =
      (maxD.getFullYear() - minD.getFullYear()) * 12 + (maxD.getMonth() - minD.getMonth()) + 1;
    const avgPerMonth = months > 0 ? total / months : total;
    return { total, thisMonth, count, avgPerMonth };
  }, [expenses, month, year]);

  /** Filters & sorting */
  const [q, setQ] = useState("");
  const [catFilter, setCatFilter] = useState<ExpenseCategory | "All">("All");
  const [vehFilter, setVehFilter] = useState<string>("all");
  const [sortKey, setSortKey] = useState<"date_desc" | "date_asc" | "amount_desc" | "amount_asc">(
    "date_desc"
  );

  const filtered = useMemo(() => {
    let list = expenses.filter((e) => {
      if (catFilter !== "All" && e.category !== catFilter) return false;
      if (vehFilter !== "all" && e.vehicleId !== vehFilter) return false;
      if (q) {
        const hay = `${e.title} ${e.description || ""}`.toLowerCase();
        if (!hay.includes(q.toLowerCase())) return false;
      }
      return true;
    });
    list = list.sort((a, b) => {
      if (sortKey === "date_desc") return new Date(b.date).getTime() - new Date(a.date).getTime();
      if (sortKey === "date_asc") return new Date(a.date).getTime() - new Date(b.date).getTime();
      if (sortKey === "amount_desc") return (b.amount || 0) - (a.amount || 0);
      return (a.amount || 0) - (b.amount || 0);
    });
    return list;
  }, [expenses, q, catFilter, vehFilter, sortKey]);

  /** Modal state */
  const [open, setOpen] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

  type Form = {
    vehicleId: string;
    category: ExpenseCategory | "";
    title: string;
    amount: string;     // string for input; we cast on submit
    date: string;       // YYYY-MM-DD
    description: string;
    receipt?: File | null;
  };

  const emptyForm: Form = {
    vehicleId: "",
    category: "",
    title: "",
    amount: "",
    date: "",
    description: "",
    receipt: null,
  };

  const [form, setForm] = useState<Form>(emptyForm);
  const [touched, setTouched] = useState<Record<keyof Form, boolean>>({
    vehicleId: false,
    category: false,
    title: false,
    amount: false,
    date: false,
    description: false,
    receipt: false,
  });
  const [submitted, setSubmitted] = useState(false);

  const err = useMemo(() => {
    const errors: Partial<Record<keyof Form, string>> = {};
    if (!form.vehicleId) errors.vehicleId = "Required";
    if (!form.category) errors.category = "Required";
    if (!form.title.trim()) errors.title = "Required";
    const amt = Number(form.amount);
    if (!(form.amount.trim() && !Number.isNaN(amt) && amt > 0)) errors.amount = "Must be > 0";
    if (!form.date) errors.date = "Required";
    return errors;
  }, [form]);

  const isValid = Object.keys(err).length === 0;

  const onOpenAdd = () => {
    setEditingId(null);
    setForm(emptyForm);
    setTouched({
      vehicleId: false, category: false, title: false, amount: false, date: false, description: false, receipt: false,
    });
    setSubmitted(false);
    setOpen(true);
  };

  const onOpenEdit = (e: Expense) => {
    setEditingId(e.id);
    setForm({
      vehicleId: e.vehicleId,
      category: e.category,
      title: e.title,
      amount: e.amount.toString(),
      date: e.date,
      description: e.description || "",
      receipt: null,
    });
    setTouched({
      vehicleId: false, category: false, title: false, amount: false, date: false, description: false, receipt: false,
    });
    setSubmitted(false);
    setOpen(true);
  };

  const closeModal = () => !submitting && setOpen(false);

  const onPick: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (f.size > 10 * 1024 * 1024) {
      alert("File too large (max 10MB)");
      e.currentTarget.value = "";
      return;
    }
    setForm((p) => ({ ...p, receipt: f }));
  };

  const saveExpense = async () => {
    setSubmitted(true);
    if (!isValid) return;

    setSubmitting(true);
    try {
      const base64 = form.receipt ? await fileToBase64(form.receipt) : undefined;
      const payload: Expense = {
        id: editingId ?? crypto.randomUUID(),
        vehicleId: form.vehicleId,
        category: form.category as ExpenseCategory,
        title: form.title.trim(),
        amount: Number(form.amount),
        date: form.date,
        description: form.description.trim() || undefined,
        receiptBase64: base64,
      };

      setExpenses((prev) => {
        if (editingId) {
          return prev.map((x) => (x.id === editingId ? payload : x));
        }
        return [payload, ...prev];
      });

      setOpen(false);
      setEditingId(null);
      setForm(emptyForm);
    } finally {
      setSubmitting(false);
    }
  };

  const remove = (id: string) => setExpenses((prev) => prev.filter((e) => e.id !== id));

  /** UI helpers */
  const errClass = (k: keyof Form) =>
    (submitted || touched[k]) && (err as any)[k] ? " ring-2 ring-red-400 border-red-400" : "";

  /** ===== Render ===== */
  return (
    <section className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Expense Tracking</h3>
          <p className="text-white/60">Track and categorize your vehicle expenses</p>
        </div>
        <GlassButton onClick={onOpenAdd} className="flex items-center gap-2">
          <Plus className="w-4 h-4" /> Add Expense
        </GlassButton>
      </div>

      {/* KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <GlassCard className="glass-hover">
          <div className="text-white/70 text-sm">Total Expenses</div>
          <div className="text-2xl font-semibold mt-1">{rm(total)}</div>
        </GlassCard>
        <GlassCard className="glass-hover">
          <div className="text-white/70 text-sm">This Month</div>
          <div className="text-2xl font-semibold mt-1">{rm(thisMonth)}</div>
        </GlassCard>
        <GlassCard className="glass-hover">
          <div className="text-white/70 text-sm">Total Records</div>
          <div className="text-2xl font-semibold mt-1">{count}</div>
        </GlassCard>
        <GlassCard className="glass-hover">
          <div className="text-white/70 text-sm">Avg/Month</div>
          <div className="text-2xl font-semibold mt-1">{rm(avgPerMonth || 0)}</div>
        </GlassCard>
      </div>

      {/* Search & filters */}
      <div className="flex flex-col lg:flex-row gap-3 items-stretch">
        <div className="relative flex-1">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-white/50" />
          <input
            className="glass-input pl-9 w-full"
            placeholder="Search expenses..."
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
        <select
          className="glass-input lg:w-56"
          value={catFilter}
          onChange={(e) => setCatFilter(e.target.value as ExpenseCategory | "All")}
        >
          <option value="All">All Categories</option>
          {CATEGORIES.map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <select
          className="glass-input lg:w-56"
          value={vehFilter}
          onChange={(e) => setVehFilter(e.target.value)}
        >
          <option value="all">All Vehicles</option>
          {vehicles.map((v) => (
            <option key={(v as any).id} value={(v as any).id}>
              {(v as any).name || (v as any).plate || (v as any).id}
            </option>
          ))}
        </select>
        <select
          className="glass-input lg:w-44"
          value={sortKey}
          onChange={(e) => setSortKey(e.target.value as any)}
        >
          <option value="date_desc">Sort by Date</option>
          <option value="date_asc">Date ↑</option>
          <option value="amount_desc">Amount ↓</option>
          <option value="amount_asc">Amount ↑</option>
        </select>
      </div>

      {/* List */}
      {filtered.length === 0 ? (
        <GlassPanel className="min-h-[140px] flex items-center justify-center text-white/60">
          No expenses yet
        </GlassPanel>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {filtered.map((e) => (
            <div key={e.id} className="glass-card glass-hover p-4">
              <div className="flex items-start justify-between gap-4">
                {/* Left */}
                <div className="flex items-start gap-3 min-w-0">
                  <IconByCategory cat={e.category} />
                  <div className="min-w-0">
                    <div className="font-semibold text-base truncate">{e.title}</div>
                    <div className="text-white/70 flex items-center gap-2 flex-wrap">
                      <span>{e.category}</span>
                      <span className="inline-flex items-center gap-1">
                        • <Car className="w-4 h-4" /> {vehicleLabel(vehicles, e.vehicleId) || "—"}
                      </span>
                      <span>• {fmtDate(e.date)}</span>
                    </div>
                    {e.description && (
                      <div className="text-white/60 text-sm mt-1 line-clamp-2">{e.description}</div>
                    )}
                  </div>
                </div>

                {/* Right */}
                <div className="flex items-center gap-3 shrink-0">
                  <div className="text-lg font-semibold">{rm(e.amount)}</div>
                  <button className="glass-btn p-2" aria-label="Edit" onClick={() => onOpenEdit(e)}>
                    <Pencil className="w-4 h-4" />
                  </button>
                  <button className="glass-btn p-2" aria-label="Delete" onClick={() => remove(e.id)}>
                    <Trash2 className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* ===== Modal (Add / Edit) ===== */}
      {open && (
        <div className="fixed inset-0 z-50 overflow-y-auto overscroll-contain">
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={closeModal} />
          <div className="relative min-h-full flex items-start justify-center py-10">
            <div className="w-[min(980px,92vw)]" role="dialog" aria-modal="true">
              <GlassPanel className="relative">
                <button
                  onClick={closeModal}
                  className="absolute right-4 top-4 p-2 rounded-lg hover:bg-white/10"
                  aria-label="Close"
                >
                  <X className="w-4 h-4" />
                </button>

                <div className="mb-6">
                  <h4 className="text-xl font-semibold">
                    {editingId ? "Edit Expense" : "Add New Expense"}
                  </h4>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-white/70 mb-1">Vehicle</label>
                    <select
                      className={`glass-input w-full${errClass("vehicleId")}`}
                      value={form.vehicleId}
                      onChange={(e) => setForm((p) => ({ ...p, vehicleId: e.target.value }))}
                      onBlur={() => setTouched((t) => ({ ...t, vehicleId: true }))}
                      required
                    >
                      <option value="">Select Vehicle</option>
                      {vehicles.map((v: any) => (
                        <option key={v.id} value={v.id}>
                          {v.name || v.plate || v.id}
                        </option>
                      ))}
                    </select>
                    {(submitted || touched.vehicleId) && err.vehicleId && (
                      <div className="mt-1 text-sm text-red-400">{err.vehicleId}</div>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm text-white/70 mb-1">Category</label>
                    <select
                      className={`glass-input w-full${errClass("category")}`}
                      value={form.category}
                      onChange={(e) =>
                        setForm((p) => ({ ...p, category: e.target.value as ExpenseCategory }))
                      }
                      onBlur={() => setTouched((t) => ({ ...t, category: true }))}
                      required
                    >
                      <option value="">Select Category</option>
                      {CATEGORIES.map((c) => (
                        <option key={c} value={c}>
                          {c}
                        </option>
                      ))}
                    </select>
                    {(submitted || touched.category) && err.category && (
                      <div className="mt-1 text-sm text-red-400">{err.category}</div>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm text-white/70 mb-1">Amount (RM)</label>
                    <input
                      inputMode="decimal"
                      className={`glass-input w-full${errClass("amount")}`}
                      placeholder="0.00"
                      value={form.amount}
                      onChange={(e) => setForm((p) => ({ ...p, amount: e.target.value }))}
                      onBlur={() => setTouched((t) => ({ ...t, amount: true }))}
                      required
                    />
                    {(submitted || touched.amount) && err.amount && (
                      <div className="mt-1 text-sm text-red-400">{err.amount}</div>
                    )}
                  </div>

                  <div>
                    <label className="block text-sm text-white/70 mb-1">Date</label>
                    <input
                      type="date"
                      className={`glass-input w-full${errClass("date")}`}
                      value={form.date}
                      onChange={(e) => setForm((p) => ({ ...p, date: e.target.value }))}
                      onBlur={() => setTouched((t) => ({ ...t, date: true }))}
                      required
                    />
                    {(submitted || touched.date) && err.date && (
                      <div className="mt-1 text-sm text-red-400">{err.date}</div>
                    )}
                  </div>

                  <div className="md:col-span-2">
                    <label className="block text-sm text-white/70 mb-1">Title</label>
                    <input
                      className={`glass-input w-full${errClass("title")}`}
                      placeholder="Shell Station, Damansara"
                      value={form.title}
                      onChange={(e) => setForm((p) => ({ ...p, title: e.target.value }))}
                      onBlur={() => setTouched((t) => ({ ...t, title: true }))}
                      required
                    />
                    {(submitted || touched.title) && err.title && (
                      <div className="mt-1 text-sm text-red-400">{err.title}</div>
                    )}
                  </div>

                  <div className="md:col-span-2">
                    <label className="block text-sm text-white/70 mb-1">Description</label>
                    <input
                      className="glass-input w-full"
                      placeholder="Brief description of the expense"
                      value={form.description}
                      onChange={(e) => setForm((p) => ({ ...p, description: e.target.value }))}
                    />
                  </div>

                  <div className="md:col-span-2">
                    <label className="block text-sm text-white/70 mb-1">Receipt</label>
                    <input ref={fileRef} type="file" hidden onChange={onPick} />
                    <GlassButton onClick={() => fileRef.current?.click()} className="flex items-center gap-2">
                      <Upload className="w-4 h-4" /> Scan Receipt
                    </GlassButton>
                    {form.receipt && (
                      <div className="text-white/70 text-sm mt-2">{form.receipt.name}</div>
                    )}
                  </div>
                </div>

                <div className="mt-6 flex items-center justify-end gap-3">
                  <GlassButton onClick={closeModal}>Cancel</GlassButton>
                  <GlassButton
                    onClick={saveExpense}
                    className={`${(!isValid || submitting) ? "opacity-60 cursor-not-allowed" : ""}`}
                    aria-disabled={!isValid || submitting}
                  >
                    {submitting ? "Saving…" : editingId ? "Save Changes" : "Add Expense"}
                  </GlassButton>
                </div>
              </GlassPanel>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}