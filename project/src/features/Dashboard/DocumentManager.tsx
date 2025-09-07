import * as React from "react";

/**
 * - Aligned to Prisma "Document" table and DocType enum.
 * - Upload uses multipart/form-data with field name "file".
 * - All requests include Authorization: Bearer <token> from localStorage.user.
 * - Glass UI + scrollable modal, consistent with the rest of the app.
 */

/* ----------------------------- Types ----------------------------- */

type DocType =
  | "License"
  | "Insurance"
  | "RoadTax"
  | "RegistrationCard"
  | "ServiceInvoice"
  | "Others";

type DocumentRow = {
  id: string;
  userId: string;
  vehicleId?: string | null;
  type?: DocType | null;
  name: string;
  expiryDate?: string | null;
  uploadedAt: string;
  size: number;
  contentBase64?: string | null;
  extractedText?: string | null;
  createdAt?: string;
  updatedAt?: string;
};

type Vehicle = {
  id: string;
  brand: string;
  model?: string | null;
  year?: number | null;
  plate?: string | null;
  color?: string | null;
  fuelType?: string | null;
};

type ValidityState = "valid" | "expiring" | "expired" | "none";

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

async function apiJson<T>(
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

async function apiUpload<T>(path: string, form: FormData): Promise<T> {
  const token = getToken();
  const headers: Record<string, string> = {};
  if (token) headers["Authorization"] = `Bearer ${token}`;
  // Do not set Content-Type manually; browser will set the boundary.
  const res = await fetch(`${API_BASE}${path}`, { method: "POST", headers, body: form });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText} — ${text || "Upload failed"}`);
  }
  return (await res.json()) as T;
}

/* ----------------------------- Helpers --------------------------- */

// Date input (YYYY-MM-DD) -> DB timestamp without tz (YYYY-MM-DDT00:00:00.000)
function toDbTimestamp(localDate: string | ""): string | null {
  if (!localDate) return null;
  const [y, m, d] = localDate.split("-").map(Number);
  if (!y || !m || !d) return null;
  const dt = new Date(y, m - 1, d, 0, 0, 0);
  const yyyy = dt.getFullYear();
  const mm = String(dt.getMonth() + 1).padStart(2, "0");
  const dd = String(dt.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}T00:00:00.000`;
}
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

function validity(dateIso?: string | null): "valid" | "expiring" | "expired" | "none" {
  if (!dateIso) return "none";
  const d = new Date(dateIso);
  if (isNaN(+d)) return "none";
  const today = new Date();
  const a = new Date(d.getFullYear(), d.getMonth(), d.getDate()).getTime();
  const b = new Date(today.getFullYear(), today.getMonth(), today.getDate()).getTime();
  if (a < b) return "expired";
  const THIRTY = 30 * 24 * 60 * 60 * 1000;
  return a - b <= THIRTY ? "expiring" : "valid";
}

function sizeLabel(n: number) {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)} MB`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)} KB`;
  return `${n} B`;
}

/* -------------------------- Main Component ----------------------- */

export default function DocumentManager() {
  const [docs, setDocs] = React.useState<DocumentRow[]>([]);
  const [vehicles, setVehicles] = React.useState<Vehicle[]>([]);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState<string | null>(null);

  // Filters
  const [q, setQ] = React.useState("");
  const [typeFilter, setTypeFilter] = React.useState<DocType | "All">("All");
  const [vehicleFilter, setVehicleFilter] = React.useState<string | "All">("All");

  // Modal state
  const [showAdd, setShowAdd] = React.useState(false);
  const [showTextDoc, setShowTextDoc] = React.useState<DocumentRow | null>(null);

  // Add form state
  const [form, setForm] = React.useState<{
    type: DocType | "";
    vehicleId: string;
    name: string;
    expiryDate: string;
    file: File | null;
  }>({
    type: "",
    vehicleId: "",
    name: "",
    expiryDate: "",
    file: null,
  });

  const [uploading, setUploading] = React.useState(false);
  const [formError, setFormError] = React.useState<string | null>(null);

  const refetch = React.useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [rows, veh] = await Promise.all([
        apiJson<DocumentRow[]>("GET", "/documents"),
        apiJson<Vehicle[]>("GET", "/vehicles"),
      ]);
      setDocs(rows || []);
      setVehicles(veh || []);
    } catch (err: any) {
      setError(err?.message || "Failed to load documents");
    } finally {
      setLoading(false);
    }
  }, []);

  React.useEffect(() => {
    refetch();
  }, [refetch]);

  /* ---------------------------- Derived ---------------------------- */

  const vehicleLabel = React.useMemo(() => {
    const map = new Map<string, string>();
    vehicles.forEach((v) => {
      const base = `${v.brand ?? "Vehicle"}${v.model ? " " + v.model : ""}`.trim();
      map.set(v.id, v.plate ? `${base} (${v.plate})` : base);
    });
    return map;
  }, [vehicles]);

  const filtered = React.useMemo(() => {
    const qlc = q.trim().toLowerCase();
    return docs
      .filter((d) => (typeFilter === "All" ? true : d.type === typeFilter))
      .filter((d) => (vehicleFilter === "All" ? true : d.vehicleId === vehicleFilter))
      .filter((d) => {
        if (!qlc) return true;
        const hay = `${d.name ?? ""} ${d.type ?? ""} ${d.extractedText ?? ""}`.toLowerCase();
        return hay.includes(qlc);
      })
      .sort((a, b) => {
        const ad = new Date(a.uploadedAt || a.createdAt || 0).getTime();
        const bd = new Date(b.uploadedAt || b.createdAt || 0).getTime();
        return bd - ad;
      });
  }, [docs, q, typeFilter, vehicleFilter]);

  const stats = React.useMemo(() => {
    let total = docs.length;
    let valid = 0,
      expiring = 0,
      expired = 0;
    docs.forEach((d) => {
      const v = validity(d.expiryDate);
      if (v === "valid") valid++;
      else if (v === "expiring") expiring++;
      else if (v === "expired") expired++;
    });
    return { total, valid, expiring, expired };
  }, [docs]);

  /* --------------------------- Handlers --------------------------- */

  function openAdd() {
    setForm({ type: "", vehicleId: "", name: "", expiryDate: "", file: null });
    setFormError(null);
    setShowAdd(true);
  }

  async function submitAdd(e: React.FormEvent) {
    e.preventDefault();
    setFormError(null);

    if (!form.type) return setFormError("Please select a document type.");
    if (!form.name.trim()) return setFormError("Please enter a document name.");
    if (!form.file) return setFormError("Please choose a file to upload.");

    // Enforce 5MB (backend multer limit)
    if (form.file.size > 5 * 1024 * 1024) {
      return setFormError("File is too large. Max 5MB.");
    }

    setUploading(true);
    try {
      const fd = new FormData();
      fd.append("type", form.type);
      fd.append("name", form.name.trim());
      if (form.vehicleId) fd.append("vehicleId", form.vehicleId);
      const ts = toDbTimestamp(form.expiryDate);
      if (ts) fd.append("expiryDate", ts);
      fd.append("file", form.file);

      await apiUpload("/documents", fd);
      setShowAdd(false);
      await refetch();
    } catch (err: any) {
      setFormError(err?.message || "Failed to upload document");
    } finally {
      setUploading(false);
    }
  }

  async function handleDelete(id: string) {
    if (!confirm("Delete this document?")) return;
    try {
      await apiJson("DELETE", `/documents/${encodeURIComponent(id)}`);
      await refetch();
    } catch (err: any) {
      alert(err?.message || "Failed to delete document");
    }
  }

  function onFileDropped(files: FileList | null) {
    if (!files || files.length === 0) return;
    const f = files[0];
    setForm((prev) => ({ ...prev, file: f }));
  }

  /* ------------------------------- UI -------------------------------- */

  if (loading) {
    return (
      <div className="min-h-[calc(100vh-5rem)] grid place-items-center px-4 text-white bg-[radial-gradient(1200px_600px_at_50%_-200px,rgba(88,101,242,.35),rgba(2,8,23,1)_60%)]">
        <div className="max-w-md w-full bg-white/10 border border-white/10 rounded-2xl p-6 backdrop-blur-xl shadow-xl">
          Loading your documents…
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-[calc(100vh-5rem)] px-3 sm:px-6 text-white bg-[radial-gradient(1200px_600px_at_50%_-200px,rgba(88,101,242,.35),rgba(2,8,23,1)_60%)]">
      <div className="max-w-7xl mx-auto py-6 space-y-6">
        {/* Header + Add */}
        <div className="flex items-center justify-between gap-3">
          <div>
            <div className="text-2xl sm:text-3xl font-semibold tracking-tight">Document Management</div>
            <div className="text-white/70 text-sm">Store and manage vehicle documents with AI text extraction</div>
          </div>
          <button
            className="px-4 py-2 rounded-xl bg-white/15 hover:bg-white/20 border border-white/20 backdrop-blur-xl shadow-lg transition"
            onClick={openAdd}
          >
            + Add Document
          </button>
        </div>

        {/* Stats */}
        <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
          <StatCard label="Total Documents" value={stats.total} icon="📄" />
          <StatCard label="Valid" value={stats.valid} icon="✅" />
          <StatCard label="Expiring" value={stats.expiring} icon="⏳" />
          <StatCard label="Expired" value={stats.expired} icon="⚠️" />
        </div>

        {/* Search + Filters */}
        <div className="bg-white/10 border border-white/10 rounded-2xl p-4 backdrop-blur-xl shadow-xl">
          <div className="grid grid-cols-1 lg:grid-cols-[1fr_auto_auto] gap-3">
            <input
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              placeholder="Search documents…"
              value={q}
              onChange={(e) => setQ(e.target.value)}
            />
            <select
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value as any)}
            >
              <option>All</option>
              {DOCTYPES.map((t) => (
                <option key={t} value={t}>
                  {t}
                </option>
              ))}
            </select>
            <select
              className="rounded-xl bg-white/10 border border-white/20 px-3 py-2 outline-none focus:ring-2 focus:ring-indigo-400/50"
              value={vehicleFilter}
              onChange={(e) => setVehicleFilter(e.target.value)}
            >
              <option value="All">All Vehicles</option>
              {vehicles.map((v) => (
                <option key={v.id} value={v.id}>
                  {vehicleLabel.get(v.id)}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="bg-red-900/30 border border-red-500/30 rounded-2xl p-4 text-sm text-red-200">
            {error}
          </div>
        )}

        {/* List */}
        <div className="space-y-4">
          {filtered.length === 0 ? (
            <div className="bg-white/10 border border-white/10 rounded-2xl p-6 backdrop-blur-xl shadow-xl text-white/70">
              No documents found.
            </div>
          ) : (
            filtered.map((d) => {
              const vlabel = d.vehicleId ? vehicleLabel.get(d.vehicleId) : undefined;
              const val = validity(d.expiryDate);
              return (
                <div key={d.id} className="bg-white/10 border border-white/10 rounded-2xl p-5 backdrop-blur-xl shadow-xl">
                  {/* Title row */}
                  <div className="flex items-start justify-between gap-3">
                    <div className="space-y-1">
                      <div className="text-lg font-semibold">{d.name}</div>
                      <div className="text-white/70 text-sm">
                        {d.type ?? "—"}
                        {vlabel ? ` • ${vlabel}` : ""}
                      </div>
                      <div className="text-white/60 text-xs">
                        Uploaded: {fmtDate(d.uploadedAt)} • Expires: {fmtDate(d.expiryDate)} • Size: {sizeLabel(d.size || 0)}
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <ValidityPill state={val} />
                    </div>
                  </div>

                  {/* Extracted text preview */}
                  {d.extractedText && (
                    <div className="mt-4 bg-white/5 border border-white/10 rounded-xl p-3">
                      <div className="text-xs uppercase tracking-wide text-white/60 mb-1">
                        Extracted Text Preview
                      </div>
                      <p className="text-sm text-white/80 line-clamp-3 whitespace-pre-wrap">
                        {d.extractedText}
                      </p>
                      <div className="mt-3 flex items-center gap-4">
                        <button
                          className="text-sm px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20"
                          onClick={() => setShowTextDoc(d)}
                        >
                          View Full Text
                        </button>
                        <button
                          className="text-sm px-3 py-1.5 rounded-lg bg-white/10 hover:bg-white/15 border border-white/20"
                          onClick={() => navigator.clipboard.writeText(d.extractedText || "")}
                        >
                          Copy Text
                        </button>
                        <button
                          className="text-sm px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40"
                          onClick={() => handleDelete(d.id)}
                        >
                          Delete
                        </button>
                      </div>
                    </div>
                  )}

                  {!d.extractedText && (
                    <div className="mt-3 flex items-center gap-4">
                      <button
                        className="text-sm px-3 py-1.5 rounded-lg bg-red-500/20 hover:bg-red-500/30 border border-red-500/40"
                        onClick={() => handleDelete(d.id)}
                      >
                        Delete
                      </button>
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </div>

      {/* Add Modal */}
      {showAdd && (
        <Modal onClose={() => setShowAdd(false)} title="Add New Document">
          {formError && <ErrorBanner text={formError} />}
          <form className="space-y-6" onSubmit={submitAdd}>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <section className="space-y-3">
                <Select
                  label="Document Type"
                  value={form.type}
                  onChange={(v) => setForm((f) => ({ ...f, type: v as DocType }))}
                  options={DOCTYPES}
                  placeholder="Select Document Type"
                  required
                />
                <Input
                  label="Document Name"
                  placeholder="e.g., Driving License — John Doe"
                  value={form.name}
                  onChange={(v) => setForm((f) => ({ ...f, name: v }))}
                  required
                />
              </section>

              <section className="space-y-3">
                <Select
                  label="Vehicle"
                  value={form.vehicleId}
                  onChange={(v) => setForm((f) => ({ ...f, vehicleId: v }))}
                  options={vehicles.map((v) => ({ id: v.id, label: vehicleLabel.get(v.id)! }))}
                  placeholder="Select Vehicle (Optional)"
                />
                <Input
                  label="Expiry Date (Optional)"
                  type="date"
                  value={form.expiryDate}
                  onChange={(v) => setForm((f) => ({ ...f, expiryDate: v }))}
                />
              </section>
            </div>

            {/* Upload area */}
            <FileDropzone
              file={form.file}
              onFileSelected={(file) => setForm((f) => ({ ...f, file }))}
              onDropFiles={(fl) => onFileDropped(fl)}
            />

            <ModalActions
              onCancel={() => setShowAdd(false)}
              submitText={uploading ? "Uploading…" : "Add Document"}
            />
          </form>
        </Modal>
      )}

      {/* Full extracted text modal */}
      {showTextDoc && (
        <Modal onClose={() => setShowTextDoc(null)} title={showTextDoc.name}>
          <div className="text-white/70 text-sm mb-3">
            {showTextDoc.type ?? "—"} {showTextDoc.vehicleId ? `• ${vehicleLabel.get(showTextDoc.vehicleId)}` : ""}
          </div>
          <div className="bg-white/5 border border-white/10 rounded-xl p-4 max-h-[60vh] overflow-y-auto whitespace-pre-wrap">
            {showTextDoc.extractedText || "No text extracted."}
          </div>
          <div className="flex items-center justify-end gap-3 mt-4">
            <button
              className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/20"
              onClick={() => setShowTextDoc(null)}
            >
              Close
            </button>
            <button
              className="px-4 py-2 rounded-xl bg-white/20 hover:bg-white/25 border border-white/30"
              onClick={() => navigator.clipboard.writeText(showTextDoc.extractedText || "")}
            >
              Copy All
            </button>
          </div>
        </Modal>
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
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />
      <div className="relative z-10 w-full max-w-4xl bg-white/10 border border-white/15 rounded-2xl shadow-2xl backdrop-blur-xl p-6 max-h-[80vh] overflow-y-auto">
        <div className="text-xl font-semibold">{title}</div>
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
      <button
        type="button"
        onClick={onCancel}
        className="px-4 py-2 rounded-xl bg-white/10 hover:bg-white/15 border border-white/20 transition"
      >
        Cancel
      </button>
      <button
        type="submit"
        className="px-4 py-2 rounded-xl bg-white/20 hover:bg-white/25 border border-white/30 backdrop-blur-xl shadow-lg transition"
      >
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
}: {
  label: string;
  value: string;
  onChange: (v: string) => void;
  placeholder?: string;
  type?: string;
  required?: boolean;
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
  options: string[] | { id: string; label: string }[];
  placeholder?: string;
  required?: boolean;
}) {
  const isObject = typeof options[0] === "object";
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
        {(options as any[]).map((opt) =>
          isObject ? (
            <option key={(opt as any).id} value={(opt as any).id}>
              {(opt as any).label}
            </option>
          ) : (
            <option key={opt as any} value={opt as any}>
              {opt as any}
            </option>
          )
        )}
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

function StatCard({ label, value, icon }: { label: string; value: number; icon: string }) {
  return (
    <div className="bg-white/10 border border-white/10 rounded-2xl p-4 backdrop-blur-xl shadow-xl">
      <div className="text-sm text-white/70">{label}</div>
      <div className="mt-1 text-2xl font-semibold flex items-center gap-2">
        <span>{icon}</span> {value}
      </div>
    </div>
  );
}

function ValidityPill({ state }: { state: ValidityState }) {
  // If there's no state to show, render nothing
  if (state === "none") return null;

  // Map only the three displayable states
  const classMap: Record<Exclude<ValidityState, "none">, string> = {
    valid: "bg-emerald-500/20 border-emerald-500/40 text-emerald-100",
    expiring: "bg-amber-500/20 border-amber-500/40 text-amber-100",
    expired: "bg-red-500/20 border-red-500/40 text-red-200",
  };

  const label =
    state === "valid" ? "Valid" : state === "expiring" ? "Expiring Soon" : "Expired";

  return (
    <span className={`text-xs px-2 py-0.5 rounded-full border ${classMap[state]}`}>
      {label}
    </span>
  );
}

function FileDropzone({
  file,
  onFileSelected,
  onDropFiles,
}: {
  file: File | null;
  onFileSelected: (f: File | null) => void;
  onDropFiles: (fl: FileList | null) => void;
}) {
  const inputRef = React.useRef<HTMLInputElement | null>(null);
  const [dragOver, setDragOver] = React.useState(false);

  return (
    <div
      className={`rounded-xl border-2 border-dashed ${dragOver ? "border-white/60" : "border-white/20"} bg-white/5 p-8 text-center`}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        onDropFiles(e.dataTransfer.files);
      }}
      onClick={() => inputRef.current?.click()}
      role="button"
      tabIndex={0}
    >
      <input
        ref={inputRef}
        type="file"
        accept=".pdf,image/png,image/jpeg,image/jpg,image/webp"
        className="hidden"
        onChange={(e) => onFileSelected(e.target.files?.[0] || null)}
      />
      {file ? (
        <div className="text-white/80">
          <div className="font-medium">{file.name}</div>
          <div className="text-sm text-white/60">{sizeLabel(file.size)}</div>
          <div className="text-xs text-white/50 mt-1">Click to replace • Max 5MB</div>
        </div>
      ) : (
        <div className="text-white/70">
          <div className="text-2xl mb-2">⬆️</div>
          <div className="font-medium">Click to upload or drag and drop</div>
          <div className="text-xs text-white/60 mt-1">PNG, JPG, PDF up to 5MB</div>
        </div>
      )}
    </div>
  );
}

/* ------------------------------ Constants ---------------------------- */

const DOCTYPES: DocType[] = [
  "License",
  "Insurance",
  "RoadTax",
  "RegistrationCard",
  "ServiceInvoice",
  "Others",
];