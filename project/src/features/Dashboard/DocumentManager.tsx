import React, { useMemo, useRef, useState } from "react";
import {
  FileText, Upload, Trash2, Copy, Eye, X,
  CheckCircle2, AlertTriangle, Clock, Search, Car
} from "lucide-react";
import { GlassButton, GlassCard, GlassPanel } from "../../components/ui/Glass";
import type { Doc, Vehicle } from "../../pages/Dashboard";

/** ---------- Types & constants ---------- */
const DOC_TYPES = [
  "License",
  "Insurance",
  "Road Tax",
  "Registration Card",
  "Service Invoice",
  "Others",
] as const;
type DocType = typeof DOC_TYPES[number];

type ExtDoc = Doc & {
  type?: DocType;
  vehicleId?: string;
  uploadedAt?: string;     // ISO
  expiryDate?: string;     // YYYY-MM-DD
  extractedText?: string;  // OCR/文本
};

function fileToBase64(file: File): Promise<string> {
  return new Promise((res, rej) => {
    const r = new FileReader();
    r.onload = () => res(String(r.result));
    r.onerror = () => rej(r.error);
    r.readAsDataURL(file);
  });
}

function tryReadText(file: File): Promise<string> {
  if (file.type.startsWith("text/") || /\.(txt|json|csv|md)$/i.test(file.name)) {
    return file.text();
  }
  return Promise.resolve("");
}

const fmtDate = (s?: string) => {
  if (!s) return "—";
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? s : d.toLocaleDateString("en-US");
};

const statusOf = (expiry?: string) => {
  if (!expiry) return "none" as const;
  const today = new Date(); today.setHours(0, 0, 0, 0);
  const d = new Date(expiry); d.setHours(0, 0, 0, 0);
  if (d < today) return "expired" as const;
  const diff = d.getTime() - today.getTime();
  if (diff <= 30 * 24 * 3600 * 1000) return "expiring" as const;
  return "valid" as const;
};

const StatusBadge: React.FC<{ expiry?: string }> = ({ expiry }) => {
  const st = statusOf(expiry);
  if (st === "valid")
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs bg-emerald-500/20 text-emerald-300 border border-emerald-400/40">
        <CheckCircle2 className="w-3.5 h-3.5" /> Valid
      </span>
    );
  if (st === "expiring")
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs bg-amber-500/20 text-amber-300 border border-amber-400/40">
        <Clock className="w-3.5 h-3.5" /> Expiring Soon
      </span>
    );
  if (st === "expired")
    return (
      <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-xs bg-red-500/20 text-red-300 border border-red-400/40">
        <AlertTriangle className="w-3.5 h-3.5" /> Expired
      </span>
    );
  return null;
};

const DocTypeIcon: React.FC<{ t?: string }> = () => (
  <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center">
    <FileText className="w-5 h-5 text-white" />
  </div>
);

/** ---------- Component ---------- */
export interface DocumentManagerProps {
  documents: Doc[];
  setDocuments: React.Dispatch<React.SetStateAction<Doc[]>>;
  vehicles: Vehicle[];
}

export default function DocumentManager({ documents, setDocuments, vehicles }: DocumentManagerProps) {
  /** KPI */
  const kpi = useMemo(() => {
    const out = { total: documents.length, valid: 0, expiring: 0, expired: 0 };
    documents.forEach((d) => {
      const st = statusOf((d as ExtDoc).expiryDate);
      if (st === "valid") out.valid += 1;
      else if (st === "expiring") out.expiring += 1;
      else if (st === "expired") out.expired += 1;
    });
    return out;
  }, [documents]);
  const { total, valid, expiring, expired } = kpi;

  /** Filters */
  const [q, setQ] = useState("");
  const [typeFilter, setTypeFilter] = useState<DocType | "All">("All");
  const [vehFilter, setVehFilter] = useState<string>("all");

  const filtered = useMemo(() => {
    return documents.filter((d) => {
      const ed = d as ExtDoc;
      if (typeFilter !== "All" && ed.type !== typeFilter) return false;
      if (vehFilter !== "all" && ed.vehicleId !== vehFilter) return false;
      if (q) {
        const hay = `${d.name || ""} ${(ed.type || "")} ${(ed.extractedText || "")}`.toLowerCase();
        if (!hay.includes(q.toLowerCase())) return false;
      }
      return true;
    });
  }, [documents, q, typeFilter, vehFilter]);

  /** Modals */
  const [openAdd, setOpenAdd] = useState(false);
  const [openFull, setOpenFull] = useState<ExtDoc | null>(null);

  /** Add form */
  const fileRef = useRef<HTMLInputElement>(null);
  const [adding, setAdding] = useState(false);
  const [form, setForm] = useState<{
    docType?: DocType;
    name: string;
    vehicleId?: string;
    expiryDate?: string;
    file?: File | null;
    text?: string;
  }>({ name: "" });

  const canSubmit = useMemo(
    () => !!form.docType && !!form.name.trim() && !!form.file,
    [form.docType, form.name, form.file]
  );

  const onPick: React.ChangeEventHandler<HTMLInputElement> = async (e) => {
    const f = e.target.files?.[0];
    if (!f) return;
    if (f.size > 10 * 1024 * 1024) {
      alert("File too large (max 10MB)");
      e.currentTarget.value = "";
      return;
    }
    setForm((prev) => ({ ...prev, file: f }));
    const t = await tryReadText(f);
    if (t) setForm((prev) => ({ ...prev, text: t }));
  };

  const addDocument = async () => {
    if (!canSubmit) return;
    setAdding(true);
    try {
      const base64 = await fileToBase64(form.file!);
      const now = new Date().toISOString();
      const doc: ExtDoc = {
        id: crypto.randomUUID(),
        name: form.name.trim(),
        type: form.docType!,
        vehicleId: form.vehicleId || undefined,
        expiryDate: form.expiryDate || undefined,
        uploadedAt: now,
        size: form.file!.size,
        contentBase64: base64,
        extractedText: form.text || "",
      };
      setDocuments((prev) => [doc, ...prev]);
      setOpenAdd(false);
      setForm({ name: "" });
    } finally {
      setAdding(false);
    }
  };

  const remove = (id: string) => setDocuments((prev) => prev.filter((d) => d.id !== id));

  const copyText = async (txt: string) => {
    try {
      await navigator.clipboard.writeText(txt);
    } catch {
      const ta = document.createElement("textarea");
      ta.value = txt; document.body.appendChild(ta); ta.select();
      document.execCommand("copy"); document.body.removeChild(ta);
    }
  };

  const vehicleLabel = (id?: string) => {
    if (!id) return undefined;
    const v = vehicles.find((x) => x.id === id) as any;
    if (!v) return undefined;
    const title = [v.brand, v.model].filter(Boolean).join(" ");
    const plate = v.plate ? `(${v.plate})` : "";
    return title ? `${title} ${plate}`.trim() : v.name || v.plate || id;
  };

  return (
    <section className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Document Management</h3>
          <p className="text-white/60">Store and manage important vehicle documents with AI text extraction</p>
        </div>
        <GlassButton onClick={() => setOpenAdd(true)} className="flex items-center gap-2">
          <Upload className="w-4 h-4" /> Add Document
        </GlassButton>
      </div>

      {/* KPI */}
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-4">
        <GlassCard className="glass-hover">
          <div className="text-white/70 text-sm">Total Documents</div>
          <div className="text-2xl font-semibold mt-1">{total}</div>
        </GlassCard>
        <GlassCard className="glass-hover">
          <div className="text-white/70 text-sm">Valid</div>
          <div className="text-2xl font-semibold mt-1">{valid}</div>
        </GlassCard>
        <GlassCard className="glass-hover">
          <div className="text-white/70 text-sm">Expiring</div>
          <div className="text-2xl font-semibold mt-1">{expiring}</div>
        </GlassCard>
        <GlassCard className="glass-hover">
          <div className="text-white/70 text-sm">Expired</div>
          <div className="text-2xl font-semibold mt-1">{expired}</div>
        </GlassCard>
      </div>

      {/* Search & filters */}
      <div className="flex flex-col md:flex-row gap-3 items-stretch">
        <div className="relative flex-1">
          <Search className="w-4 h-4 absolute left-3 top-1/2 -translate-y-1/2 text-white/50" />
          <input
            className="glass-input pl-9 w-full"
            placeholder="Search documents..."
            value={q}
            onChange={(e) => setQ(e.target.value)}
          />
        </div>
        <select className="glass-input md:w-56" value={typeFilter} onChange={(e) => setTypeFilter(e.target.value as DocType | "All")}>
          <option value="All">All Types</option>
          {DOC_TYPES.map((dt) => (
            <option key={dt} value={dt}>{dt}</option>
          ))}
        </select>
        <select className="glass-input md:w-56" value={vehFilter} onChange={(e) => setVehFilter(e.target.value)}>
          <option value="all">All Vehicles</option>
          {vehicles.map((v) => (
            <option key={v.id} value={v.id}>{v.name || v.plate || v.id}</option>
          ))}
        </select>
      </div>

      {/* List */}
      {filtered.length === 0 ? (
        <GlassPanel className="min-h-[160px] flex items-center justify-center text-white/60">
          No documents yet
        </GlassPanel>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {filtered.map((d) => {
            const ed = d as ExtDoc;
            const veh = vehicleLabel(ed.vehicleId);
            return (
              <div key={d.id} className="glass-card glass-hover p-4">
                {/* header */}
                <div className="flex items-start justify-between gap-4">
                  <div className="flex items-start gap-3 min-w-0">
                    <DocTypeIcon t={ed.type} />
                    <div className="min-w-0">
                      <div className="font-semibold text-lg truncate">{d.name}</div>
                      <div className="text-white/70 flex items-center gap-2 flex-wrap">
                        <span>{ed.type || "Document"}</span>
                        {veh && (
                          <span className="inline-flex items-center gap-1">
                            • <Car className="w-4 h-4" /> {veh}
                          </span>
                        )}
                      </div>
                      <div className="text-white/60 text-sm mt-1 flex gap-4 flex-wrap">
                        <span>Uploaded: {fmtDate(ed.uploadedAt)}</span>
                        <span>Expires: {fmtDate(ed.expiryDate)}</span>
                      </div>
                    </div>
                  </div>
                  <div className="shrink-0">
                    <StatusBadge expiry={ed.expiryDate} />
                  </div>
                </div>

                {/* preview */}
                <div className="mt-4">
                  <div className="flex items-center justify-between mb-2">
                    <div className="text-white/70 font-medium">Extracted Text Preview</div>
                    {ed.extractedText && (
                      <button
                        className="text-xs text-white/70 hover:text-white"
                        onClick={() => copyText(ed.extractedText || "")}
                      >
                        Copy All
                      </button>
                    )}
                  </div>
                  <div className="rounded-xl bg-white/5 px-3 py-2 text-white/80 whitespace-pre-wrap max-h-24 overflow-hidden">
                    {ed.extractedText ? ed.extractedText : "No text extracted."}
                  </div>
                </div>

                {/* actions */}
                <div className="mt-3 flex items-center gap-5 text-sm">
                  <button className="inline-flex items-center gap-1 text-white/80 hover:text-white"
                          onClick={() => setOpenFull(ed)}>
                    <Eye className="w-4 h-4" /> View Full Text
                  </button>
                  {ed.extractedText && (
                    <button className="inline-flex items-center gap-1 text-white/80 hover:text-white"
                            onClick={() => copyText(ed.extractedText || "")}>
                      <Copy className="w-4 h-4" /> Copy Text
                    </button>
                  )}
                  <button className="inline-flex items-center gap-1 text-red-300 hover:text-red-200"
                          onClick={() => remove(d.id)}>
                    <Trash2 className="w-4 h-4" /> Delete
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Add modal */}
      {openAdd && (
        <div className="fixed inset-0 z-50 overflow-y-auto overscroll-contain">
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpenAdd(false)} />
          <div className="relative min-h-full flex items-start justify-center py-10">
            <div className="w-[min(980px,92vw)]" role="dialog" aria-modal="true">
              <GlassPanel className="relative">
                <button
                  onClick={() => setOpenAdd(false)}
                  className="absolute right-4 top-4 p-2 rounded-lg hover:bg-white/10"
                  aria-label="Close"
                >
                  <X className="w-4 h-4" />
                </button>

                <div className="mb-6">
                  <h4 className="text-xl font-semibold">Add New Document</h4>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm text-white/70 mb-1">Document Type</label>
                    <select
                      className={`glass-input w-full ${!form.docType && "ring-1 ring-red-400/60"}`}
                      value={form.docType || ""}
                      onChange={(e) => setForm((p) => ({ ...p, docType: e.target.value as DocType }))}
                      required
                    >
                      <option value="">Select Document Type</option>
                      {DOC_TYPES.map((dt) => <option key={dt} value={dt}>{dt}</option>)}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm text-white/70 mb-1">Vehicle</label>
                    <select
                      className="glass-input w-full"
                      value={form.vehicleId || ""}
                      onChange={(e) => setForm((p) => ({ ...p, vehicleId: e.target.value || undefined }))}
                    >
                      <option value="">Select Vehicle (Optional)</option>
                      {vehicles.map((v) => (
                        <option key={v.id} value={v.id}>{v.name || v.plate || v.id}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm text-white/70 mb-1">Document Name</label>
                    <input
                      className={`glass-input w-full ${!form.name.trim() && "ring-1 ring-red-400/60"}`}
                      placeholder="e.g., Driving License – John Doe"
                      value={form.name}
                      onChange={(e) => setForm((p) => ({ ...p, name: e.target.value }))}
                      required
                    />
                  </div>

                  <div>
                    <label className="block text-sm text-white/70 mb-1">Expiry Date (Optional)</label>
                    <input
                      type="date"
                      className="glass-input w-full"
                      value={form.expiryDate || ""}
                      onChange={(e) => setForm((p) => ({ ...p, expiryDate: e.target.value || undefined }))}
                    />
                  </div>

                  <div className="md:col-span-2">
                    <label className="block text-sm text-white/70 mb-1">Upload Document</label>
                    <div className="rounded-xl border border-dashed border-white/20 bg-white/5 px-4 py-10 text-center">
                      <input ref={fileRef} type="file" hidden onChange={onPick} />
                      <div className="space-y-2">
                        <Upload className="w-6 h-6 mx-auto text-white/70" />
                        <div className="text-white/80">
                          {form.file ? <b>{form.file.name}</b> : "Click to upload or drag and drop"}
                        </div>
                        <div className="text-white/50 text-sm">PNG, JPG, PDF up to 10MB</div>
                        <div>
                          <GlassButton onClick={() => fileRef.current?.click()} className="mt-2">Choose File</GlassButton>
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="md:col-span-2">
                    <label className="block text-sm text-white/70 mb-1">Extracted Text (Optional)</label>
                    <textarea
                      className="glass-input w-full min-h-[120px]"
                      placeholder="Paste OCR text if available…"
                      value={form.text || ""}
                      onChange={(e) => setForm((p) => ({ ...p, text: e.target.value }))}
                    />
                  </div>
                </div>

                <div className="mt-6 flex items-center justify-end gap-3">
                  <GlassButton onClick={() => setOpenAdd(false)}>Cancel</GlassButton>
                  <GlassButton
                    onClick={addDocument}
                    className={`${(!canSubmit || adding) ? "opacity-60 cursor-not-allowed" : ""}`}
                    aria-disabled={!canSubmit || adding}
                  >
                    {adding ? "Adding…" : "Add Document"}
                  </GlassButton>
                </div>
              </GlassPanel>
            </div>
          </div>
        </div>
      )}

      {/* Full text modal */}
      {openFull && (
        <div className="fixed inset-0 z-50 overflow-y-auto overscroll-contain">
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpenFull(null)} />
          <div className="relative min-h-full flex items-start justify-center py-10">
            <div className="w-[min(760px,92vw)]" role="dialog" aria-modal="true">
              <GlassPanel className="relative">
                <button
                  onClick={() => setOpenFull(null)}
                  className="absolute right-4 top-4 p-2 rounded-lg hover:bg-white/10"
                  aria-label="Close"
                >
                  <X className="w-4 h-4" />
                </button>

                <h4 className="text-xl font-semibold mb-4">{openFull.name}</h4>
                <div className="rounded-xl bg-white/5 px-3 py-3 text-white/80 whitespace-pre-wrap">
                  {openFull.extractedText || "No text extracted."}
                </div>

                <div className="mt-4 flex items-center justify-end gap-3">
                  <GlassButton onClick={() => copyText(openFull.extractedText || "")} className="flex items-center gap-2">
                    <Copy className="w-4 h-4" /> Copy All Text
                  </GlassButton>
                  <GlassButton onClick={() => setOpenFull(null)}>Close</GlassButton>
                </div>
              </GlassPanel>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}