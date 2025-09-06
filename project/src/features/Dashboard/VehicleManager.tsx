import React, { useEffect, useMemo, useRef, useState } from "react";
import {
  Car, Plus, Trash2, Pencil, Upload, Scan, X,
  Shield, Wrench, BadgeCheck
} from "lucide-react";
import { GlassButton, GlassCard, GlassPanel } from "../../components/ui/Glass";
import type { Vehicle } from "../../pages/Dashboard";

export interface VehicleManagerProps {
  vehicles: Vehicle[];
  setVehicles: React.Dispatch<React.SetStateAction<Vehicle[]>>;
}

type FullVehicle = Vehicle & {
  brand?: string;
  model?: string;
  chassisNumber?: string;
  engineNumber?: string;
  color?: string;
  fuelType?: "Petrol" | "Diesel" | "Hybrid" | "EV";
  roadTaxExpiry?: string;     // yyyy-mm-dd
  insuranceExpiry?: string;   // yyyy-mm-dd
  currentMileageKm?: number;
  lastServiceDate?: string;   // yyyy-mm-dd
  nextServiceDate?: string;   // yyyy-mm-dd
};

const BRANDS = [
  "Perodua","Proton","Toyota","Honda","Nissan","Mazda","Mitsubishi",
  "Hyundai","Kia","Volkswagen","BMW","Mercedes-Benz","Audi","Volvo",
  "BYD","Geely","Ford","Lexus","Subaru"
];

const fmtDate = (s?: string) => {
  if (!s) return "—";
  const d = new Date(s);
  if (Number.isNaN(d.getTime())) return s;
  return d.toLocaleDateString("en-US");
};
const isExpired = (s?: string) => {
  if (!s) return false;
  const d = new Date(s);
  const today = new Date();
  // 去掉时间部分，按日期比较
  d.setHours(0,0,0,0); today.setHours(0,0,0,0);
  return d < today;
};
const km = (n?: number) =>
  typeof n === "number" ? n.toLocaleString("en-US") : "0";

export default function VehicleManager({ vehicles, setVehicles }: VehicleManagerProps) {
  /** 列表内临时改名（保留，兼容旧交互） */
  const [editingId, setEditingId] = useState<string | null>(null);
  const [newName, setNewName] = useState("");

  /** 弹窗 */
  const [open, setOpen] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [editingVehicleId, setEditingVehicleId] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const currentYear = new Date().getFullYear();
  const emptyForm: FullVehicle = {
    id: "",
    name: "",
    brand: "",
    model: "",
    year: currentYear,
    plate: "",
    chassisNumber: "",
    engineNumber: "",
    color: "",
    fuelType: "Petrol",
    roadTaxExpiry: "",
    insuranceExpiry: "",
    currentMileageKm: 0,
    lastServiceDate: "",
    nextServiceDate: "",
  };

  const [form, setForm] = useState<FullVehicle>(emptyForm);

  /** 校验 */
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [submitted, setSubmitted] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const markTouched = (k: string) => setTouched((t) => ({ ...t, [k]: true }));

  const computedName = useMemo(() => {
    const parts = [form.brand, form.model].filter(Boolean).join(" ");
    return parts || form.plate || "Untitled Vehicle";
  }, [form.brand, form.model, form.plate]);

  /** 列表动作 */
  const remove = (id: string) => setVehicles(prev => prev.filter(v => v.id !== id));
  const startRename = (v: Vehicle) => { setEditingId(v.id); setNewName(v.name); };
  const saveRename = () => {
    if (!editingId) return;
    setVehicles(prev => prev.map(v => v.id === editingId ? { ...v, name: newName || v.name } : v));
    setEditingId(null);
  };

  /** 打开弹窗（新增/编辑） */
  const resetForm = () => { setForm(emptyForm); setTouched({}); setSubmitted(false); setErrors({}); };
  const openAdd = () => { setEditingVehicleId(null); resetForm(); setOpen(true); };
  const openEdit = (v: any) => { setEditingVehicleId(v.id); setForm({ ...emptyForm, ...v }); setTouched({}); setSubmitted(false); setErrors({}); setOpen(true); };
  const closeModal = () => { if (!submitting) setOpen(false); };

  /** 锁背景滚动 & ESC */
  useEffect(() => {
    if (!open) return;
    const original = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && closeModal();
    window.addEventListener("keydown", onKey);
    return () => {
      document.body.style.overflow = original;
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  /** 规则校验 */
  const validate = (v: FullVehicle) => {
    const err: Record<string, string> = {};
    const vinOk = v.chassisNumber ? /^[A-HJ-NPR-Z0-9]{17}$/i.test(v.chassisNumber) : false;
    const yearOk = typeof v.year === "number" && v.year >= 1980 && v.year <= currentYear + 1;
    const plateOk = (v.plate || "").trim().length >= 3;
    const engOk = !!(v.engineNumber && v.engineNumber.trim());
    const brandOk = !!(v.brand && v.brand.trim());
    const modelOk = !!(v.model && v.model.trim());
    const colorOk = !!(v.color && v.color.trim());
    const fuelOk = !!v.fuelType;
    const roadOk = !!v.roadTaxExpiry;
    const insOk = !!v.insuranceExpiry;
    const mileOk = typeof v.currentMileageKm === "number" && v.currentMileageKm >= 0;
    const lastOk = !!v.lastServiceDate;
    const nextOk = !!v.nextServiceDate;

    if (!brandOk) err.brand = "Required";
    if (!modelOk) err.model = "Required";
    if (!yearOk) err.year = "Invalid year";
    if (!plateOk) err.plate = "Required";
    if (!vinOk) err.chassisNumber = "VIN must be 17 characters (no I/O/Q)";
    if (!engOk) err.engineNumber = "Required";
    if (!colorOk) err.color = "Required";
    if (!fuelOk) err.fuelType = "Required";
    if (!roadOk) err.roadTaxExpiry = "Required";
    if (!insOk) err.insuranceExpiry = "Required";
    if (!mileOk) err.currentMileageKm = "Must be ≥ 0";
    if (!lastOk) err.lastServiceDate = "Required";
    if (!nextOk) err.nextServiceDate = "Required";

    if (lastOk && nextOk) {
      const last = new Date(v.lastServiceDate!);
      const next = new Date(v.nextServiceDate!);
      if (next < last) err.nextServiceDate = "Must be after last service date";
    }
    return err;
  };

  useEffect(() => { if (open) setErrors(validate(form)); }, [form, open]);
  const isValid = useMemo(() => Object.keys(validate(form)).length === 0, [form]);

  /** 提交（新增/编辑） */
  const saveVehicle = async () => {
    setSubmitted(true);
    const errNow = validate(form);
    setErrors(errNow);
    if (Object.keys(errNow).length > 0) {
      const firstKey = Object.keys(errNow)[0];
      const el = document.getElementById(`field-${firstKey}`);
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }

    setSubmitting(true);
    try {
      const payload: FullVehicle = {
        ...form,
        id: editingVehicleId ?? crypto.randomUUID(),
        name: [form.brand, form.model, form.year ? `(${form.year})` : ""].filter(Boolean).join(" ") || computedName,
      };

      setVehicles(prev => {
        if (editingVehicleId) {
          // 更新
          return prev.map(v => v.id === editingVehicleId ? { ...(payload as any) } : v);
        }
        // 新增
        return [{ ...(payload as any) }, ...prev];
      });

      closeModal();
      resetForm();
      setEditingVehicleId(null);
    } finally {
      setSubmitting(false);
    }
  };

  /** Scan Document 占位 */
  const onScan = () => fileRef.current?.click();
  const onPickFiles: React.ChangeEventHandler<HTMLInputElement> = (e) => {
    const files = Array.from(e.target.files || []);
    if (!files.length) return;
    console.log("Scanned files:", files.map(f => f.name));
    e.currentTarget.value = "";
  };

  /** 辅助：错误样式与提示 */
  const errClass = (k: string) =>
    (submitted || touched[k]) && errors[k] ? " ring-2 ring-red-400 border-red-400" : "";
  const help = (k: string) =>
    (submitted || touched[k]) && errors[k] ? (
      <div className="mt-1 text-sm text-red-400">{errors[k]}</div>
    ) : null;

  /** —— UI —— */
  return (
    <section className="space-y-4">
      {/* 顶部动作 */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold">Vehicle Management</h3>
          <p className="text-white/60">Manage your fleet with Malaysian regulatory compliance</p>
        </div>
        <div className="flex items-center gap-2">
          <input ref={fileRef} type="file" multiple hidden onChange={onPickFiles} />
          <GlassButton onClick={onScan} className="flex items-center gap-2">
            <Scan className="w-4 h-4" /> Scan Document
          </GlassButton>
          <GlassButton onClick={openAdd} className="flex items-center gap-2">
            <Plus className="w-4 h-4" /> Add Vehicle
          </GlassButton>
        </div>
      </div>

      {/* 列表：卡片改为“品牌 型号 (年份) + 状态三栏 + 里程” */}
      {vehicles.length === 0 ? (
        <GlassCard className="text-white/70">No vehicles yet</GlassCard>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          {vehicles.map((v: any) => {
            const title =
              [v.brand, v.model, v.year ? `(${v.year})` : ""].filter(Boolean).join(" ") || v.name;
            const subtitle = [v.plate, v.color, v.fuelType].filter(Boolean).join(" • ");

            const roadExpired = isExpired(v.roadTaxExpiry);
            const insExpired = isExpired(v.insuranceExpiry);
            const svcExpired = isExpired(v.nextServiceDate);

            return (
              <div key={v.id} className="glass-card glass-hover p-4">
                <div className="flex items-start justify-between gap-4">
                  {/* 左侧主信息 */}
                  <div className="flex items-start gap-3 min-w-0">
                    <div className="w-10 h-10 rounded-xl bg-white/10 flex items-center justify-center shrink-0">
                      <Car className="w-5 h-5 text-white" />
                    </div>
                    <div className="min-w-0">
                      <div className="font-semibold text-lg truncate">{title}</div>
                      <div className="text-white/70 truncate">{subtitle || "—"}</div>
                    </div>
                  </div>

                  {/* 右侧动作 */}
                  <div className="flex items-center gap-2 shrink-0">
                    {editingId === v.id ? (
                      <GlassButton onClick={saveRename}>Save</GlassButton>
                    ) : (
                      <>
                        <button className="glass-btn p-2" onClick={()=>openEdit(v)} aria-label="Edit">
                          <Pencil className="w-4 h-4" />
                        </button>
                        <button className="glass-btn p-2" onClick={()=>remove(v.id)} aria-label="Delete">
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </>
                    )}
                  </div>
                </div>

                {/* 三栏状态 */}
                <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                  {/* Road Tax */}
                  <div className="glass-card bg-white/5">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                        <BadgeCheck className="w-4 h-4" />
                      </span>
                      <span className="font-medium">Road Tax</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {roadExpired && (
                        <span className="px-2 py-0.5 rounded-md text-xs bg-red-500/20 text-red-300 border border-red-400/40">
                          Expired
                        </span>
                      )}
                      <span className="text-white/70">{fmtDate(v.roadTaxExpiry)}</span>
                    </div>
                  </div>

                  {/* Insurance */}
                  <div className="glass-card bg-white/5">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                        <Shield className="w-4 h-4" />
                      </span>
                      <span className="font-medium">Insurance</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {insExpired && (
                        <span className="px-2 py-0.5 rounded-md text-xs bg-red-500/20 text-red-300 border border-red-400/40">
                          Expired
                        </span>
                      )}
                      <span className="text-white/70">{fmtDate(v.insuranceExpiry)}</span>
                    </div>
                  </div>

                  {/* Next Service */}
                  <div className="glass-card bg-white/5">
                    <div className="flex items-center gap-2 mb-1">
                      <span className="w-8 h-8 rounded-lg bg-white/10 flex items-center justify-center">
                        <Wrench className="w-4 h-4" />
                      </span>
                      <span className="font-medium">Next Service</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {svcExpired && (
                        <span className="px-2 py-0.5 rounded-md text-xs bg-red-500/20 text-red-300 border border-red-400/40">
                          Expired
                        </span>
                      )}
                      <span className="text-white/70">{fmtDate(v.nextServiceDate)}</span>
                    </div>
                  </div>
                </div>

                {/* 里程条 */}
                <div className="mt-4 rounded-xl bg-white/5 px-3 py-2 text-white/80">
                  Current Mileage: <span className="font-semibold">{km(v.currentMileageKm)} km</span>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* ===== 可滚动弹窗（新增/编辑 + 校验） ===== */}
      {open && (
        <div className="fixed inset-0 z-50 overflow-y-auto overscroll-contain">
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm" onClick={closeModal} />
          <div className="relative min-h-full flex items-start justify-center py-10">
            <div className="w-[min(1100px,92vw)]" role="dialog" aria-modal="true">
              <GlassPanel className="relative">
                <button
                  onClick={closeModal}
                  className="absolute right-4 top-4 p-2 rounded-lg hover:bg-white/10"
                  aria-label="Close"
                >
                  <X className="w-4 h-4" />
                </button>

                <div className="mb-6">
                  <h4 className="text-xl font-semibold">Vehicle Management</h4>
                  <p className="text-white/60">Manage your fleet with Malaysian regulatory compliance</p>
                </div>

                <div className="flex items-center justify-end gap-2 mb-4">
                  <GlassButton onClick={onScan} className="flex items-center gap-2">
                    <Upload className="w-4 h-4" /> Scan Document
                  </GlassButton>
                </div>

                <div className="glass-card mb-4">
                  <h5 className="text-lg font-semibold mb-4">
                    {editingVehicleId ? "Edit Vehicle" : "Add New Vehicle"}
                  </h5>

                  {/* 表单 */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    {/* Basic Information */}
                    <div className="space-y-3">
                      <div className="text-white/70 font-medium mb-1">Basic Information</div>

                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-brand">Brand</label>
                        <select
                          id="field-brand"
                          className={`glass-input w-full${(submitted || touched["brand"]) && errors["brand"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          value={form.brand || ""}
                          onChange={(e)=>setForm({...form, brand: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, brand:true}))}
                          required
                        >
                          <option value="">Select Brand</option>
                          {BRANDS.map(b => <option key={b} value={b}>{b}</option>)}
                        </select>
                        {(submitted || touched["brand"]) && errors["brand"] && <div className="mt-1 text-sm text-red-400">{errors["brand"]}</div>}
                      </div>

                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-model">Model</label>
                        <input
                          id="field-model"
                          className={`glass-input w-full${(submitted || touched["model"]) && errors["model"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          placeholder="e.g., Vios, Accord"
                          value={form.model || ""}
                          onChange={(e)=>setForm({...form, model: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, model:true}))}
                          required
                        />
                        {(submitted || touched["model"]) && errors["model"] && <div className="mt-1 text-sm text-red-400">{errors["model"]}</div>}
                      </div>

                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-year">Year</label>
                        <input
                          id="field-year"
                          type="number"
                          className={`glass-input w-full${(submitted || touched["year"]) && errors["year"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          value={form.year || ""}
                          onChange={(e)=>setForm({...form, year: Number(e.target.value) as any})}
                          onBlur={()=>setTouched(t=>({...t, year:true}))}
                          required
                        />
                        {(submitted || touched["year"]) && errors["year"] && <div className="mt-1 text-sm text-red-400">{errors["year"]}</div>}
                      </div>

                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-plate">
                          Plate Number <span className="text-white/40 text-xs">(e.g., WA 1234 A)</span>
                        </label>
                        <input
                          id="field-plate"
                          className={`glass-input w-full${(submitted || touched["plate"]) && errors["plate"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          placeholder="WA 1234 A"
                          value={form.plate || ""}
                          onChange={(e)=>setForm({...form, plate: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, plate:true}))}
                          required
                        />
                        {(submitted || touched["plate"]) && errors["plate"] && <div className="mt-1 text-sm text-red-400">{errors["plate"]}</div>}
                      </div>
                    </div>

                    {/* Technical Details */}
                    <div className="space-y-3">
                      <div className="text-white/70 font-medium mb-1">Technical Details</div>

                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-chassisNumber">Chassis Number</label>
                        <input
                          id="field-chassisNumber"
                          className={`glass-input w-full${(submitted || touched["chassisNumber"]) && errors["chassisNumber"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          placeholder="17-digit chassis number"
                          value={form.chassisNumber || ""}
                          onChange={(e)=>setForm({...form, chassisNumber: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, chassisNumber:true}))}
                          required
                        />
                        {(submitted || touched["chassisNumber"]) && errors["chassisNumber"] && <div className="mt-1 text-sm text-red-400">{errors["chassisNumber"]}</div>}
                      </div>

                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-engineNumber">Engine Number</label>
                        <input
                          id="field-engineNumber"
                          className={`glass-input w-full${(submitted || touched["engineNumber"]) && errors["engineNumber"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          placeholder="Engine identification number"
                          value={form.engineNumber || ""}
                          onChange={(e)=>setForm({...form, engineNumber: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, engineNumber:true}))}
                          required
                        />
                        {(submitted || touched["engineNumber"]) && errors["engineNumber"] && <div className="mt-1 text-sm text-red-400">{errors["engineNumber"]}</div>}
                      </div>

                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-color">Color</label>
                        <input
                          id="field-color"
                          className={`glass-input w-full${(submitted || touched["color"]) && errors["color"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          placeholder="e.g., Silver, White, Black"
                          value={form.color || ""}
                          onChange={(e)=>setForm({...form, color: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, color:true}))}
                          required
                        />
                        {(submitted || touched["color"]) && errors["color"] && <div className="mt-1 text-sm text-red-400">{errors["color"]}</div>}
                      </div>

                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-fuelType">Fuel Type</label>
                        <select
                          id="field-fuelType"
                          className={`glass-input w-full${(submitted || touched["fuelType"]) && errors["fuelType"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          value={form.fuelType || "Petrol"}
                          onChange={(e)=>setForm({...form, fuelType: e.target.value as any})}
                          onBlur={()=>setTouched(t=>({...t, fuelType:true}))}
                          required
                        >
                          <option>Petrol</option>
                          <option>Diesel</option>
                          <option>Hybrid</option>
                          <option>EV</option>
                        </select>
                        {(submitted || touched["fuelType"]) && errors["fuelType"] && <div className="mt-1 text-sm text-red-400">{errors["fuelType"]}</div>}
                      </div>
                    </div>
                  </div>

                  {/* 合规&维护 */}
                  <div className="mt-6 space-y-3">
                    <div className="text-white/70 font-medium">Compliance & Maintenance</div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-roadTaxExpiry">Road Tax Expiry</label>
                        <input
                          id="field-roadTaxExpiry"
                          type="date"
                          className={`glass-input w-full${(submitted || touched["roadTaxExpiry"]) && errors["roadTaxExpiry"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          value={form.roadTaxExpiry || ""}
                          onChange={(e)=>setForm({...form, roadTaxExpiry: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, roadTaxExpiry:true}))}
                          required
                        />
                        {(submitted || touched["roadTaxExpiry"]) && errors["roadTaxExpiry"] && <div className="mt-1 text-sm text-red-400">{errors["roadTaxExpiry"]}</div>}
                      </div>
                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-insuranceExpiry">Insurance Expiry</label>
                        <input
                          id="field-insuranceExpiry"
                          type="date"
                          className={`glass-input w-full${(submitted || touched["insuranceExpiry"]) && errors["insuranceExpiry"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          value={form.insuranceExpiry || ""}
                          onChange={(e)=>setForm({...form, insuranceExpiry: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, insuranceExpiry:true}))}
                          required
                        />
                        {(submitted || touched["insuranceExpiry"]) && errors["insuranceExpiry"] && <div className="mt-1 text-sm text-red-400">{errors["insuranceExpiry"]}</div>}
                      </div>
                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-currentMileageKm">Current Mileage (km)</label>
                        <input
                          id="field-currentMileageKm"
                          type="number"
                          className={`glass-input w-full${(submitted || touched["currentMileageKm"]) && errors["currentMileageKm"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          value={form.currentMileageKm ?? 0}
                          onChange={(e)=>setForm({...form, currentMileageKm: Number(e.target.value)})}
                          onBlur={()=>setTouched(t=>({...t, currentMileageKm:true}))}
                          required
                          min={0}
                        />
                        {(submitted || touched["currentMileageKm"]) && errors["currentMileageKm"] && <div className="mt-1 text-sm text-red-400">{errors["currentMileageKm"]}</div>}
                      </div>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-lastServiceDate">Last Service Date</label>
                        <input
                          id="field-lastServiceDate"
                          type="date"
                          className={`glass-input w-full${(submitted || touched["lastServiceDate"]) && errors["lastServiceDate"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          value={form.lastServiceDate || ""}
                          onChange={(e)=>setForm({...form, lastServiceDate: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, lastServiceDate:true}))}
                          required
                        />
                        {(submitted || touched["lastServiceDate"]) && errors["lastServiceDate"] && <div className="mt-1 text-sm text-red-400">{errors["lastServiceDate"]}</div>}
                      </div>
                      <div>
                        <label className="block text-sm text-white/70 mb-1" htmlFor="field-nextServiceDate">Next Service Date</label>
                        <input
                          id="field-nextServiceDate"
                          type="date"
                          className={`glass-input w-full${(submitted || touched["nextServiceDate"]) && errors["nextServiceDate"] ? " ring-2 ring-red-400 border-red-400" : ""}`}
                          value={form.nextServiceDate || ""}
                          onChange={(e)=>setForm({...form, nextServiceDate: e.target.value})}
                          onBlur={()=>setTouched(t=>({...t, nextServiceDate:true}))}
                          required
                        />
                        {(submitted || touched["nextServiceDate"]) && errors["nextServiceDate"] && <div className="mt-1 text-sm text-red-400">{errors["nextServiceDate"]}</div>}
                      </div>
                    </div>
                  </div>

                  {/* footer */}
                  <div className="mt-6 flex items-center justify-end gap-3">
                    <GlassButton onClick={closeModal} className="px-5">Cancel</GlassButton>
                    <GlassButton
                      onClick={saveVehicle}
                      className={`px-5 ${(!isValid || submitting) ? "opacity-60 cursor-not-allowed" : ""}`}
                      aria-disabled={!isValid || submitting}
                    >
                      {submitting ? "Saving…" : (editingVehicleId ? "✓ Save Changes" : "✓ Add Vehicle")}
                    </GlassButton>
                  </div>
                </div>
              </GlassPanel>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}