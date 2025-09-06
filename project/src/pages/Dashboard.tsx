import { useEffect, useState } from "react";
import { VehiclesAPI, DocumentsAPI, ExpensesAPI } from "../api/index"; //
import VehicleManager from "../features/Dashboard/VehicleManager";
import DocumentManager from "../features/Dashboard/DocumentManager";
import ExpenseTracker from "../features/Dashboard/ExpenseTracker";
import Notifications from "../features/Dashboard/Notifications";
import Analytics from "../features/Dashboard/Analytics";

export default function Dashboard() {
  const [vehicles, setVehicles] = useState<any[]>([]);
  const [documents, setDocuments] = useState<any[]>([]);
  const [expenses, setExpenses] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // 首次加载：从后端取数
  useEffect(() => {
    (async () => {
      try {
        const [vs, ds, es] = await Promise.all([
          VehiclesAPI.list(),
          DocumentsAPI.list(),
          ExpensesAPI.list(),
        ]);
        setVehicles(Array.isArray(vs) ? vs : []);
        setDocuments(Array.isArray(ds) ? ds : []);
        // 后端枚举是 Toll_Parking，前端里大多写成 "Toll/Parking"，这里统一成前端显示值
        const norm = (x: any) => ({ ...x, category: x?.category === "Toll_Parking" ? "Toll/Parking" : x?.category });
        setExpenses(Array.isArray(es) ? es.map(norm) : []);
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) {
    return <div className="p-6 text-white/70">Loading…</div>;
  }

  return (
    <div className="space-y-6">
      <Analytics vehicles={vehicles} expenses={expenses} />
      {/* 现有组件的 props 形态：保留 setXxx，保证不改 UI、不动类型 */}
      <VehicleManager vehicles={vehicles} setVehicles={setVehicles} />
      <DocumentManager documents={documents} setDocuments={setDocuments} vehicles={vehicles} />
      <ExpenseTracker expenses={expenses} setExpenses={setExpenses} vehicles={vehicles} />
      <Notifications vehicles={vehicles} />
    </div>
  );
}