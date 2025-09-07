export type Vehicle = {
    id: string;
    plateNumber: string;
    make?: string;
    model?: string;
    year?: number;
    // 扩展字段（你的功能页里会用到）
    brand?: string;
    chassisNumber?: string;
    engineNumber?: string;
    color?: string;
    fuelType?: "Petrol" | "Diesel" | "Hybrid" | "EV";
    roadTaxExpiry?: string;     // yyyy-mm-dd
    insuranceExpiry?: string;   // yyyy-mm-dd
    currentMileageKm?: number;
    lastServiceDate?: string;   // yyyy-mm-dd
    nextServiceDueKm?: number;
  };
  
  export type DocStatus = "uploaded" | "processing" | "validated" | "failed";
  export type DocType =
    | "License" | "Insurance" | "Road Tax" | "Registration Card" | "Service Invoice" | "Others";
  
  export type Doc = {
    id: string;
    name: string;
    docType: DocType;
    status?: DocStatus;
    expiresAt?: string;     // yyyy-mm-dd
    vehicleId?: string;
    text?: string;          // 识别/备注
    fileName?: string;
    url?: string;           // 预览地址（可选）
  };
  
  export type ExpenseCategory = "Fuel" | "Maintenance" | "Insurance" | "Road Tax" | "Toll/Parking" | "Other";
  export type Expense = {
    id: string;
    vehicleId: string;
    title: string;
    category: ExpenseCategory;
    amount: number;
    date: string;           // yyyy-mm-dd
    description?: string;
    receiptBase64?: string;
  };