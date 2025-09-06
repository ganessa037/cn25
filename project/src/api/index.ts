const BASE = import.meta.env.VITE_API_URL as string;

function getAuthHeaders(): Record<string, string> {
  const headers: Record<string, string> = { "Content-Type": "application/json" };

  try {
    const raw = localStorage.getItem("user");
    const token = raw ? JSON.parse(raw)?.token : "";
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    } else {
      // 开发阶段的兜底，后端中间件已支持
      headers["x-demo-user-id"] = "demo-user-1";
    }
  } catch {
    headers["x-demo-user-id"] = "demo-user-1";
  }

  return headers;
}

async function request<T = any>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    method,
    headers: getAuthHeaders(),
    body: body ? JSON.stringify(body) : undefined,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "");
    throw new Error(text || `${res.status} ${res.statusText}`);
  }
  // No Content
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

// --- API 分组 --- //
export const VehiclesAPI = {
  list: () => request("GET", "/vehicles"),
  create: (data: any) => request("POST", "/vehicles", data),
  update: (id: string, data: any) => request("PUT", `/vehicles/${id}`, data),
  remove: (id: string) => request("DELETE", `/vehicles/${id}`),
};

export const DocumentsAPI = {
  list: () => request("GET", "/documents"),
  create: (data: any) => request("POST", "/documents", data),
  update: (id: string, data: any) => request("PUT", `/documents/${id}`, data),
  remove: (id: string) => request("DELETE", `/documents/${id}`),
};

export const ExpensesAPI = {
  list: () => request("GET", "/expenses"),
  create: (data: any) => request("POST", "/expenses", data),
  update: (id: string, data: any) => request("PUT", `/expenses/${id}`, data),
  remove: (id: string) => request("DELETE", `/expenses/${id}`),
};