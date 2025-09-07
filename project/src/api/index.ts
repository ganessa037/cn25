import axios, { AxiosError } from "axios";

const base =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.VITE_BACKEND_URL
    ? `${String(import.meta.env.VITE_BACKEND_URL).replace(/\/$/, "")}/api`
    : "/api");

export const api = axios.create({
  baseURL: base,
  withCredentials: true,
});

// 请求：自动带上 JWT
api.interceptors.request.use((config) => {
  try {
    const raw = localStorage.getItem("user");
    const token = raw ? JSON.parse(raw)?.token : null;
    if (token) {
      config.headers = config.headers ?? {};
      config.headers.Authorization = `Bearer ${token}`;
    }
  } catch {}
  return config;
});

// 响应：401 统一跳转登录
api.interceptors.response.use(
  (res) => res,
  (err: AxiosError) => {
    const status = err.response?.status;
    if (status === 401) {
      try {
        localStorage.removeItem("user");
      } catch {}
      const next = encodeURIComponent(
        window.location.pathname + window.location.search + window.location.hash
      );
      window.location.assign(`/signin?next=${next}`);
    }
    return Promise.reject(err);
  }
);