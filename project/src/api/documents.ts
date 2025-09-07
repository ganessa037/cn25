import { api } from "./index";
import type { Doc } from "../types";

export type ListParams = {
  vehicleId?: string;
  status?: string;  // uploaded/processing/validated/failed
  q?: string;       // 关键字搜索（若后端支持）
};

export async function listDocuments(params?: ListParams): Promise<Doc[]> {
  const { data } = await api.get<Doc[]>("/documents", { params });
  return data;
}

export type UploadPayload = {
  file: File;             // 必填：与后端 multer 对应字段名 file
  name?: string;
  docType: string;        // 例如 'License' | 'Insurance' ...（会转小写发给后端）
  vehicleId?: string;
  expiryDate?: string;    // yyyy-mm-dd
  extractedText?: string; // 可选：OCR/用户备注
};

// 支持进度回调（0-100）
export async function uploadDocument(
  payload: UploadPayload,
  onProgress?: (pct: number) => void
): Promise<Doc> {
  const fd = new FormData();
  fd.append("file", payload.file); // 字段名必须叫 file（匹配后端 upload.single('file')）
  if (payload.name) fd.append("name", payload.name);
  if (payload.docType) fd.append("documentType", payload.docType.toLowerCase());
  if (payload.vehicleId) fd.append("vehicleId", payload.vehicleId);
  if (payload.expiryDate) fd.append("expiryDate", payload.expiryDate);
  if (payload.extractedText) fd.append("extractedText", payload.extractedText);

  const { data } = await api.post<Doc>("/documents", fd, {
    headers: { "Content-Type": "multipart/form-data" },
    onUploadProgress: (e) => {
      if (!onProgress || !e.total) return;
      onProgress(Math.round((e.loaded / e.total) * 100));
    },
  });
  return data;
}

export async function updateDocument(id: string, patch: Partial<Doc>): Promise<Doc> {
  const { data } = await api.put<Doc>(`/documents/${id}`, patch);
  return data;
}

export async function deleteDocument(id: string): Promise<void> {
  await api.delete(`/documents/${id}`);
}