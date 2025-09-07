import axios from "axios";
import FormData from "form-data";

const baseURL = process.env.ML_BASE_URL || "http://127.0.0.1:8001";
const timeout = Number(process.env.ML_TIMEOUT_MS || 20000);

const client = axios.create({ baseURL, timeout });

export async function mlHealth() {
  const { data } = await client.get("/health");
  return data;
}

export async function mlPredict({ text }) {
  const { data } = await client.post("/predict", { text });
  return data; // { label, score }
}

export async function mlExtractText(fileBuffer, filename, mimetype) {
  const form = new FormData();
  form.append("file", fileBuffer, { filename, contentType: mimetype });
  const { data } = await client.post("/extract-text", form, {
    headers: form.getHeaders(),
    maxBodyLength: Infinity,
  });
  return data; // { filename, size, text }
}

export async function mlTrain(payload) {
  const { data } = await client.post("/train", payload);
  return data; // { jobId }
}

export async function mlJobStatus(jobId) {
  const { data } = await client.get(`/jobs/${encodeURIComponent(jobId)}`);
  return data; // { jobId, status }
}