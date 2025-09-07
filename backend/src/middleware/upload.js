import multer from "multer";
import fs from "fs";

const UPLOAD_DIR = "uploads";
try { fs.mkdirSync(UPLOAD_DIR, { recursive: true }); } catch {}

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, UPLOAD_DIR),
  filename: (_req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`),
});

function fileFilter(_req, file, cb) {
  const ok = ["image/jpeg","image/png","image/webp","application/pdf"].includes(file.mimetype);
  cb(ok ? null : new Error("Unsupported file type"), ok);
}

export const upload = multer({
  storage,
  fileFilter,
  limits: { fileSize: 5 * 1024 * 1024 }, // 5MB
});