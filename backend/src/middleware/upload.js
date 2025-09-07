import multer from "multer";


// Disk (kept for anything that really needs files on disk)
const disk = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, "uploads"),
  filename: (_req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
});
function fileFilter(_req, file, cb) {
  const ok = ["image/jpeg","image/png","application/pdf","image/webp"].includes(file.mimetype);
  cb(ok ? null : new Error("Unsupported file type"), ok);
}
export const uploadDisk = multer({ storage: disk, fileFilter, limits: { fileSize: 10 * 1024 * 1024 } });

// Memory (needed for OCR → we need Buffer)
const memory = multer.memoryStorage();
export const uploadMemory = multer({ storage: memory, fileFilter, limits: { fileSize: 10 * 1024 * 1024 } });

// Back-compat: if other code imports `upload`, keep it pointing to disk
export const upload = uploadDisk;