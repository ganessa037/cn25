import multer from "multer";
const storage = multer.diskStorage({
  destination: (_req, _file, cb) => cb(null, "uploads"),
  filename: (_req, file, cb) => cb(null, `${Date.now()}-${file.originalname}`)
});
function fileFilter(_req, file, cb) {
  const ok = ["image/jpeg","image/png","application/pdf","image/webp"].includes(file.mimetype);
  cb(ok ? null : new Error("Unsupported file type"), ok);
}
export const upload = multer({ storage, fileFilter, limits: { fileSize: 5*1024*1024 } });