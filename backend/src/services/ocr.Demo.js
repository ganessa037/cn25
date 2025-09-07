import fs from "fs/promises";
import pdf from "pdf-parse";
import Tesseract from "tesseract.js";

/**
 * Demo OCR/Text extraction:
 *  - PDFs → pdf-parse (fast, no model download)
 *  - Images (PNG/JPG/WebP) → tesseract.js (first run may download the ENG model)
 */
export async function extractTextFromUpload(file) {
  try {
    // Use in-memory buffer if available, else read from disk (multer diskStorage)
    let buffer = file?.buffer;
    if (!buffer && file?.path) buffer = await fs.readFile(file.path);
    if (!buffer) return null;

    const mime = file.mimetype || "";

    // PDF text extraction (preferred for demo)
    if (mime === "application/pdf") {
      const out = await pdf(buffer);
      return (out.text || "").trim() || null;
    }

    // Image OCR (may take a few seconds the first time)
    if (mime.startsWith("image/")) {
      const { data } = await Tesseract.recognize(buffer, "eng");
      const text = (data?.text || "").trim();
      return text || null;
    }

    return null;
  } catch (_err) {
    // For demo, never fail the request on OCR error
    return null;
  }
}