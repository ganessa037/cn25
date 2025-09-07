// backend/src/services/ocrDemo.js
import fs from "fs/promises";
import pdf from "pdf-parse";
import Tesseract from "tesseract.js";

/**
 * Demo OCR/Text extraction:
 *  - PDFs → pdf-parse (fast)
 *  - Images (PNG/JPG/WebP) → tesseract.js (first run may take a few seconds)
 * Returns a trimmed string or null. Never throws (safe for demos).
 */
export async function extractTextFromUpload(file) {
  try {
    let buffer = file?.buffer;
    if (!buffer && file?.path) buffer = await fs.readFile(file.path);
    if (!buffer) return null;

    const mime = file.mimetype || "";

    // PDF extraction
    if (mime === "application/pdf") {
      const out = await pdf(buffer);
      return (out.text || "").trim() || null;
    }

    // Image OCR
    if (mime.startsWith("image/")) {
      const { data } = await Tesseract.recognize(buffer, "eng");
      const text = (data?.text || "").trim();
      return text || null;
    }

    return null;
  } catch {
    return null;
  }
}
